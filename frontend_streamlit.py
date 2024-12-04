import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_option('client.showErrorDetails', False)

# List of available stocks (these are the ones we have models for)
AVAILABLE_STOCKS = [
    "TATAMOTORS", "DRREDDY", "CIPLA", "HINDUNILVR",
    "BAJFINANCE", "BHARTIARTL", "HINDALCO", "HDFCBANK", "TCS",
    "TATASTEEL", "INFY", "ITC", "RELIANCE", "MARUTI"
]

def load_stock_data(symbol, years=10):
    """Load historical stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    # Add .NS suffix for NSE stocks
    ticker = f"{symbol}.NS"
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def prepare_sequence_data(data, sequence_length=60):
    """Prepare sequence data for LSTM prediction"""
    # Calculate required technical indicators
    data['Returns'] = data['Close'].pct_change()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Fill NaN values
    data = data.fillna(method='bfill')

    # Select and order features to match the model's expected input
    selected_features = ['Close', 'RSI', 'Volatility', 'MA20', 'MA50', 'Returns']
    feature_data = data[selected_features].values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)

    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:(i + sequence_length)])

    return np.array(sequences), scaler

def generate_predictions(model, last_sequence, scaler, days_ahead):
    """Generate predictions for specified number of days"""
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days_ahead):
        # Reshape input for prediction (batch_size, timesteps, features)
        model_input = current_sequence.reshape((1, current_sequence.shape[0], current_sequence.shape[1]))

        # Predict next value
        next_pred = model.predict(model_input, verbose=0)

        # Create a complete feature set for the prediction
        # We'll use the last known values for technical indicators
        last_features = current_sequence[-1].copy()
        last_features[0] = next_pred[0, 0]  # Update only the Close price

        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = last_features

        # Store the predicted close price
        predictions.append(next_pred[0, 0])

    # Inverse transform predictions
    # Create a dummy array with all features to match scaler's expected shape
    dummy_features = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy_features[:, 0] = predictions  # Fill only the Close price column
    predictions_transformed = scaler.inverse_transform(dummy_features)[:, 0]

    return predictions_transformed

def plot_stock_history(df, stock_name):
    """Create interactive stock price plot"""
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))

    # Add volume bar chart
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    # Update layout
    fig.update_layout(
        title=f'{stock_name} Historical Data',
        yaxis_title='Price (₹)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig

def plot_predictions(df, stock_name, predictions_dict, chart_type="Candlestick"):
    """Plot historical data with predictions"""
    fig = go.Figure()

    # Plot historical data based on chart type
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Historical Data'
        ))
    else:  # Line chart
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Historical Price',
            line=dict(color='blue')
        ))

    # Add volume bar chart
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    # Plot predictions for different time horizons
    colors = {'30d': 'red', '60d': 'orange', '120d': 'green'}
    last_date = df.index[-1]

    for period, pred_values in predictions_dict.items():
        future_dates = pd.date_range(
            start=last_date,
            periods=len(pred_values) + 1,
            freq='D'
        )[1:]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred_values,
            name=f'{period} Prediction',
            line=dict(color=colors[period], dash='dash')
        ))

    fig.update_layout(
        title=f'{stock_name} Price Predictions',
        yaxis_title='Price (₹)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_title='Date',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis_rangeslider_visible=False,  # Disable range slider
        dragmode='zoom',  # Enable zoom by default
        selectdirection='h'  # Horizontal selection
    )
    # Add modebar buttons for both chart types
    fig.update_layout(
        modebar=dict(
            add=[
                'select2d',
                'lasso2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d'
            ]
        )
    )

    return fig

def display_lstm_predictions(df, stock_name, chart_type):
    """Display LSTM-based predictions"""
    current_price = df['Close'].iloc[-1]
    predictions = {}
    
    try:
        model = load_model(f'lstm_model_{stock_name.lower()}.keras')
        sequence_data, scaler = prepare_sequence_data(df)
        last_sequence = sequence_data[-1]
        
        # Generate predictions for different time horizons
        col1, col2, col3 = st.columns(3)
        for days, col in zip([30, 60, 120], [col1, col2, col3]):
            pred = generate_predictions(model, last_sequence, scaler, days)
            predictions[f'{days}d'] = pred
            
            # Display prediction metrics
            final_price = pred[-1]
            change_pct = ((final_price - current_price) / current_price) * 100
            with col:
                st.metric(
                    f"{days}-Day Prediction",
                    f"₹{final_price:.2f}",
                    f"{change_pct:+.2f}%"
                )
        
        # Display prediction chart
        fig = plot_predictions(df, stock_name, predictions, chart_type)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        st.error("Please ensure the LSTM model file exists and is valid.")
        return None
    
    return predictions

def display_technical_analysis(df, stock_name):
    """Display technical analysis dashboard"""
    st.subheader("📊 Technical Analysis Dashboard")
    
    # Add tabs for different types of analysis
    tech_tab1, tech_tab2, tech_tab3 = st.tabs([
        "📈 Technical Indicators",
        "📅 Monthly Trends",
        "🎯 Support & Resistance"
    ])
    
    with tech_tab1:
        # Display technical indicators with explanations
        st.markdown("""
        ### Key Technical Indicators
        Monitor these indicators to understand market momentum, trends, and volatility:
        """)
        
        # Technical Indicators in columns
        tech_indicators = {
            "RSI (14)": {
                "value": df['RSI'].iloc[-1] if 'RSI' in df else None,
                "desc": "Relative Strength Index - Measures momentum. Values > 70 suggest overbought, < 30 oversold."
            },
            "20-Day MA": {
                "value": df['MA20'].iloc[-1] if 'MA20' in df else None,
                "desc": "20-Day Moving Average - Short-term trend indicator."
            },
            "50-Day MA": {
                "value": df['MA50'].iloc[-1] if 'MA50' in df else None,
                "desc": "50-Day Moving Average - Medium-term trend indicator."
            },
            "Volatility": {
                "value": df['Volatility'].iloc[-1] * 100 if 'Volatility' in df else None,
                "desc": "20-Day Volatility - Measures price fluctuation intensity."
            }
        }
        
        # Display indicators in a grid
        cols = st.columns(2)
        for i, (name, info) in enumerate(tech_indicators.items()):
            with cols[i % 2]:
                st.metric(
                    name,
                    f"₹{info['value']:.2f}" if name.endswith("MA") else f"{info['value']:.2f}%",
                    help=info['desc']
                )
    
    with tech_tab2:
        # Monthly trends analysis
        monthly_data, monthly_stats = calculate_monthly_trends(df)
        
        st.markdown("### Monthly Price Trends")
        
        # Display monthly statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Monthly Change", 
                     f"{monthly_stats['Change'].mean():.2f}%",
                     f"{monthly_stats['Change'].iloc[-1]:.2f}% (Last Month)")
        
        with col2:
            st.metric("Average Monthly Range",
                     f"{monthly_stats['Range'].mean():.2f}%",
                     f"{monthly_stats['Range'].iloc[-1]:.2f}% (Last Month)")
        
        # Plot monthly price trends
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['Close'],
            name='Monthly Close',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Monthly Price Trend",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tech_tab3:
        st.markdown("### Support & Resistance Analysis")
        st.markdown("""
        Key price levels where the stock tends to find support (floor) or resistance (ceiling).
        These levels can help identify potential entry and exit points.
        """)
        
        # Calculate pivot points and support/resistance levels
        pivot_data = calculate_pivot_points(df)
        levels = find_support_resistance_levels(df)
        
        current_price = df['Close'].iloc[-1]
        
        # Display current levels
        st.subheader("Price Levels")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Pivot Points**")
            st.metric("Current Price", f"₹{current_price:.2f}")
            st.metric("Pivot Point", f"₹{pivot_data['PP'].iloc[-1]:.2f}")
        
        with col2:
            st.markdown("**Resistance Levels**")
            st.metric("R1", f"₹{pivot_data['R1'].iloc[-1]:.2f}")
            st.metric("R2", f"₹{pivot_data['R2'].iloc[-1]:.2f}")
        
        with col3:
            st.markdown("**Support Levels**")
            st.metric("S1", f"₹{pivot_data['S1'].iloc[-1]:.2f}")
            st.metric("S2", f"₹{pivot_data['S2'].iloc[-1]:.2f}")
        
        # Plot with both pivot points and support/resistance levels
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df.index[-90:],  # Last 90 days
            y=df['Close'][-90:],
            name='Price',
            line=dict(color='blue')
        ))
        
        # Add pivot point lines
        pivot_levels = {
            'PP': ('gray', 'Pivot'),
            'R1': ('red', 'Resistance 1'),
            'R2': ('darkred', 'Resistance 2'),
            'S1': ('green', 'Support 1'),
            'S2': ('darkgreen', 'Support 2')
        }
        
        for level, (color, name) in pivot_levels.items():
            fig.add_hline(
                y=pivot_data[level].iloc[-1],
                line_dash="dash",
                line_color=color,
                annotation_text=f"{name} (₹{pivot_data[level].iloc[-1]:.2f})"
            )
        
        # Add historical support/resistance levels
        if not levels.empty:
            for _, level in levels.iterrows():
                fig.add_hline(
                    y=level['price'],
                    line_dash="dot",
                    line_color="green" if level['type'] == "support" else "red",
                    opacity=0.5,
                    annotation_text=f"Historical {level['type'].title()} (₹{level['price']:.2f})"
                )
        
        fig.update_layout(
            title="Price with Support & Resistance Levels (90 Days)",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show historical levels in an expander
        if not levels.empty:
            with st.expander("View Historical Price Levels"):
                # Sort levels by price
                levels_sorted = levels.sort_values('price', ascending=False)
                
                # Calculate distance from current price
                levels_sorted['Distance'] = ((levels_sorted['price'] - current_price) / current_price) * 100
                
                # Format and display
                st.dataframe(
                    levels_sorted.style.format({
                        'price': '₹{:.2f}',
                        'strength': '{:.0f}',
                        'Distance': '{:+.2f}%'
                    })
                )

def display_stock_features(stock_name, chart_type):
    """Display all features and analysis for the selected stock."""
    st.header(f"Analysis Dashboard: {stock_name}")
    
    # Load and preprocess data
    df = load_stock_data(stock_name)
    
    if df is not None and not df.empty:
        # Display current price and daily change
        current_price = df['Close'].iloc[-1]
        price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Current Price",
                f"₹{current_price:.2f}",
                f"{price_change:+.2f}%"
            )
        with col2:
            st.metric(
                "Trading Volume",
                f"{df['Volume'].iloc[-1]:,.0f}"
            )
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3 = st.tabs(["📈 LSTM Predictions", "📊 Technical Analysis", "🎯 Risk Assessment"])
        
        with tab1:
            predictions = display_lstm_predictions(df, stock_name, chart_type)
        
        with tab2:
            display_technical_analysis(df, stock_name)
        
        with tab3:
            if predictions and '30d' in predictions:
                display_risk_assessment(df, stock_name, predictions['30d'])
            else:
                st.warning("Predictions not available for risk assessment")
    else:
        st.error("Failed to load stock data. Please try again.")

def calculate_trading_probabilities(df, predictions, window=30):
    """Calculate buy/sell/hold probabilities based on LSTM predictions"""
    
    # Check if predictions is valid
    if predictions is None or not isinstance(predictions, (list, np.ndarray)) or len(predictions) == 0:
        # Return default probabilities if no predictions available
        return {
            'probabilities': {
                'buy': 33.33,
                'sell': 33.33,
                'hold': 33.34
            },
            'metrics': {
                'momentum': 0,
                'volatility': df['Close'].pct_change().std() * np.sqrt(window) * 100,
                'risk_reward': 0
            }
        }
    
    # Convert predictions to numpy array if it's not already
    predictions = np.array(predictions).flatten()  # Ensure 1D array
    current_price = df['Close'].iloc[-1]
    
    # Calculate price changes as percentage
    price_changes = np.zeros_like(predictions)
    for i in range(len(predictions)):
        price_changes[i] = (predictions[i] - current_price) / current_price * 100
    
    # Calculate momentum from recent predictions
    momentum = np.mean(price_changes)
    momentum_std = np.std(price_changes)
    
    # Calculate volatility
    volatility = df['Close'].pct_change().std() * np.sqrt(window) * 100
    
    # Calculate Risk/Reward Ratio
    # Use historical volatility to estimate potential loss
    potential_loss = current_price * (volatility / 100)  # Estimated max loss based on volatility
    
    # Use predicted price movement for potential gain
    if momentum > 0:
        potential_gain = current_price * (abs(momentum) / 100)
        risk_reward = potential_loss / max(potential_gain, 0.01)  # Avoid division by zero
    else:
        potential_gain = current_price * (abs(momentum) / 100)
        risk_reward = potential_loss / max(potential_gain, 0.01)  # Avoid division by zero
    
    # Base probabilities on price movement and volatility
    if momentum > 0:
        buy_base = 60 + min(momentum * 5, 30)
        sell_base = max(10, 40 - momentum * 5)
    else:
        buy_base = max(10, 60 + momentum * 5)
        sell_base = min(90, 40 - momentum * 5)
    
    # Adjust for volatility
    volatility_factor = min(volatility / 2, 20)  # Cap volatility impact at 20%
    hold_base = volatility_factor
    
    # Normalize probabilities to sum to 100
    total = buy_base + sell_base + hold_base
    buy_prob = (buy_base / total) * 100
    sell_prob = (sell_base / total) * 100
    hold_prob = (hold_base / total) * 100
    
    return {
        'probabilities': {
            'buy': buy_prob,
            'sell': sell_prob,
            'hold': hold_prob
        },
        'metrics': {
            'momentum': momentum,
            'volatility': volatility,
            'risk_reward': risk_reward,
            'potential_loss': potential_loss,
            'potential_gain': potential_gain
        }
    }

def display_risk_assessment(df, stock_name, predictions):
    """Display risk assessment and trading probabilities"""
    st.header("🎯 Risk Assessment")
    
    # Calculate trading probabilities
    analysis = calculate_trading_probabilities(df, predictions)
    probs = analysis['probabilities']
    metrics = analysis['metrics']
    
    # Display probability gauge charts
    st.subheader("Trading Probabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probs['buy'],
            title={'text': "Buy Probability"},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "green"},
                  'steps': [
                      {'range': [0, 30], 'color': "lightgray"},
                      {'range': [30, 70], 'color': "gray"},
                      {'range': [70, 100], 'color': "darkgray"}]}))
        fig.update_layout(height=250, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probs['sell'],
            title={'text': "Sell Probability"},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "red"},
                  'steps': [
                      {'range': [0, 30], 'color': "lightgray"},
                      {'range': [30, 70], 'color': "gray"},
                      {'range': [70, 100], 'color': "darkgray"}]}))
        fig.update_layout(height=250, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probs['hold'],
            title={'text': "Hold Probability"},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "blue"},
                  'steps': [
                      {'range': [0, 30], 'color': "lightgray"},
                      {'range': [30, 70], 'color': "gray"},
                      {'range': [70, 100], 'color': "darkgray"}]}))
        fig.update_layout(height=250, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Display Risk/Reward Ratio
    st.subheader("Risk/Reward Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=metrics['risk_reward'],
            title={'text': "Risk/Reward Ratio"},
            delta={'reference': 1, 'relative': True},
            number={'prefix': "1:", 'valueformat': ".2f"}))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Potential Gain", f"₹{metrics['potential_gain']:,.2f}")
        st.metric("Potential Loss", f"₹{metrics['potential_loss']:,.2f}")
    
    # Display analysis summary
    st.subheader("Analysis Summary")
    
    momentum = metrics['momentum']
    volatility = metrics['volatility']
    
    # Determine market condition based on metrics
    if abs(momentum) < 5:
        market_condition = "neutral"
    elif momentum > 0:
        market_condition = "bullish"
    else:
        market_condition = "bearish"
    
    # Generate summary text
    summary = f"""Based on the 30-day prediction model, the market appears to be {market_condition}. 
The analysis shows a {abs(momentum):.1f}% predicted price movement, with a volatility of {volatility:.1f}%.

Trading Recommendation:
{'Buy' if probs['buy'] > max(probs['sell'], probs['hold']) else 'Sell' if probs['sell'] > max(probs['buy'], probs['hold']) else 'Hold'} 
with {max(probs['buy'], probs['sell'], probs['hold']):.1f}% probability.

Risk/Reward Assessment:
The trade has a risk/reward ratio of 1:{metrics['risk_reward']:.2f}, indicating a 
{'favorable' if metrics['risk_reward'] < 1 else 'balanced' if metrics['risk_reward'] == 1 else 'cautionary'} risk profile."""
    
    st.write(summary)

def calculate_monthly_trends(df):
    """Calculate monthly trends from historical data"""
    # Resample data to monthly frequency
    monthly_data = df.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # Calculate monthly statistics
    monthly_stats = pd.DataFrame()
    monthly_stats['Average'] = (monthly_data['High'] + monthly_data['Low'] + monthly_data['Close']) / 3
    monthly_stats['Change'] = monthly_data['Close'].pct_change() * 100
    monthly_stats['Range'] = (monthly_data['High'] - monthly_data['Low']) / monthly_data['Low'] * 100
    monthly_stats['Volume_Change'] = monthly_data['Volume'].pct_change() * 100
    
    return monthly_data, monthly_stats

def calculate_pivot_points(df):
    """Calculate pivot points and support/resistance levels"""
    pivot_data = pd.DataFrame()
    
    # Calculate classic pivot points
    pivot_data['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    pivot_data['R1'] = 2 * pivot_data['PP'] - df['Low']
    pivot_data['S1'] = 2 * pivot_data['PP'] - df['High']
    pivot_data['R2'] = pivot_data['PP'] + (df['High'] - df['Low'])
    pivot_data['S2'] = pivot_data['PP'] - (df['High'] - df['Low'])
    
    return pivot_data

def find_support_resistance_levels(df, window=20, price_threshold=0.02):
    """Find support and resistance levels based on price action"""
    levels = []
    
    # Convert to numpy array for faster computation
    prices = df['Close'].values
    
    for i in range(window, len(prices) - window):
        # Get the window of prices around current point
        window_prices = prices[i-window:i+window]
        current_price = prices[i]
        
        # Check if current price is a local minimum (support)
        if current_price == min(window_prices):
            levels.append({
                'price': current_price,
                'type': 'support',
                'date': df.index[i]
            })
        
        # Check if current price is a local maximum (resistance)
        if current_price == max(window_prices):
            levels.append({
                'price': current_price,
                'type': 'resistance',
                'date': df.index[i]
            })
    
    # Convert to DataFrame
    levels_df = pd.DataFrame(levels)
    
    if not levels_df.empty:
        # Group nearby levels
        grouped_levels = []
        current_price = df['Close'].iloc[-1]
        
        for level_type in ['support', 'resistance']:
            type_levels = levels_df[levels_df['type'] == level_type]['price'].values
            
            if len(type_levels) > 0:
                # Use clustering to group nearby price levels
                from sklearn.cluster import DBSCAN
                clusters = DBSCAN(eps=current_price * price_threshold, min_samples=1).fit(type_levels.reshape(-1, 1))
                
                # Calculate mean price for each cluster
                unique_labels = set(clusters.labels_)
                for label in unique_labels:
                    cluster_prices = type_levels[clusters.labels_ == label]
                    grouped_levels.append({
                        'price': float(np.mean(cluster_prices)),
                        'type': level_type,
                        'strength': len(cluster_prices)  # Number of points in cluster indicates strength
                    })
        
        return pd.DataFrame(grouped_levels)
    
    return pd.DataFrame(columns=['price', 'type', 'strength'])

def main():
    st.set_page_config(
        page_title="Stock Prediction Dashboard",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Stock Price Prediction Dashboard")

    # Sidebar for stock selection
    with st.sidebar:
        st.header("Settings")
        selected_stock = st.selectbox(
            "Select a Stock",
            AVAILABLE_STOCKS,
            help="Choose a stock to analyze"
        )

        # Add time range selector
        st.subheader("Chart Settings")
        chart_type = st.radio(
            "Chart Type",
            ["Candlestick", "Line"],
            disabled=False  # Enabled now
        )

    # Display features for selected stock
    if selected_stock:
        display_stock_features(selected_stock, chart_type)

if __name__ == "__main__":
    main()
