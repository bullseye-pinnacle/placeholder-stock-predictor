import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from PIL import ImageColor

# List of available stocks (these are the ones we have models for)
AVAILABLE_STOCKS = [
    "TATAMOTORS", "DRREDDY", "CIPLA", "HINDUNILVR", "TRENT",
    "BAJFINANCE", "BHARTIARTL", "HINDALCO", "HDFCBANK", "TCS",
    "TATASTEEL", "ICICIBANK", "INFY", "ITC", "RELIANCE", "MARUTI"
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

def calculate_monthly_trends(predictions_dict):
    """Convert daily predictions to monthly trends"""
    monthly_trends = {}
    
    for period, daily_preds in predictions_dict.items():
        # Convert predictions to DataFrame with dates
        last_date = datetime.now()
        dates = pd.date_range(
            start=last_date,
            periods=len(daily_preds),
            freq='D'
        )
        df_preds = pd.DataFrame({
            'date': dates,
            'prediction': daily_preds
        })
        
        # Resample to monthly frequency
        monthly_df = df_preds.set_index('date').resample('M').agg({
            'prediction': [
                ('mean', 'mean'),
                ('min', 'min'),
                ('max', 'max'),
                ('volatility', 'std')
            ]
        }).droplevel(0, axis=1)
        
        monthly_trends[period] = monthly_df
    
    return monthly_trends

def plot_monthly_trends(monthly_trends, current_price):
    """Create monthly trends visualization"""
    fig = go.Figure()
    
    colors = {'30d': 'red', '60d': 'orange', '120d': 'green'}
    
    for period, df in monthly_trends.items():
        # Plot mean prediction
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['mean'],
            name=f'{period} Mean',
            line=dict(color=colors[period])
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=df.index.tolist() + df.index.tolist()[::-1],
            y=df['max'].tolist() + df['min'].tolist()[::-1],
            fill='tonexty',
            fillcolor=f'rgba{tuple(list(ImageColor.getrgb(colors[period])) + [0.2])}',
            line=dict(width=0),
            showlegend=False,
            name=f'{period} Range'
        ))
    
    # Add current price reference line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="gray",
        annotation_text="Current Price"
    )
    
    fig.update_layout(
        title="Monthly Price Trend Forecast",
        yaxis_title='Price (₹)',
        xaxis_title='Month',
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def display_stock_features(stock_name, chart_type):
    """Display all features and analysis for the selected stock."""
    st.header(f"Analysis Dashboard: {stock_name}")
    
    # Load and display historical data
    with st.spinner('Loading historical data...'):
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
            
            predictions = {}
            # Load LSTM model and generate predictions
            try:
                model = load_model(f'lstm_model_{stock_name.lower()}.keras')
                sequence_data, scaler = prepare_sequence_data(df)
                last_sequence = sequence_data[-1]
                
                for days in [30, 60, 120]:
                    pred = generate_predictions(model, last_sequence, scaler, days)
                    predictions[f'{days}d'] = pred
                    
                    # Display prediction metrics
                    final_price = pred[-1]
                    change_pct = ((final_price - current_price) / current_price) * 100
                    st.metric(
                        f"{days}-Day Prediction",
                        f"₹{final_price:.2f}",
                        f"{change_pct:+.2f}%"
                    )
                
                # Add Monthly Trend Analysis
                st.subheader("Monthly Trend Analysis")
                monthly_trends = calculate_monthly_trends(predictions)
                fig_monthly = plot_monthly_trends(monthly_trends, current_price)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Display monthly statistics
                st.write("Monthly Price Statistics")
                for period, df in monthly_trends.items():
                    with st.expander(f"{period} Monthly Statistics"):
                        st.dataframe(
                            df.style.format({
                                'mean': '₹{:.2f}',
                                'min': '₹{:.2f}',
                                'max': '₹{:.2f}',
                                'volatility': '₹{:.2f}'
                            })
                        )
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.error("Please ensure the LSTM model file exists and is valid.")
                predictions = {}  # Reset predictions if they failed
            
            # Always show the historical data plot with any available predictions
            st.subheader("Price History and Predictions")
            fig = plot_predictions(df, stock_name, predictions, chart_type)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"Failed to load data for {stock_name}")
            
    # Additional features section
    st.subheader("Technical Indicators")
    if df is not None and not df.empty:
        tech_indicators = {
            "RSI (14)": df['RSI'].iloc[-1] if 'RSI' in df else None,
            "20-Day MA": df['MA20'].iloc[-1] if 'MA20' in df else None,
            "50-Day MA": df['MA50'].iloc[-1] if 'MA50' in df else None,
            "Volatility": df['Volatility'].iloc[-1] if 'Volatility' in df else None
        }
        
        # Display technical indicators in columns
        cols = st.columns(len(tech_indicators))
        for col, (name, value) in zip(cols, tech_indicators.items()):
            if value is not None:
                col.metric(name, f"{value:.2f}")
            else:
                col.metric(name, "N/A")

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
