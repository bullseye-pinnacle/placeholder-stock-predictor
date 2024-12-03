import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

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

def display_stock_features(stock_name):
    """
    Display all features and analysis for the selected stock.
    This function will be expanded with actual features later.
    """
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
            
            # Load LSTM model and generate predictions
            try:
                model = load_model(f'lstm_model_{stock_name.lower()}.keras')
                sequence_data, scaler = prepare_sequence_data(df)
                last_sequence = sequence_data[-1]
                
                predictions = {}
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
                
                # Plot historical data with predictions
                fig = plot_predictions(df, stock_name, predictions)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.error("Please ensure the LSTM model file exists and is valid.")
                
                # Show historical data without predictions
                fig = plot_stock_history(df, stock_name)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error(f"Failed to load data for {stock_name}")
            # Display interactive plot
            fig = plot_stock_history(df, stock_name)
            st.plotly_chart(fig, use_container_width=True)
    
    # Placeholder for features (to be implemented)
    st.write("Features to be implemented:")
    features = [
        "LSTM Price Predictions (30, 60, 120 days)",
        "Technical Indicators",
        "Risk Metrics",
        "Trading Signals",
        "Position Sizing",
        "Trend Analysis"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")

def prepare_sequence_data(data, sequence_length=60):
    """Prepare sequence data for LSTM prediction"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    sequences = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:(i + sequence_length)])
    
    return np.array(sequences), scaler

def generate_predictions(model, last_sequence, scaler, days_ahead):
    """Generate predictions for specified number of days"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Predict next value
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def plot_predictions(df, stock_name, predictions_dict):
    """Plot historical data with predictions"""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Historical Price',
        line=dict(color='blue')
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
        xaxis_title='Date',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

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
    
    # Display features for selected stock
    if selected_stock:
        display_stock_features(selected_stock)

if __name__ == "__main__":
    main()
