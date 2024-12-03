import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
