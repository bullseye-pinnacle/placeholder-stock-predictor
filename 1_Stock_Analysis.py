import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from utils.data_processing import load_stock_data
from utils.visualization import plot_predictions

def display_stock_analysis(stock_name, chart_type):
    """Display stock analysis dashboard"""
    st.header(f"Stock Analysis: {stock_name}")
    
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
            
            # Display the chart
            fig = plot_predictions(df, stock_name, {}, chart_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Indicators
            st.subheader("Technical Indicators")
            tech_indicators = {
                "RSI (14)": df['RSI'].iloc[-1] if 'RSI' in df else None,
                "20-Day MA": df['MA20'].iloc[-1] if 'MA20' in df else None,
                "50-Day MA": df['MA50'].iloc[-1] if 'MA50' in df else None,
                "Volatility": df['Volatility'].iloc[-1] if 'Volatility' in df else None
            }
            
            cols = st.columns(len(tech_indicators))
            for col, (name, value) in zip(cols, tech_indicators.items()):
                if value is not None:
                    col.metric(name, f"{value:.2f}")
                else:
                    col.metric(name, "N/A")
        else:
            st.error(f"Failed to load data for {stock_name}")

def main():
    st.title("📊 Stock Analysis")
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        selected_stock = st.selectbox(
            "Select Stock",
            ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
            help="Choose a stock to analyze"
        )
        
        st.subheader("Chart Settings")
        chart_type = st.radio(
            "Chart Type",
            ["Candlestick", "Line"]
        )
    
    if selected_stock:
        display_stock_analysis(selected_stock, chart_type)

if __name__ == "__main__":
    main()
