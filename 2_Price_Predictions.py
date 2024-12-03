import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from tensorflow.keras.models import load_model
from utils.data_processing import load_stock_data, prepare_sequence_data, generate_predictions
from utils.visualization import plot_predictions

def display_predictions(stock_name, chart_type):
    """Display stock predictions dashboard"""
    st.header(f"Price Predictions: {stock_name}")
    
    with st.spinner('Loading data and generating predictions...'):
        df = load_stock_data(stock_name)
        if df is not None and not df.empty:
            current_price = df['Close'].iloc[-1]
            predictions = {}
            
            try:
                model = load_model(f'lstm_model_{stock_name.lower()}.keras')
                sequence_data, scaler = prepare_sequence_data(df)
                last_sequence = sequence_data[-1]
                
                # Generate predictions for different time horizons
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
                
                # Display prediction chart
                fig = plot_predictions(df, stock_name, predictions, chart_type)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.error("Please ensure the LSTM model file exists and is valid.")
        else:
            st.error(f"Failed to load data for {stock_name}")

def main():
    st.title("🔮 Price Predictions")
    
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
        display_predictions(selected_stock, chart_type)

if __name__ == "__main__":
    main()
