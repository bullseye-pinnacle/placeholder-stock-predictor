import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import CustomBusinessDay
from tensorflow.keras.models import load_model
import os
from dateutil.relativedelta import relativedelta



# Add this at the beginning of the script to suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

@st.cache_resource  # This will cache the model load so it's only done once
def load_model():
    """
    Load the saved Keras model
    """
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the model file
        model_path = os.path.join(script_dir,f'{idk}_best_model.keras')
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_stock_price(model, stock_symbol, num_days):
    """
    Make predictions using the loaded Keras model.
    """
    if model is None:
        raise Exception("Model not loaded properly")
    
    # Fetch recent stock data for feature creation
    # Adjust the period according to your model's input requirements
    stock = yf.Ticker(stock_symbol)
    hist_data = stock.history(period="60d")  # Adjust period based on your model's requirements
    
    # Prepare input data for prediction
    # Note: Modify this according to how your model expects the input data
    try:
        # Create predictions for each day
        dates = get_business_days(datetime.now(), num_days)
        predictions = []
        
        # Example prediction loop - modify according to your model's requirements
        last_window = hist_data['Close'].values[-60:]  # Adjust window size based on your model
        
        for _ in range(len(dates)):
            # Reshape input data according to your model's requirements
            model_input = last_window.reshape((1, -1))  # Adjust shape based on your model
            
            # Make prediction
            pred = model.predict(model_input, verbose=0)[0][0]
            predictions.append(pred)
            
            # Update the window for next prediction
            last_window = np.append(last_window[1:], pred)
        
        return dates, predictions
    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")

def predict_target_price_date(model, stock_symbol, target_price):
    """
    Predict when the stock will reach the target price using the loaded Keras model.
    """
    if model is None:
        raise Exception("Model not loaded properly")
    
    # Use the same prediction logic as above but continue until target price is reached or 365 days
    try:
        dates = get_business_days(datetime.now(), 365)  # Get full year of business days
        stock = yf.Ticker(stock_symbol)
        hist_data = stock.history(period="60d")
        
        predictions = []
        last_window = hist_data['Close'].values[-60:]  # Adjust window size based on your model
        
        for _ in range(len(dates)):
            model_input = last_window.reshape((1, -1))  # Adjust shape based on your model
            pred = model.predict(model_input, verbose=0)[0][0]
            predictions.append(pred)
            
            # Update the window for next prediction
            last_window = np.append(last_window[1:], pred)
            
            # Check if we've reached the target price
            if pred >= target_price:
                # Return only up to this point
                return dates[:len(predictions)], predictions
        
        # If we haven't reached the target price, return all predictions
        return dates, predictions
    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")

def main():
    st.title("Stock Price Prediction Dashboard")
    
    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    
    # Stock symbol input
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    
    # Tab selection
    tab1, tab2 = st.tabs(["Price Prediction", "Target Price Analysis"])
    
    # Load the model (do this once)
    model = load_model()
    
    with tab1:
        st.header("Stock Price Prediction")
        
        # Number of days input
        num_days = st.slider("Select number of days to predict:", 
                           min_value=1, 
                           max_value=365, 
                           value=30)
        
        if st.button("Predict Price"):
            try:
                # Get predictions
                dates, predictions = predict_stock_price(model, stock_symbol, num_days)
                
                # Create DataFrame for display
                df_pred = pd.DataFrame({
                    'Date': dates,
                    'Predicted Price': predictions
                })
                
                # Display predictions in a line chart using Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_pred['Date'],
                    y=df_pred['Predicted Price'],
                    mode='lines+markers',
                    name='Predicted Price'
                ))
                
                fig.update_layout(
                    title=f'Predicted Stock Prices for {stock_symbol}',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    hovermode='x'
                )
                
                st.plotly_chart(fig)
                
                # Display the prediction data in a table
                st.subheader("Predicted Values")
                st.dataframe(df_pred)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        st.header("Target Price Analysis")
        
        # Target price input
        target_price = st.number_input("Enter target price:", 
                                     min_value=0.0, 
                                     value=100.0)
        
        if st.button("Analyze Target Price"):
            try:
                # Get predictions
                dates, predictions = predict_target_price_date(model, stock_symbol, target_price)
                
                # Check if target price is reached
                reached_index = np.where(predictions >= target_price)[0]
                
                if len(reached_index) > 0:
                    target_date = dates[reached_index[0]]
                    days_to_target = (target_date - datetime.now()).days
                    
                    st.success(f"Target price of ${target_price:.2f} predicted to be reached on {target_date.strftime('%Y-%m-%d')} "
                             f"({days_to_target} days from now)")
                else:
                    st.warning(f"The target price of ${target_price:.2f} is not predicted to be reached within the next 365 days")
                
                # Create prediction visualization
                df_pred = pd.DataFrame({
                    'Date': dates,
                    'Predicted Price': predictions
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_pred['Date'],
                    y=df_pred['Predicted Price'],
                    mode='lines+markers',
                    name='Predicted Price'
                ))
                
                # Add target price line
                fig.add_hline(y=target_price, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text="Target Price")
                
                fig.update_layout(
                    title=f'Price Predictions vs Target Price for {stock_symbol}',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    hovermode='x'
                )
                
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
