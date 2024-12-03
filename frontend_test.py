import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import ta
import os

# Page configuration
st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main {padding: 0rem;}
    .stButton>button {width: 100%;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)
