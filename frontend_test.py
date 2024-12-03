import streamlit as st

# List of available stocks (these are the ones we have models for)
AVAILABLE_STOCKS = [
    "TATAMOTORS", "DRREDDY", "CIPLA", "HINDUNILVR", "TRENT",
    "BAJFINANCE", "BHARTIARTL", "HINDALCO", "HDFCBANK", "TCS",
    "TATASTEEL", "ICICIBANK", "INFY", "ITC", "RELIANCE", "MARUTI"
]

def display_stock_features(stock_name):
    """
    Display all features and analysis for the selected stock.
    This function will be expanded with actual features later.
    """
    st.header(f"Analysis Dashboard: {stock_name}")
    
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
