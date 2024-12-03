import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from datetime import datetime

class StockPredictor:
    def __init__(self, sequence_length=60, batch_size=32):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.training_history = {}
        self.stock_metrics = {}
        
    def combine_stock_data(self, symbols, start_date='2000-01-01'):
        """Fetch and combine data from multiple stocks for training"""
        print("Fetching and combining stock data...")
        combined_data = []
        valid_symbols = []
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, progress=False)
                if len(data) > self.sequence_length * 2:  # Ensure sufficient data
                    # Normalize each stock's prices to its starting price
                    normalized_close = data['Close'] / data['Close'].iloc[0] * 100
                    combined_data.append(normalized_close)
                    valid_symbols.append(symbol)
                    print(f"Successfully processed {symbol}")
                else:
                    print(f"Insufficient data for {symbol}")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                
        if not combined_data:
            return None, []
            
        # Combine all normalized stock data
        combined_df = pd.concat(combined_data, axis=1)
        combined_df.columns = valid_symbols
        
        return combined_df, valid_symbols

    def prepare_batch_data(self, data):
        """Prepare data for batch training"""
        X, y = [], []
        
        # For each stock column
        for column in data.columns:
            stock_data = data[column].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(stock_data)
            
            # Create sequences
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Split into train and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        return X_train, X_val, y_train, y_val

    def create_advanced_model(self):
        """Create an enhanced LSTM model"""
        model = Sequential([
            LSTM(256, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.3),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(128),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber'  # More robust to outliers
        )
        return model

    def train_on_batch(self, symbols, epochs=100):
        """Train the model on a batch of stocks"""
        # Fetch and prepare data
        combined_data, valid_symbols = self.combine_stock_data(symbols)
        if combined_data is None:
            return False
            
        X_train, X_val, y_train, y_val = self.prepare_batch_data(combined_data)
        
        # Create or load model
        if self.model is None:
            self.model = self.create_advanced_model()
        
        # Training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.0001
            ),
            ModelCheckpoint(
                'temp_best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model on each stock
        for symbol in valid_symbols:
            stock_data = yf.download(symbol, start='2023-01-01', progress=False)
            if len(stock_data) > self.sequence_length:
                eval_metrics = self.evaluate_on_stock(stock_data, symbol)
                self.stock_metrics[symbol] = eval_metrics
        
        return True

    def evaluate_on_stock(self, stock_data, symbol):
        """Evaluate model performance on a single stock"""
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        predictions = self.model.predict(X, verbose=0)
        
        # Calculate metrics
        metrics = self.evaluate_predictions(y, predictions)
        
        # Plot predictions
        self.plot_predictions(y, predictions, symbol)
        
        return metrics

    def save_complete_model(self, output_dir='model_output'):
        """Save model with all necessary components"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(output_dir, 'best_model.keras'))
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.joblib'))
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'stock_metrics': self.stock_metrics,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': '2.0'
        }
        
        with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

def main():
    """Train model on multiple batches of stocks"""
    # All NIFTY50 stocks (you can extend this list)
    ALL_STOCKS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        # Add more stocks here
    ]
    
    # Initialize predictor
    predictor = StockPredictor(sequence_length=60, batch_size=32)
    
    # Train in batches of 10 stocks
    batch_size = 10
    for i in range(0, len(ALL_STOCKS), batch_size):
        batch_stocks = ALL_STOCKS[i:i + batch_size]
        print(f"\nTraining on batch {i//batch_size + 1}:")
        print("Stocks:", batch_stocks)
        
        success = predictor.train_on_batch(batch_stocks)
        if not success:
            print(f"Failed to train on batch {i//batch_size + 1}")
    
    # Save final model
    predictor.save_complete_model()
    print("\nTraining completed. Model saved in 'model_output' directory")
    
    # Print performance metrics
    print("\nPerformance metrics by stock:")
    for symbol, metrics in predictor.stock_metrics.items():
        print(f"\n{symbol}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
