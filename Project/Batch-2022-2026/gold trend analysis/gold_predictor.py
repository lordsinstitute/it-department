import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except:
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class GoldPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60
        
    def fetch_gold_data(self, period='2y'):
        """Load gold price data from CSV"""
        try:
            # Try to load from CSV file
            if os.path.exists('data/gold_data.csv'):
                print("Loading data from data/gold_data.csv...")
                df = pd.read_csv('data/gold_data.csv', parse_dates=['Date'], index_col='Date')
                
                # Handle 'Adj Close' column if present (Yahoo Finance format)
                if 'Adj Close' in df.columns:
                    df = df.drop('Adj Close', axis=1)
                
                # Remove rows with null values
                df = df.dropna()
                
                # Filter by period if needed
                if period == '2y':
                    days = 730
                elif period == '6mo':
                    days = 180
                elif period == '3mo':
                    days = 90
                else:
                    days = 730
                
                df = df.tail(days)
                
                if len(df) < 100:
                    print(f"Warning: Only {len(df)} rows found. Need at least 100 rows.")
                    return self.create_sample_data()
                
                print(f"Loaded {len(df)} rows of data")
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
                print(f"Latest Close price: ${df['Close'].iloc[-1]:.2f}")
                return df
            else:
                print("data/gold_data.csv not found. Creating sample data...")
                return self.create_sample_data()
        except Exception as e:
            print(f"Error loading CSV: {e}. Creating sample data...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample gold price data"""
        print("Generating sample gold price data...")
        dates = pd.date_range(end=datetime.now(), periods=730, freq='D')
        
        # Generate realistic gold price data (2024-2025 prices around $5000-5400)
        np.random.seed(42)
        base_price = 5000
        trend = np.linspace(0, 400, 730)
        noise = np.random.normal(0, 50, 730)
        seasonal = 100 * np.sin(np.linspace(0, 4*np.pi, 730))
        
        close_prices = base_price + trend + noise + seasonal
        
        df = pd.DataFrame({
            'Open': close_prices + np.random.uniform(-20, 20, 730),
            'High': close_prices + np.random.uniform(10, 40, 730),
            'Low': close_prices - np.random.uniform(10, 40, 730),
            'Close': close_prices,
            'Volume': np.random.randint(100000, 500000, 730)
        }, index=dates)
        
        # Save to CSV for future use
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/gold_data.csv')
        print("Sample data saved to data/gold_data.csv")
        print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        df.dropna(inplace=True)
        return df
    
    def prepare_data(self, df):
        """Prepare data for LSTM"""
        features = ['Close', 'Volume', 'MA_10', 'MA_30', 'RSI', 'MACD']
        data = df[features].values
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, epochs=50):
        """Train the model"""
        print("Fetching gold data...")
        df = self.fetch_gold_data()
        
        if df.empty:
            raise Exception("Failed to fetch gold data. Check internet connection.")
        
        print("Calculating indicators...")
        df = self.calculate_indicators(df)
        
        print("Preparing data...")
        X, y = self.prepare_data(df)
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print("Building model...")
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        print("Training model...")
        self.model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
        
        print("Saving model...")
        self.model.save('gold_model.h5')
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Model and scaler saved successfully!")
        return df
    
    def predict_tomorrow(self, current_price=None):
        """Predict tomorrow's gold price"""
        try:
            if not os.path.exists('gold_model.h5'):
                print("Model not found. Training new model...")
                self.train()
            else:
                self.model = load_model('gold_model.h5')
                
                # Load scaler
                if os.path.exists('scaler.pkl'):
                    with open('scaler.pkl', 'rb') as f:
                        self.scaler = pickle.load(f)
            
            df = self.fetch_gold_data(period='3mo')
            
            if df.empty:
                raise Exception("Failed to fetch gold data. Check internet connection.")
            
            df = self.calculate_indicators(df)
            
            features = ['Close', 'Volume', 'MA_10', 'MA_30', 'RSI', 'MACD']
            data = df[features].values
            scaled_data = self.scaler.fit_transform(data)
            
            # If user provides current_price, update the last row with it
            if current_price:
                # Create a copy of the last sequence
                last_sequence = scaled_data[-self.sequence_length:].copy()
                
                # Update the last row's Close price with user input
                dummy_input = np.zeros((1, len(features)))
                dummy_input[0, 0] = current_price
                dummy_input[0, 1:] = data[-1, 1:]  # Keep other features same
                scaled_input = self.scaler.transform(dummy_input)
                last_sequence[-1, 0] = scaled_input[0, 0]  # Update Close price
                
                today_price = current_price
            else:
                last_sequence = scaled_data[-self.sequence_length:]
                today_price = df['Close'].iloc[-1]
            
            last_sequence = np.reshape(last_sequence, (1, self.sequence_length, len(features)))
            
            prediction_scaled = self.model.predict(last_sequence, verbose=0)
            
            # Inverse transform
            dummy = np.zeros((1, len(features)))
            dummy[0, 0] = prediction_scaled[0, 0]
            prediction = self.scaler.inverse_transform(dummy)[0, 0]
            direction = "UP" if prediction > today_price else "DOWN"
            change_percent = ((prediction - today_price) / today_price) * 100
            
            # Get dates
            today_date = df.index[-1].strftime('%Y-%m-%d')
            tomorrow_date = (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
            
            return {
                'today_price': float(today_price),
                'predicted_price': float(prediction),
                'direction': direction,
                'change_percent': float(change_percent),
                'confidence': float(min(abs(change_percent) * 10, 95)),
                'today_date': today_date,
                'tomorrow_date': tomorrow_date
            }
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
    def get_historical_data(self):
        """Get historical data for charts"""
        try:
            df = self.fetch_gold_data(period='6mo')
            
            if df.empty:
                raise Exception("Failed to fetch gold data. Check internet connection.")
            
            df = self.calculate_indicators(df)
            
            return {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'prices': df['Close'].tolist(),
                'volume': df['Volume'].tolist(),
                'ma_10': df['MA_10'].tolist(),
                'ma_30': df['MA_30'].tolist(),
                'rsi': df['RSI'].tolist(),
                'macd': df['MACD'].tolist()
            }
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            raise
    
    def get_recent_data(self, period='1w'):
        """Get recent data for trend analysis"""
        try:
            if period == '1w':
                days = 7
            elif period == '1mo':
                days = 30
            else:
                days = 7
            
            df = self.fetch_gold_data(period='3mo')
            df = self.calculate_indicators(df)
            df = df.tail(days)
            
            return {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'prices': df['Close'].tolist()
            }
        except Exception as e:
            print(f"Error fetching recent data: {str(e)}")
            raise
    
    def predict_future(self, num_days=7, current_price=None):
        """Predict future prices for multiple days"""
        try:
            if not os.path.exists('gold_model.h5'):
                self.train()
            else:
                self.model = load_model('gold_model.h5')
                if os.path.exists('scaler.pkl'):
                    with open('scaler.pkl', 'rb') as f:
                        self.scaler = pickle.load(f)
            
            df = self.fetch_gold_data(period='3mo')
            df = self.calculate_indicators(df)
            
            features = ['Close', 'Volume', 'MA_10', 'MA_30', 'RSI', 'MACD']
            data = df[features].values
            scaled_data = self.scaler.fit_transform(data)
            
            # Start with last sequence
            current_sequence = scaled_data[-self.sequence_length:].copy()
            
            # Update with current price if provided
            if current_price:
                dummy_input = np.zeros((1, len(features)))
                dummy_input[0, 0] = current_price
                dummy_input[0, 1:] = data[-1, 1:]
                scaled_input = self.scaler.transform(dummy_input)
                current_sequence[-1, 0] = scaled_input[0, 0]
            
            predictions = []
            last_date = df.index[-1]
            
            for day in range(1, num_days + 1):
                seq_input = np.reshape(current_sequence, (1, self.sequence_length, len(features)))
                pred_scaled = self.model.predict(seq_input, verbose=0)
                
                dummy = np.zeros((1, len(features)))
                dummy[0, 0] = pred_scaled[0, 0]
                pred_price = self.scaler.inverse_transform(dummy)[0, 0]
                
                pred_date = (last_date + timedelta(days=day)).strftime('%Y-%m-%d')
                predictions.append({
                    'date': pred_date,
                    'price': float(pred_price)
                })
                
                # Update sequence for next prediction
                new_row = current_sequence[-1].copy()
                new_row[0] = pred_scaled[0, 0]
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            return predictions
        except Exception as e:
            print(f"Error in future prediction: {str(e)}")
            raise
    
    def predict_with_manual_data(self, manual_data):
        """Predict using manually entered technical indicators"""
        try:
            if not os.path.exists('gold_model.h5'):
                raise Exception("Model not found. Train the model first.")
            
            self.model = load_model('gold_model.h5')
            if os.path.exists('scaler.pkl'):
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Get historical data for sequence
            df = self.fetch_gold_data(period='3mo')
            df = self.calculate_indicators(df)
            
            features = ['Close', 'Volume', 'MA_10', 'MA_30', 'RSI', 'MACD']
            data = df[features].values
            scaled_data = self.scaler.fit_transform(data)
            
            # Use last 59 days from history + 1 manual entry
            last_sequence = scaled_data[-(self.sequence_length-1):].copy()
            
            # Create manual entry row
            manual_row = np.array([[
                manual_data['close'],
                manual_data['volume'],
                manual_data.get('ma_10', df['MA_10'].iloc[-1]),
                manual_data.get('ma_30', df['MA_30'].iloc[-1]),
                manual_data.get('rsi', df['RSI'].iloc[-1]),
                manual_data.get('macd', df['MACD'].iloc[-1])
            ]])
            
            # Scale manual entry
            scaled_manual = self.scaler.transform(manual_row)
            
            # Combine sequence
            full_sequence = np.vstack([last_sequence, scaled_manual])
            full_sequence = np.reshape(full_sequence, (1, self.sequence_length, len(features)))
            
            # Predict
            prediction_scaled = self.model.predict(full_sequence, verbose=0)
            
            # Inverse transform
            dummy = np.zeros((1, len(features)))
            dummy[0, 0] = prediction_scaled[0, 0]
            prediction = self.scaler.inverse_transform(dummy)[0, 0]
            
            current_price = manual_data['close']
            direction = "UP" if prediction > current_price else "DOWN"
            change_percent = ((prediction - current_price) / current_price) * 100
            
            return {
                'predicted_price': float(prediction),
                'direction': direction,
                'change_percent': float(change_percent),
                'confidence': float(min(abs(change_percent) * 10, 95))
            }
        except Exception as e:
            print(f"Error in manual prediction: {str(e)}")
            raise
    
    def validate_model(self, num_predictions=30):
        """Validate model on past data"""
        try:
            if not os.path.exists('gold_model.h5'):
                raise Exception("Model not found. Train the model first.")
            
            self.model = load_model('gold_model.h5')
            if os.path.exists('scaler.pkl'):
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            df = self.fetch_gold_data(period='6mo')
            df = self.calculate_indicators(df)
            
            features = ['Close', 'Volume', 'MA_10', 'MA_30', 'RSI', 'MACD']
            data = df[features].values
            scaled_data = self.scaler.fit_transform(data)
            
            results = []
            start_idx = len(scaled_data) - num_predictions - self.sequence_length
            
            for i in range(start_idx, len(scaled_data) - self.sequence_length):
                sequence = scaled_data[i:i+self.sequence_length]
                sequence = np.reshape(sequence, (1, self.sequence_length, len(features)))
                
                prediction_scaled = self.model.predict(sequence, verbose=0)
                dummy = np.zeros((1, len(features)))
                dummy[0, 0] = prediction_scaled[0, 0]
                predicted_price = self.scaler.inverse_transform(dummy)[0, 0]
                
                actual_price = df['Close'].iloc[i + self.sequence_length]
                date = df.index[i + self.sequence_length].strftime('%Y-%m-%d')
                error = abs(predicted_price - actual_price)
                error_percent = (error / actual_price) * 100
                
                results.append({
                    'date': date,
                    'actual': float(actual_price),
                    'predicted': float(predicted_price),
                    'error': float(error),
                    'error_percent': float(error_percent)
                })
            
            avg_error = np.mean([r['error'] for r in results])
            avg_error_percent = np.mean([r['error_percent'] for r in results])
            accuracy = 100 - avg_error_percent
            
            return {
                'predictions': results,
                'summary': {
                    'total_predictions': len(results),
                    'avg_error': float(avg_error),
                    'avg_error_percent': float(avg_error_percent),
                    'accuracy': float(accuracy)
                }
            }
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            raise

if __name__ == "__main__":
    predictor = GoldPredictor()
    predictor.train(epochs=50)
