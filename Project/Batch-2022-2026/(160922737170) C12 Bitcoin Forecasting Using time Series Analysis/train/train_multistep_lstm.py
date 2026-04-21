import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import save_model
import joblib

LOOKBACK = 30
FUTURE_DAYS = 60

df = pd.read_csv("../bitcoin_last_10_years.csv")
df["Date"] = pd.to_datetime(df["Date"])
prices = df["Price"].values.reshape(-1,1)

# Train/Test split (no leakage)
train_size = int(len(prices) * 0.8)
train_data = prices[:train_size]
test_data = prices[train_size:]

scaler = MinMaxScaler()
scaler.fit(train_data)
joblib.dump(scaler, "../models/scaler.save")

train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# Create sequences
X, y = [], []

for i in range(LOOKBACK, len(train_scaled) - FUTURE_DAYS):
    X.append(train_scaled[i-LOOKBACK:i])
    y.append(train_scaled[i:i+FUTURE_DAYS])

X = np.array(X)
y = np.array(y)

# Build Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(LOOKBACK,1)),
    LSTM(100),
    Dense(FUTURE_DAYS)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, batch_size=32)

model.save("../models/lstm_60day_model.h5")