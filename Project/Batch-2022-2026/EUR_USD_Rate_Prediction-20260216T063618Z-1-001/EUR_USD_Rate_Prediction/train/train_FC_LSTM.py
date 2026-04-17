# ===============================
# Imports
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# Configuration
# ===============================
CSV_PATH = "../EUR_USD_Last_5_Years.csv"   # your Yahoo Finance CSV
TARGET_COL = "Close"
LOOKBACK = 60               # days
EPOCHS = 40
BATCH_SIZE = 32

OUTPUT_DIR = "../static/lstm_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Load Data
# ===============================
df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

data = df[[TARGET_COL]].values

# ===============================
# Train–Validation Split (Last 2 months)
# ===============================
val_days = 60
train_data = data[:-val_days]
val_data = data[-(val_days + LOOKBACK):]

# ===============================
# Scaling
# ===============================
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)

# ===============================
# Sequence Generator
# ===============================
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_val, y_val = create_sequences(val_scaled, LOOKBACK)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# ===============================
# LSTM Model
# ===============================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# ===============================
# Training
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# Prediction
# ===============================
y_pred_scaled = model.predict(X_val)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_actual = scaler.inverse_transform(y_val.reshape(-1, 1))

# ===============================
# Metrics
# ===============================
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

metrics_df = pd.DataFrame({
    "RMSE": [rmse],
    "MAE": [mae],
    "R2 Score": [r2]
})

metrics_df.to_csv(f"{OUTPUT_DIR}/lstm_metrics.csv", index=False)

print("\n📊 Model Performance")
print(metrics_df)

# ===============================
# Plot: Actual vs Predicted
# ===============================
plt.figure(figsize=(12, 6))
plt.plot(df.index[-val_days:], y_actual, label="Actual Close")
plt.plot(df.index[-val_days:], y_pred, label="Predicted Close")
plt.title("EUR → USD Closing Price Prediction (LSTM)")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)

plt.savefig(f"{OUTPUT_DIR}/lstm_actual_vs_predicted.png", dpi=300)
plt.show()

# ===============================
# Save Model
# ===============================
model.save(f"{OUTPUT_DIR}/lstm_eurusd_model.h5")
