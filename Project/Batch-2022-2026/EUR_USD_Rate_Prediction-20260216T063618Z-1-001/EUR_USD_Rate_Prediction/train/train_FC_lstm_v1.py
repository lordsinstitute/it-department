# ===============================
# FX EUR/USD GRU Prediction
# ===============================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # FIX backend error
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# ===============================
# Paths
# ===============================
DATA_PATH = "../EUR_USD_Last_5_Years.csv"
OUTPUT_DIR = "../static/lstm_v1"
MODEL_PATH = os.path.join(OUTPUT_DIR, "gru_forex_model.h5")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Load Data
# ===============================
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# ===============================
# Feature Engineering (KEY FIX)
# ===============================
df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

FEATURES = ['Open', 'High', 'Low', 'Close']
TARGET = 'Return'

X_data = df[FEATURES].values
y_data = df[TARGET].values.reshape(-1, 1)

# ===============================
# Scaling
# ===============================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data)

# ===============================
# Sequence Creation
# ===============================
LOOKBACK = 90

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOKBACK)

# ===============================
# Train / Validation Split
# Last 2 months as validation
# ===============================
val_days = 60
X_train = X_seq[:-val_days]
y_train = y_seq[:-val_days]
X_val = X_seq[-val_days:]
y_val = y_seq[-val_days:]

# ===============================
# Build GRU Model
# ===============================
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

# ===============================
# Train
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# Save Model
# ===============================
model.save(MODEL_PATH)

# ===============================
# Predictions
# ===============================
y_pred_scaled = model.predict(X_val)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_val)


# ===============================
# Overlapping Train + Actual + Predicted Plot
# ===============================

# Prepare full timeline
train_returns = scaler_y.inverse_transform(y_train)
val_returns = y_true
val_predictions = y_pred

# Build aligned arrays
full_actual = np.empty(len(train_returns) + len(val_returns))
full_actual[:] = np.nan

full_predicted = np.empty_like(full_actual)
full_predicted[:] = np.nan

# Fill values
full_actual[:len(train_returns)] = train_returns.flatten()
full_actual[len(train_returns):] = val_returns.flatten()

full_predicted[len(train_returns):] = val_predictions.flatten()

# Plot
plt.figure(figsize=(15, 6))

plt.plot(full_actual, label="Actual (Train + Validation)", color='blue')
plt.plot(
    range(len(train_returns), len(full_actual)),
    val_predictions,
    label="Predicted (Validation)",
    color='red'
)

plt.axvline(
    x=len(train_returns),
    linestyle='--',
    color='black',
    label='Train / Validation Split'
)

plt.title("EUR/USD Returns – Training + Actual + Predicted Overlap")
plt.xlabel("Time Steps")
plt.ylabel("Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "train_actual_predicted_overlap.png"))
plt.close()


# ===============================
# Metrics
# ===============================
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2 Score"],
    "Value": [rmse, mae, r2]
})

metrics_df.to_csv(os.path.join(OUTPUT_DIR, "gru_metrics.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "gru_metrics.txt"), "w") as f:
    f.write(metrics_df.to_string(index=False))

# ===============================
# Plot Prediction vs Actual
# ===============================
plt.figure(figsize=(14, 6))
plt.plot(y_true, label='Actual Returns')
plt.plot(y_pred, label='Predicted Returns')
plt.title("EUR/USD Return Prediction – GRU")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "prediction_vs_actual.png"))
plt.close()

# ===============================
# Plot Training Loss
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("GRU Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
plt.close()

# ===============================
# Console Summary
# ===============================
print("\nTraining Complete ✅")
print(f"RMSE: {rmse}")
print(f"MAE : {mae}")
print(f"R2  : {r2}")
print(f"\nSaved outputs in: {OUTPUT_DIR}")
