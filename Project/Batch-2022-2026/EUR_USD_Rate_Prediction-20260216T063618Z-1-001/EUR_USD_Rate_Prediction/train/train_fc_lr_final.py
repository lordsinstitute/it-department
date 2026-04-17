# ------------------------------------
# FIX matplotlib backend (Windows)
# ------------------------------------
import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------
# Paths
# ------------------------------------
DATA_PATH = "../EUR_USD_Last_5_Years.csv"
OUTPUT_DIR = "../static/outputs_lr"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------
# Load & Prepare Data
# ------------------------------------
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Feature engineering
df['SMA_20'] = df['Close'].rolling(20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

df.dropna(inplace=True)

# Target: next-day Close
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

df.to_csv("../XY_train.csv")

features = ['Close', 'SMA_20', 'EMA_20']
X = df[features]
y = df['Target']

# ------------------------------------
# Train / Validation Split (Last 2 Months)
# ------------------------------------
split_date = df.index.max() - pd.DateOffset(months=2)

X_train = X[X.index < split_date]
y_train = y[y.index < split_date]

X_val = X[X.index >= split_date]
y_val = y[y.index >= split_date]

# ------------------------------------
# Scaling
# ------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler & feature list
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(features, os.path.join(OUTPUT_DIR, "feature_columns.pkl"))

# ------------------------------------
# Train Linear Regression
# ------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, os.path.join(OUTPUT_DIR, "linear_regression_model.pkl"))

# ------------------------------------
# Predictions
# ------------------------------------
y_pred = model.predict(X_val_scaled)

# ------------------------------------
# Performance Metrics
# ------------------------------------
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2 Score"],
    "Value": [rmse, mae, r2]
})

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "linear_regression_metrics.csv"),
    index=False
)

with open(os.path.join(OUTPUT_DIR, "linear_regression_metrics.txt"), "w") as f:
    f.write(metrics_df.to_string(index=False))

# ------------------------------------
# Plot Actual vs Predicted
# ------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_val.values, label="Actual Close", linewidth=2)
plt.plot(y_pred, label="Predicted Close", linewidth=2)
plt.title("Linear Regression – EUR/USD Close Price Forecast")
plt.xlabel("Time")
plt.ylabel("EUR/USD Close")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "linear_regression_prediction.png")
)
plt.close()


# ------------------------------------
# Keep only last 1 year of data
# ------------------------------------
df = df[df.index >= (df.index.max() - pd.DateOffset(years=1))]
#df.set_index('Date', inplace=True)

# ------------------------------------
# Training + Actual + Predicted Overlap Plot
# ------------------------------------

# Training actual close
train_actual = y_train.values

# Validation actual & predicted
val_actual = y_val.values
val_predicted = y_pred

# Build aligned arrays
full_actual = np.concatenate([train_actual, val_actual])
full_pred = np.concatenate([np.full(len(train_actual), np.nan), val_predicted])

# Plot
plt.figure(figsize=(15, 6))

plt.plot(full_actual, label="Actual Close (Train + Validation)", linewidth=2)
plt.plot(full_pred, label="Predicted Close (Validation)", linewidth=2)

plt.axvline(
    x=len(train_actual),
    color="black",
    linestyle="--",
    label="Train / Validation Split"
)

plt.title("EUR/USD Close – Training + Actual + Predicted Overlap (Last 1 Year)")
plt.xlabel("Time")
plt.ylabel("EUR/USD Close")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "train_actual_predicted_overlap.png")
)
plt.close()


# ------------------------------------
# Console Summary
# ------------------------------------
print("\nLinear Regression Training Complete ✅")
print(f"RMSE: {rmse}")
print(f"MAE : {mae}")
print(f"R2  : {r2}")
print(f"\nSaved outputs in: {OUTPUT_DIR}")
