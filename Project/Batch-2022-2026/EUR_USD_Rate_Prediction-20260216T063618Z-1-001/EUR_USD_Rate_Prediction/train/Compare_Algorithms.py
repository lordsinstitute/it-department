# ------------------------------------
# FIX matplotlib backend (Windows)
# ------------------------------------
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ------------------------------------
# Load & Prepare Data
# ------------------------------------
df = pd.read_csv("../EUR_USD_Last_5_Years.csv")
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
# Scale features (important for SVR)
# ------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ------------------------------------
# Models
# ------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR": SVR(kernel='rbf'),
    "XGBoost": XGBRegressor()
}

results = []
predictions = {}

# ------------------------------------
# Train, Predict & Evaluate
# ------------------------------------
for name, model in models.items():
    if name == "SVR":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    predictions[name] = y_pred

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_val, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
        "R2_Score": r2_score(y_val, y_pred)
    })

# ------------------------------------
# Save Metrics
# ------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("../static/performance/model_performance_metrics.csv", index=False)

# ------------------------------------
# Plot Actual vs Predicted
# ------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_val.index, y_val.values, label="Actual Close", linewidth=2)

for name, preds in predictions.items():
    plt.plot(y_val.index, preds, label=name)

plt.title("EUR/USD Closing Price Prediction (Last 2 Months)")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)
plt.savefig("../static/performance/prediction_comparison.png", dpi=300)
plt.close()

print("✅ Training, validation, and evaluation completed successfully.")
print("📁 Files saved:")
print(" - prediction_comparison.png")
print(" - model_performance_metrics.csv")
