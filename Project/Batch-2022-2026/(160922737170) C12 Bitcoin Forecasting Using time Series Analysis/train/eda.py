import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create Folder
# -----------------------------
output_dir = "../static/eda"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("../bitcoin_last_10_years.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.dropna()

# -----------------------------
# 3. Feature Engineering
# -----------------------------
df['Daily Return %'] = df['Price'].pct_change() * 100
df['MA_30'] = df['Price'].rolling(window=30).mean()
df['MA_100'] = df['Price'].rolling(window=100).mean()

# -----------------------------
# 4. Price Trend
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Price'])
plt.title("Bitcoin Price Trend (10 Years)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig(f"{output_dir}/price_trend.png")
plt.close()

# -----------------------------
# 5. Moving Averages
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Price'])
plt.plot(df['Date'], df['MA_30'])
plt.plot(df['Date'], df['MA_100'])
plt.title("Bitcoin Price with Moving Averages")
plt.legend(["Price", "30 Day MA", "100 Day MA"])
plt.tight_layout()
plt.savefig(f"{output_dir}/moving_averages.png")
plt.close()

# -----------------------------
# 6. Volume Trend
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Vol.'])
plt.title("Bitcoin Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.tight_layout()
plt.savefig(f"{output_dir}/volume_trend.png")
plt.close()

# -----------------------------
# 7. Daily Return Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Daily Return %'].dropna(), bins=50, kde=True)
plt.title("Distribution of Daily Returns")
plt.tight_layout()
plt.savefig(f"{output_dir}/daily_return_distribution.png")
plt.close()

# -----------------------------
# 8. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))
corr = df[['Price','Open','High','Low','Vol.','Daily Return %']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# -----------------------------
# 9. Outlier Detection
# -----------------------------
Q1 = df['Daily Return %'].quantile(0.25)
Q3 = df['Daily Return %'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Daily Return %'] < Q1 - 1.5*IQR) |
              (df['Daily Return %'] > Q3 + 1.5*IQR)]

print("Number of Outliers in Daily Returns:", len(outliers))

# -----------------------------
# 10. Save Processed Data
# -----------------------------
df.to_csv("../bitcoin_eda_processed.csv", index=False)

print("✅ EDA Completed! Plots saved in static/eda/")