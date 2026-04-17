import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# -----------------------------
# Load & Prepare Data
# -----------------------------
df = pd.read_csv("../EUR_USD_Last_5_Years.csv")

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# -----------------------------
# Moving Averages
# -----------------------------
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# -----------------------------
# 1. Closing Price Trend
# -----------------------------
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'])
plt.title("EUR/USD Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.grid(True)
plt.savefig("../static/eda/close_price_trend.png", dpi=300)
plt.close()

# -----------------------------
# 2. SMA Analysis
# -----------------------------
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['SMA_20'], label='SMA 20')
plt.plot(df.index, df['SMA_50'], label='SMA 50')
plt.plot(df.index, df['SMA_200'], label='SMA 200')
plt.title("Simple Moving Average Analysis")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)
plt.savefig("../static/eda/sma_analysis.png", dpi=300)
plt.close()

# -----------------------------
# 3. EMA Analysis
# -----------------------------
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['EMA_20'], label='EMA 20')
plt.plot(df.index, df['EMA_50'], label='EMA 50')
plt.title("Exponential Moving Average Analysis")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)
plt.savefig("../static/eda/ema_analysis.png", dpi=300)
plt.close()

# -----------------------------
# 4. SMA Crossover Analysis
# -----------------------------
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['SMA_50'], label='SMA 50')
plt.plot(df.index, df['SMA_200'], label='SMA 200')
plt.title("SMA 50 / SMA 200 Crossover")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)
plt.savefig("../static/eda/sma_crossover.png", dpi=300)
plt.close()

# -----------------------------
# 5. SMA vs EMA (20-day)
# -----------------------------
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['SMA_20'], label='SMA 20')
plt.plot(df.index, df['EMA_20'], label='EMA 20')
plt.title("SMA vs EMA (20-day)")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)
plt.savefig("../static/eda/sma_vs_ema_20.png", dpi=300)
plt.close()

print("✅ All moving average analysis graphs saved successfully.")
