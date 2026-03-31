import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

ticker = "EURUSD=X"

end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# Download data
eur_usd = yf.download(
    ticker,
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval="1d"
)

# Reset index
eur_usd.reset_index(inplace=True)

# 🔹 FIX: Flatten MultiIndex columns if present
if isinstance(eur_usd.columns, pd.MultiIndex):
    eur_usd.columns = [col[0] for col in eur_usd.columns]

# 🔹 Forex data usually has NO "Adj Close"
expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
eur_usd = eur_usd[expected_cols]

# Save CSV
eur_usd.to_csv("EUR_USD_Last_5_Years.csv", index=False)

print(eur_usd.head())
print("\nColumns:", eur_usd.columns.tolist())
print("Date range:", eur_usd['Date'].min(), "to", eur_usd['Date'].max())
