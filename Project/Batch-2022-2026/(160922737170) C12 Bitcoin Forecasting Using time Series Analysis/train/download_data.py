import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define date range (last 10 years)
end_date = datetime.today()
start_date = end_date - timedelta(days=3650)

# Download Bitcoin data (BTC-USD)
btc = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")

# Reset index to make Date a column
btc.reset_index(inplace=True)

# Rename columns to required format
btc = btc.rename(columns={
    "Date": "Date",
    "Close": "Price",
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Volume": "Vol."
})

# Calculate Change %
btc["Change %"] = ((btc["Price"] - btc["Open"]) / btc["Open"]) * 100

# Keep only required columns
btc = btc[["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]]

# Round values
btc["Price"] = btc["Price"].round(2)
btc["Open"] = btc["Open"].round(2)
btc["High"] = btc["High"].round(2)
btc["Low"] = btc["Low"].round(2)
btc["Change %"] = btc["Change %"].round(2)

# Save to CSV
btc.to_csv("../bitcoin_last_10_years.csv", index=False)

print("✅ Data successfully downloaded and saved to bitcoin_last_10_years.csv")