# feature_engineering_jpx.py

import pandas as pd
import os

# ✅ Update path to match your structure
DATA_PATH = "../data/train_files/stock_prices.csv"
OUTPUT_PATH = "../data/processed/stock_prices_features.csv"

# 📦 Load data
print("📂 Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])

# 🧹 Sort for rolling/lag operations
df.sort_values(by=['SecuritiesCode', 'Date'], inplace=True)

# ➕ Daily return
df['Return'] = df.groupby('SecuritiesCode')['Close'].pct_change()

# 📈 Rolling moving averages
df['MA_5'] = df.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=5).mean())
df['MA_10'] = df.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=10).mean())

# 📉 Rolling volatility
df['Volatility_5'] = df.groupby('SecuritiesCode')['Return'].transform(lambda x: x.rolling(window=5).std())
df['Volatility_10'] = df.groupby('SecuritiesCode')['Return'].transform(lambda x: x.rolling(window=10).std())

# 📊 Price ratios
df['Close/Open'] = df['Close'] / df['Open']
df['High/Low'] = df['High'] / df['Low']

# ⏪ Lag features
df['Lag_1'] = df.groupby('SecuritiesCode')['Close'].shift(1)
df['Lag_3'] = df.groupby('SecuritiesCode')['Close'].shift(3)
df['Lag_5'] = df.groupby('SecuritiesCode')['Close'].shift(5)

# 🚿 Clean
df_fe = df.dropna().reset_index(drop=True)

# 📁 Make sure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 💾 Save
df_fe.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Feature-engineered data saved to: {OUTPUT_PATH}")
print("📊 Final shape:", df_fe.shape)
