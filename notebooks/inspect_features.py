import pandas as pd

# Load the processed data
df = pd.read_csv('../data/processed/stock_prices_features.csv', parse_dates=['Date'])

# Show structure
print("🔢 Shape:", df.shape)
print("\n🧱 Columns:")
print(df.columns.tolist())

# Show sample rows
print("\n📋 Sample rows:")
print(df.head())

# Check for NaNs
print("\n🧼 Missing values per column:")
print(df.isnull().sum())

# Summary statistics
print("\n📊 Summary stats:")
print(df.describe())
