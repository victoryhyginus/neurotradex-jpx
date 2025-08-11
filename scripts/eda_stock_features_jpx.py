import pandas as pd

# Load dataset
df = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])

# Get top 5 stocks by total trading volume
top5_codes = df.groupby('SecuritiesCode')['Volume'].sum().nlargest(5).index.tolist()
print("Top 5 SecuritiesCode by trading volume:", top5_codes)



import plotly.graph_objects as go

# Filter only top 5
top5_df = df[df['SecuritiesCode'].isin(top5_codes)]

# Initialize Plotly figure
fig = go.Figure()

for code in top5_codes:
    stock_df = top5_df[top5_df['SecuritiesCode'] == code]
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['Volume'],
        mode='lines',
        name=f"Stock {code}"
    ))

fig.update_layout(
    title="📊 Volume Over Time — Top 5 Most Traded Stocks",
    xaxis_title="Date",
    yaxis_title="Volume",
    template="plotly_white"
)

# Save the chart to an HTML file
fig.write_html("top5_volume_over_time.html")
print("✅ Saved to top5_volume_over_time.html")


# ✅ 1. Close Price Over Time – Top 5 Stocks
import plotly.graph_objects as go

fig = go.Figure()

for code in top5_codes:
    stock_df = top5_df[top5_df['SecuritiesCode'] == code]
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['Close'],
        mode='lines',
        name=f"Stock {code}"
    ))

fig.update_layout(
    title="📈 Close Price Over Time — Top 5 Most Traded Stocks",
    xaxis_title="Date",
    yaxis_title="Closing Price",
    template="plotly_white"
)

fig.write_html("top5_close_price_over_time.html")
print("✅ Saved to top5_close_price_over_time.html")

# ✅ 2. Daily Return Over Time – Top 5 Stocks

# Calculate daily return
top5_df['Return'] = top5_df.groupby('SecuritiesCode')['Close'].pct_change()

fig = go.Figure()
for code in top5_codes:
    stock_df = top5_df[top5_df['SecuritiesCode'] == code]
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['Return'],
        mode='lines',
        name=f"Stock {code}"
    ))

fig.update_layout(
    title="📉 Daily Return Over Time — Top 5 Most Traded Stocks",
    xaxis_title="Date",
    yaxis_title="Return",
    template="plotly_white"
)

fig.write_html("top5_daily_return.html")
print("✅ Saved to top5_daily_return.html")

# ✅ 3. Rolling Volatility (e.g., 10-day Std Dev of Return)
# Compute rolling volatility
top5_df['RollingVolatility'] = top5_df.groupby('SecuritiesCode')['Return'].rolling(10).std().reset_index(0, drop=True)

fig = go.Figure()
for code in top5_codes:
    stock_df = top5_df[top5_df['SecuritiesCode'] == code]
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['RollingVolatility'],
        mode='lines',
        name=f"Stock {code}"
    ))

fig.update_layout(
    title="📊 10-Day Rolling Volatility — Top 5 Most Traded Stocks",
    xaxis_title="Date",
    yaxis_title="Volatility (Std Dev)",
    template="plotly_white"
)

fig.write_html("top5_rolling_volatility.html")
print("✅ Saved to top5_rolling_volatility.html")
