import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the JPX stock price data
df = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])

# Drop rows with missing Close or Target
df = df.dropna(subset=['Close', 'Target'])

# 📊 Group by Date for market-level analysis
daily_grouped = df.groupby('Date')

# Compute statistics
returns = daily_grouped['Target'].mean().mul(100).rename('Average Return (%)')
close_avg = daily_grouped['Close'].mean().rename('Average Close Price')
vol_avg = daily_grouped['Volume'].mean().rename('Average Volume')
stock_count = daily_grouped['SecuritiesCode'].nunique().rename('Number of Traded Stocks')
ma_7d_return = returns.rolling(7).mean().rename('7-Day MA Return')
ma_30d_return = returns.rolling(30).mean().rename('30-Day MA Return')

# 📈 Multi-panel Plotly chart
fig = make_subplots(rows=5, cols=1, shared_xaxes=True)

# Main series
series_list = [returns, close_avg, vol_avg, stock_count]
colors = ['blue', 'red', 'green', 'purple']
for i, series in enumerate(series_list):
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series,
        mode='lines',
        name=series.name,
        marker_color=colors[i]
    ), row=i+1, col=1)

# Add moving averages to top row
fig.add_trace(go.Scatter(
    x=ma_7d_return.index,
    y=ma_7d_return,
    mode='lines',
    name=ma_7d_return.name,
    line=dict(dash='dot', color='gray')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=ma_30d_return.index,
    y=ma_30d_return,
    mode='lines',
    name=ma_30d_return.name,
    line=dict(dash='dash', color='black')
), row=1, col=1)

# Layout
fig.update_layout(
    template='plotly_white',
    title='📊 JPX Market Overview: Return, Price, Volume, Stock Count',
    hovermode='x unified',
    height=1000,
    yaxis1_title='Return (%)',
    yaxis2_title='Close Price',
    yaxis3_title='Volume',
    yaxis4_title='Traded Stocks',
    showlegend=True
)

# Save as interactive HTML
fig.write_html("jpx_market_overview.html")
print("✅ EDA chart saved as jpx_market_overview.html — open it in browser.")
