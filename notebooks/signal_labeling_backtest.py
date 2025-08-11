import pandas as pd
import numpy as np
from pathlib import Path

# Load feature-engineered data
data_path = Path("../data/processed/stock_prices_features.csv")
df = pd.read_csv(data_path, parse_dates=["Date"])

# Sort to ensure proper alignment
df.sort_values(["SecuritiesCode", "Date"], inplace=True)

# Step 1: Create NextReturn
df["NextReturn"] = df.groupby("SecuritiesCode")["Return"].shift(-1)

# Step 2: Label signals (very simple rule: Buy if NextReturn > 1%)
df["Signal"] = np.where(df["NextReturn"] > 0.01, 1, 0)  # Binary signals

# Step 3: Strategy vs Market Return
df["StrategyReturn"] = df["Signal"] * df["NextReturn"]
df["MarketReturn"] = df["NextReturn"]

# Clean NaNs
df.dropna(subset=["StrategyReturn", "MarketReturn"], inplace=True)

# Step 4: Daily average returns
daily_returns = df.groupby("Date")[["StrategyReturn", "MarketReturn"]].mean()
daily_returns["StrategyCumulative"] = (1 + daily_returns["StrategyReturn"]).cumprod()
daily_returns["MarketCumulative"] = (1 + daily_returns["MarketReturn"]).cumprod()

# Save for visualization
daily_returns.to_csv("../data/processed/strategy_backtest.csv")

print("✅ Strategy backtest data saved to: ../data/processed/strategy_backtest.csv")
print(daily_returns.tail())

### visualize results using Plotly
# import pandas as pd
import plotly.graph_objects as go
# from pathlib import Path

# Load backtest result
df = pd.read_csv("../data/processed/strategy_backtest.csv", parse_dates=["Date"])

# Create Plotly line chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["StrategyCumulative"],
    mode="lines",
    name="📈 Strategy",
    line=dict(color="green")
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["MarketCumulative"],
    mode="lines",
    name="📉 Market",
    line=dict(color="red")
))

fig.update_layout(
    title="Strategy vs Market — Cumulative Return",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    hovermode="x unified",
    template="plotly_white",
    height=500
)

# Save to HTML
fig.write_html("strategy_vs_market_plot.html")
print("✅ Plot saved as: strategy_vs_market_plot.html")
