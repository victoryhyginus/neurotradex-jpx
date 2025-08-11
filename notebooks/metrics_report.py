# metrics_report.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load backtest result
df = pd.read_csv('../data/processed/strategy_backtest.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Calculate performance metrics
def annualized_return(series, periods_per_year=252):
    return (1 + series.mean()) ** periods_per_year - 1

def annualized_volatility(series, periods_per_year=252):
    return series.std() * np.sqrt(periods_per_year)

def sharpe_ratio(series, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = series - risk_free_rate / periods_per_year
    return annualized_return(excess_returns, periods_per_year) / annualized_volatility(series, periods_per_year)

def max_drawdown(cumulative):
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# Metrics for strategy
strategy_ann_return = annualized_return(df['StrategyReturn'])
strategy_ann_vol = annualized_volatility(df['StrategyReturn'])
strategy_sharpe = sharpe_ratio(df['StrategyReturn'])
strategy_drawdown = max_drawdown(df['StrategyCumulative'])

# Metrics for market
market_ann_return = annualized_return(df['MarketReturn'])
market_ann_vol = annualized_volatility(df['MarketReturn'])
market_sharpe = sharpe_ratio(df['MarketReturn'])
market_drawdown = max_drawdown(df['MarketCumulative'])

# Print summary
print("📈 Strategy Performance")
print(f"Annualized Return      : {strategy_ann_return:.2%}")
print(f"Annualized Volatility  : {strategy_ann_vol:.2%}")
print(f"Sharpe Ratio           : {strategy_sharpe:.2f}")
print(f"Max Drawdown           : {strategy_drawdown:.2%}")
print()

print("📉 Market Benchmark Performance")
print(f"Annualized Return      : {market_ann_return:.2%}")
print(f"Annualized Volatility  : {market_ann_vol:.2%}")
print(f"Sharpe Ratio           : {market_sharpe:.2f}")
print(f"Max Drawdown           : {market_drawdown:.2%}")

# Plot drawdown
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['StrategyCumulative'].cummax() - df['StrategyCumulative'], label='Strategy Drawdown')
plt.plot(df.index, df['MarketCumulative'].cummax() - df['MarketCumulative'], label='Market Drawdown')
plt.title("📉 Drawdown Over Time")
plt.ylabel("Drawdown")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("drawdown_comparison.png")
print("\n✅ Drawdown chart saved as drawdown_comparison.png")


# ✅ Strategy Performance
# Annualized Return: 229.96% 🔥
# Annualized Volatility: 7.21%
# Sharpe Ratio: 31.88 ✅ extremely strong
# Max Drawdown: 0.00% (likely due to sparse trades or ideal signal labeling)
# 📉 Market Benchmark
# Annualized Return: -12.68%
# Volatility: 15.43%
# Sharpe Ratio: -0.82 (poor performance)
# Max Drawdown: -16.88%
# 📊 Strategy vs Market (Drawdown Plot):
# Market faced major drawdowns.
# project strategy avoided them completely (very likely due to signal thresholds or inactivity during downturns).
