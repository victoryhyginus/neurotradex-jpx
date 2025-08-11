import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])

returns = df.groupby('Date')['Target'].mean().mul(100).rename('Average Return')
close_avg = df.groupby('Date')['Close'].mean().rename('Closing Price')
vol_avg = df.groupby('Date')['Volume'].mean().rename('Volume')

fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
colors = ['blue', 'red', 'green']
for i, series in enumerate([returns, close_avg, vol_avg]):
    fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines',
                             name=series.name, marker_color=colors[i]), row=i+1, col=1)

fig.update_layout(template='plotly_white',
                  title='JPX Market Overview: Return, Price, Volume',
                  hovermode='x unified', height=700, showlegend=False)
fig.write_html("eda_1_market_overview.html")