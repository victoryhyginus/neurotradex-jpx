import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['SectorName'] = stock_list['17SectorName'].str.strip().str.lower().str.capitalize()
train_df = train.merge(stock_list[['SecuritiesCode','SectorName']], on='SecuritiesCode', how='left')
train_df['Year'] = train_df['Date'].dt.year

years = {year: train_df[train_df.Year == year].groupby('SectorName')['Target'].mean().mul(100)
         for year in sorted(train_df['Year'].unique(), reverse=True)}
df = pd.concat([v.rename(f"Avg_return_{k}") for k, v in years.items()], axis=1).sort_values(by="Avg_return_2021")

fig = make_subplots(rows=1, cols=len(df.columns), shared_yaxes=True)
for i, col in enumerate(df.columns):
    x = df[col]
    mask = x <= 0
    fig.add_trace(go.Bar(x=x[mask], y=df.index[mask], orientation='h',
                         text=x[mask], marker_color='red', name=col), row=1, col=i+1)
    fig.add_trace(go.Bar(x=x[~mask], y=df.index[~mask], orientation='h',
                         text=x[~mask], marker_color='green', name=col), row=1, col=i+1)
fig.update_layout(title='Sector-wise Average Return by Year', height=600, width=1200, showlegend=False)
fig.write_html("eda_2_sector_avg_return_multiyear.html")