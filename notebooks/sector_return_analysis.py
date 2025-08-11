# sector_return_analysis.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Load datasets ===
stock_prices = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')

# === Clean sector and stock names ===
stock_list['SectorName'] = [i.strip().lower().capitalize() for i in stock_list['17SectorName']]
stock_list['Name'] = [i.strip().lower().capitalize() for i in stock_list['Name']]

# === Merge to stock_prices ===
df = stock_prices.merge(stock_list[['SecuritiesCode', 'Name', 'SectorName']], 
                        on='SecuritiesCode', how='left')
df['Year'] = df['Date'].dt.year

# === Calculate average return by sector per year ===
years = {year: pd.DataFrame() for year in df['Year'].unique()[::-1]}
for year in years:
    df_year = df[df['Year'] == year]
    sector_mean = df_year.groupby('SectorName')['Target'].mean().mul(100)
    years[year] = sector_mean.rename(f"Avg_return_{year}")

# === Combine all yearly sector returns ===
df_sector_returns = pd.concat([years[year].to_frame() for year in years], axis=1)
df_sector_returns = df_sector_returns.sort_values(by="Avg_return_2021", ascending=True)

# === Plotting ===
fig = make_subplots(rows=1, cols=len(df_sector_returns.columns), shared_yaxes=True)

for i, col in enumerate(df_sector_returns.columns):
    x = df_sector_returns[col]
    mask = x <= 0
    fig.add_trace(go.Bar(x=x[mask], y=df_sector_returns.index[mask], orientation='h',
                         text=x[mask], texttemplate='%{text:.2f}%', textposition='auto',
                         hovertemplate='Average Return in %{y} = %{x:.2f}%',
                         marker=dict(color='red', opacity=0.7), name=col[-4:]),
                  row=1, col=i+1)
    fig.add_trace(go.Bar(x=x[~mask], y=df_sector_returns.index[~mask], orientation='h',
                         text=x[~mask], texttemplate='%{text:.2f}%', textposition='auto',
                         hovertemplate='Average Return in %{y} = %{x:.2f}%',
                         marker=dict(color='green', opacity=0.7), name=col[-4:]),
                  row=1, col=i+1)
    fig.update_xaxes(title=f'{col[-4:]} Returns', row=1, col=i+1, showticklabels=False)

fig.update_layout(template='plotly_white',
                  title='📊 Yearly Average Stock Returns by Sector',
                  hovermode='closest',
                  margin=dict(l=250, r=50),
                  height=600, width=1200,
                  showlegend=False)

fig.write_html('sector_returns_by_year.html')
print("✅ Plot saved to: sector_returns_by_year.html")
