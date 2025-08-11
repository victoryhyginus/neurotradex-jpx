import pandas as pd
import plotly.graph_objects as go

train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['SectorName'] = stock_list['17SectorName'].str.strip().str.lower().str.capitalize()
train_df = train.merge(stock_list[['SecuritiesCode','SectorName']], on='SecuritiesCode', how='left')

sectors = train_df['SectorName'].dropna().unique()
colors = ['hsl(' + str(h) + ',50%,50%)' for h in range(0, 360, int(360/len(sectors)))]
fig = go.Figure()
for i, sector in enumerate(sorted(sectors)):
    y_data = train_df[train_df['SectorName'] == sector]['Target'] * 100
    fig.add_trace(go.Box(y=y_data, name=sector, marker_color=colors[i]))
fig.update_layout(title='Target Return Distribution by Sector', height=800)
fig.write_html("eda_3_sector_target_distribution.html")