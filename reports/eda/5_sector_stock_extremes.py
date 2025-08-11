import pandas as pd
import plotly.graph_objects as go

train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['SectorName'] = stock_list['17SectorName'].str.strip().str.lower().str.capitalize()
stock_list['Name'] = stock_list['Name'].str.strip().str.lower().str.capitalize()

train_df = train.merge(stock_list[['SecuritiesCode','SectorName','Name']], on='SecuritiesCode', how='left')
train_df = train_df[train_df['Date'] > '2020-12-23']

stock = train_df.groupby('Name')['Target'].mean().mul(100)
top = stock.nlargest(7)
low = stock.nsmallest(7)[::-1]
stock_df = pd.concat([top, low], axis=0).reset_index()
stock_df['Sector'] = 'All'

for sector in train_df['SectorName'].dropna().unique():
    s_df = train_df[train_df['SectorName'] == sector]
    s_top = s_df.groupby('Name')['Target'].mean().mul(100).nlargest(7)
    s_low = s_df.groupby('Name')['Target'].mean().mul(100).nsmallest(7)[::-1]
    temp = pd.concat([s_top, s_low], axis=0).reset_index()
    temp['Sector'] = sector
    stock_df = pd.concat([stock_df, temp])

fig = go.Figure()
buttons = []
sectors = stock_df['Sector'].unique()

for i, sec in enumerate(sectors):
    d = stock_df[stock_df['Sector'] == sec]
    mask = d['Target'] > 0
    fig.add_trace(go.Bar(x=d['Name'][mask], y=d['Target'][mask], marker_color='green', name=sec, visible=(i==0)))
    fig.add_trace(go.Bar(x=d['Name'][~mask], y=d['Target'][~mask], marker_color='red', name=sec, visible=(i==0)))
    vis = [False] * len(sectors) * 2
    vis[i*2] = True
    vis[i*2 + 1] = True
    buttons.append(dict(label=sec, method='update', args=[{'visible': vis}]))

fig.update_layout(title='Top and Bottom Performing Stocks by Sector',
                  updatemenus=[dict(buttons=buttons, direction='down')],
                  height=800)
fig.write_html("eda_5_sector_stock_extremes.html")