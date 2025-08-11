import pandas as pd
import plotly.graph_objects as go

train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['SectorName'] = stock_list['17SectorName'].str.strip().str.lower().str.capitalize()
train_df = train.merge(stock_list[['SecuritiesCode','SectorName']], on='SecuritiesCode', how='left')

train_df = train_df[train_df['Date'] > '2020-12-23']
sectors = ['All'] + sorted(train_df['SectorName'].dropna().unique().tolist())
buttons = []
fig = go.Figure()

for i, sector in enumerate(sectors):
    if sector == 'All':
        df_sector = train_df.copy()
    else:
        df_sector = train_df[train_df['SectorName'] == sector]
    open_avg = df_sector.groupby('Date')['Open'].mean()
    high_avg = df_sector.groupby('Date')['High'].mean()
    low_avg = df_sector.groupby('Date')['Low'].mean()
    close_avg = df_sector.groupby('Date')['Close'].mean()
    fig.add_trace(go.Candlestick(x=open_avg.index, open=open_avg, high=high_avg, low=low_avg, close=close_avg,
                                 name=sector, visible=(i==0)))
    vis = [False] * len(sectors)
    vis[i] = True
    buttons.append(dict(label=sector, method='update', args=[{'visible': vis}]))

fig.update_layout(title='Candlestick View by Sector',
                  updatemenus=[dict(buttons=buttons, direction='down')], height=800)
fig.write_html("eda_4_sectorwise_candlestick.html")