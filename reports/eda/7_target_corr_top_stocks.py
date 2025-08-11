import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib

train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['Name'] = stock_list['Name'].str.strip().str.lower().str.capitalize()

train_df = train.merge(stock_list[['SecuritiesCode','Name']], on='SecuritiesCode', how='left')
corr = train_df.groupby('SecuritiesCode')[['Target','Close']].corr().unstack().iloc[:,1]
stocks = corr.nlargest(10).rename("Return").reset_index()
stocks = stocks.merge(train_df[['Name','SecuritiesCode']], on='SecuritiesCode').drop_duplicates()

pal = sns.color_palette("magma_r", 14).as_hex()
rgb = ['rgba'+str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]

fig = go.Figure()
fig.add_trace(go.Bar(x=stocks['Name'], y=stocks['Return'],
                     text=stocks['Return'].round(2), textposition='outside',
                     marker=dict(color=rgb, line=dict(color=pal, width=1))))
fig.update_layout(title='Stocks Most Correlated with Target Variable',
                  yaxis_title='Correlation', height=500)
fig.write_html("eda_7_target_corr_top_stocks.html")