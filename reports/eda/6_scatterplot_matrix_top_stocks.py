import pandas as pd
import plotly.figure_factory as ff
import seaborn as sns

top_codes = [4169, 7089, 4582, 2158, 7036]

# Load data
train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['Name'] = stock_list['Name'].str.strip().str.lower().str.capitalize()

# Merge and filter
train_df = train.merge(stock_list[['SecuritiesCode','Name']], on='SecuritiesCode', how='left')
df = train_df[train_df['SecuritiesCode'].isin(top_codes)]

# Pivot data
df_pivot = df.pivot_table(index='Date', columns='Name', values='Close').dropna().reset_index()

# Generate color palette
palette = sns.color_palette("coolwarm", len(df_pivot))
colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1)' for r, g, b in palette]

# Create scatterplot matrix
fig = ff.create_scatterplotmatrix(df_pivot.iloc[:,1:], diag='histogram', name='')
fig.update_traces(marker=dict(color=colors, opacity=0.8, line=dict(width=0.5)))
fig.update_layout(title='Scatterplot Matrix of Top Performing Stocks')

# Save HTML
fig.write_html("eda_6_scatterplot_matrix_top_stocks.html")
