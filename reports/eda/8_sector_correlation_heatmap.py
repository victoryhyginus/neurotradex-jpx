import pandas as pd
import numpy as np
import plotly.figure_factory as ff

# Load and prepare data
train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['SectorName'] = stock_list['17SectorName'].str.strip().str.lower().str.capitalize()
train_df = train.merge(stock_list[['SecuritiesCode', 'SectorName']], on='SecuritiesCode', how='left')

# Pivot table by Date and SectorName
df_pivot = train_df.pivot_table(index='Date', columns='SectorName', values='Close').dropna().reset_index()

# Compute correlation matrix
corr = df_pivot.corr().round(2)

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
c_mask = np.where(~mask, corr, np.nan)

# Prepare lower triangle for annotated heatmap
z = c_mask[1:, :-1][::-1]
x_labels = corr.columns[:-1].tolist()
y_labels = corr.index[1:][::-1].tolist()

# Format annotations (replace nan with blank)
annotation_text = [[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in z]

# Create annotated heatmap
fig = ff.create_annotated_heatmap(
    z=z,
    x=x_labels,
    y=y_labels,
    annotation_text=annotation_text,
    colorscale='Viridis',
    showscale=True,
    hovertemplate='Correlation between %{x} and %{y}: %{z}<extra></extra>'
)

# Customize layout to match Kaggle-like style
fig.update_layout(
    title='Stock Correlation between Sectors',
    title_font=dict(size=20, family='Arial'),
    font=dict(size=12, family='Arial'),
    margin=dict(l=200, r=50, t=100, b=150),
    width=1000,
    height=900,
    xaxis=dict(
        tickangle=-45,
        side='top',
        ticks='',
        showgrid=False
    ),
    yaxis=dict(
        autorange='reversed',
        ticks='',
        showgrid=False
    ),
    plot_bgcolor='white'
)

# Save as HTML
fig.write_html("eda_8_sector_correlation_heatmap.html")
