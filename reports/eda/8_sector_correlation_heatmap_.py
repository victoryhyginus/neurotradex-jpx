import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.colors

# === Load and prepare data ===
train = pd.read_csv('../data/train_files/stock_prices.csv', parse_dates=['Date'])
stock_list = pd.read_csv('../data/stock_list.csv')
stock_list['SectorName'] = stock_list['17SectorName'].str.strip().str.lower().str.capitalize()

train_df = train.merge(stock_list[['SecuritiesCode', 'SectorName']], on='SecuritiesCode', how='left')
df_pivot = train_df.pivot_table(index='Date', columns='SectorName', values='Close').dropna().reset_index()

# === Compute correlation matrix ===
corr = df_pivot.corr().round(2)

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
c_mask = np.where(~mask, corr, np.nan)

# Prepare z values and axis labels
z = c_mask[1:, :-1][::-1]
x_labels = corr.columns[:-1].tolist()
y_labels = corr.index[1:][::-1].tolist()
annotation_text = [[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in z]

# === Sector groups definition ===
sector_groups = {
    'Energy': ['Oil & gas', 'Utilities'],
    'Finance': ['Banks', 'Insurance', 'Securities', 'Other financials'],
    'Consumer': ['Foods', 'Retail', 'Textiles'],
    'Industrial': ['Chemicals', 'Construction', 'Machinery', 'Transportation'],
    'Technology': ['Information', 'Telecommunication'],
    'Real Estate': ['Real estate']
}

# Map group to color
group_names = list(sector_groups.keys())
group_colors = plotly.colors.qualitative.Set3[:len(group_names)]
group_color_map = dict(zip(group_names, group_colors))

# === Helper: Find index range of sectors in labels ===
def find_sector_range(labels, sectors):
    idxs = [i for i, lbl in enumerate(labels) if lbl in sectors]
    return (min(idxs), max(idxs)) if idxs else (None, None)

# === Build shapes and annotations for group highlights ===
shapes = []
annotations = []

for group, sectors in sector_groups.items():
    color = group_color_map[group]
    
    # Vertical band (x-axis, top of heatmap)
    x_start, x_end = find_sector_range(x_labels, sectors)
    if x_start is not None:
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=x_labels[x_start], x1=x_labels[x_end],
            y0=1.00, y1=1.06,
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        ))
        annotations.append(dict(
            x=x_labels[(x_start + x_end) // 2],
            y=1.075,
            xref='x', yref='paper',
            text=group,
            showarrow=False,
            font=dict(size=11),
            align='center'
        ))

    # Horizontal band (y-axis, left side)
    y_start, y_end = find_sector_range(y_labels, sectors)
    if y_start is not None:
        shapes.append(dict(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=0.02,
            y0=y_labels[y_end], y1=y_labels[y_start],
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        ))

# === Create annotated heatmap ===
fig = ff.create_annotated_heatmap(
    z=z,
    x=x_labels,
    y=y_labels,
    annotation_text=annotation_text,
    colorscale='Viridis',
    showscale=True,
    hovertemplate='Correlation between %{x} and %{y}: %{z}<extra></extra>'
)

# === Update layout for styling ===
fig.update_layout(
    title='Stock Correlation between Sectors',
    title_font=dict(size=20, family='Arial'),
    font=dict(size=12, family='Arial'),
    margin=dict(l=200, r=50, t=120, b=150),
    width=1000,
    height=900,
    xaxis=dict(tickangle=-45, side='top', ticks='', showgrid=False),
    yaxis=dict(autorange='reversed', ticks='', showgrid=False),
    plot_bgcolor='white',
    shapes=shapes,
    annotations=fig.layout.annotations + tuple(annotations)
)

# === Export to HTML ===
fig.write_html("eda_8_sector_correlation_heatmap_.html")
