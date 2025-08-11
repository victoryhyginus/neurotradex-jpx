#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPX Phase 1+2: EDA + Feature Engineering (Ready-to-Run Script)

What this script does
---------------------
1) Loads JPX datasets (train + supplemental) and stock_list metadata
2) Cleans, merges, and computes Adjusted Close (split-adjusted) using the official-style method
3) Builds classic technical features per stock: Returns, Moving Averages, EMAs, Volatility (5/10/20/30/50-day)
4) Recreates sector-level EDA: correlation heatmap (PNG) and an interactive Plotly dashboard (HTML)
5) Saves a modeling-ready CSV with features

Usage
-----
python JPX_phase1_2_eda_features.py \
  --data-dir ./data \
  --out-dir ./artifacts \
  --start-date 2020-12-29

Expected files inside --data-dir:
  - train_files/stock_prices.csv
  - supplemental_files/stock_prices.csv
  - stock_list.csv

Outputs in --out-dir:
  - combined_prices.csv
  - prices_adjusted.csv
  - price_features.csv
  - eda_sector_correlation.png
  - sector_features_dashboard.html

Notes
-----
- Correlation heatmap is computed on sector-average *AdjustedClose* levels (daily) for simplicity.
  If you prefer correlation of *returns*, change `CORR_ON_RETURNS = True` below.
- Plotly dashboard is saved as an HTML file that you can open in your browser.
"""

import argparse
import warnings
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# Avoid issues on some environments
plt.switch_backend("Agg")

import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import unicodedata


warnings.filterwarnings("ignore", category=FutureWarning)

# =============================
# Config
# =============================
PERIODS = [5, 10, 20, 30, 50]
CORR_ON_RETURNS = False  # set True to compute sector correlation on returns instead of price levels


# =============================
# Helpers
# =============================
def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_datasets(data_dir: Path):
    train_fp = data_dir / "train_files" / "stock_prices.csv"
    supp_fp = data_dir / "supplemental_files" / "stock_prices.csv"
    stock_list_fp = data_dir / "stock_list.csv"

    if not train_fp.exists():
        raise FileNotFoundError(f"Missing file: {train_fp}")
    if not supp_fp.exists():
        raise FileNotFoundError(f"Missing file: {supp_fp}")
    if not stock_list_fp.exists():
        raise FileNotFoundError(f"Missing file: {stock_list_fp}")

    train = pd.read_csv(train_fp, parse_dates=["Date"])
    supp = pd.read_csv(supp_fp, parse_dates=["Date"])
    stock_list = pd.read_csv(stock_list_fp)

    # Clean sector name if present
    if "17SectorName" in stock_list.columns:
        stock_list["SectorName"] = (
            stock_list["17SectorName"].astype(str).str.strip().str.lower().str.capitalize()
        )
    elif "SectorName" not in stock_list.columns:
        # Fallback
        stock_list["SectorName"] = "Unknown"

    return train, supp, stock_list


def adjust_price(price: pd.DataFrame) -> pd.DataFrame:
    """
    Create AdjustedClose using cumulative AdjustmentFactor (split/reverse-split adjusted).
    Mirrors the approach used in many JPX starter notebooks.

    Args:
        price (pd.DataFrame): Must contain columns ['Date','SecuritiesCode','AdjustmentFactor','Close']

    Returns:
        pd.DataFrame with new columns:
            - CumulativeAdjustmentFactor
            - AdjustedClose
    """
    df = price.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date", ascending=False).copy()
        g["CumulativeAdjustmentFactor"] = g["AdjustmentFactor"].cumprod()
        # Quantize to 0.1 as in example
        g["AdjustedClose"] = (g["CumulativeAdjustmentFactor"] * g["Close"]).map(
            lambda x: float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        )
        g = g.sort_values("Date").copy()
        g.loc[g["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        g["AdjustedClose"] = g["AdjustedClose"].ffill()
        return g

    df = df.sort_values(["SecuritiesCode", "Date"]).copy()
    df = df.groupby("SecuritiesCode", group_keys=False).apply(_per_code)
    return df


def create_features(df: pd.DataFrame, col: str = "AdjustedClose") -> pd.DataFrame:
    """
    Create technical features per stock for PERIODS.
    - Return: pct change over window
    - MovingAvg: rolling mean
    - ExpMovingAvg: EWMA
    - Volatility: rolling std of log returns

    Uses groupby-transform to preserve index alignment.
    """
    out = df.copy()

    # Precompute log returns for volatility
    out["log_ret"] = out.groupby("SecuritiesCode")[col].transform(lambda x: np.log(x).diff())

    for p in PERIODS:
        # Returns over p days
        out[f"Return_{p}Day"] = out.groupby("SecuritiesCode")[col].transform(lambda x: x.pct_change(p))

        # Simple moving average
        out[f"MovingAvg_{p}Day"] = out.groupby("SecuritiesCode")[col].transform(
            lambda x: x.rolling(window=p, min_periods=1).mean()
        )

        # Exponential moving average
        out[f"ExpMovingAvg_{p}Day"] = out.groupby("SecuritiesCode")[col].transform(
            lambda x: x.ewm(span=p, adjust=False).mean()
        )

        # Volatility (std of log returns) using p-day rolling window
        out[f"Volatility_{p}Day"] = out.groupby("SecuritiesCode")["log_ret"].transform(
            lambda x: x.rolling(window=p, min_periods=1).std()
        )

    out.drop(columns=["log_ret"], inplace=True)
    return out


def sector_correlation_plot(prices_adj: pd.DataFrame, stock_list: pd.DataFrame, out_png: Path, on_returns: bool = False):
    """
    Build a sector-by-sector correlation heatmap and save as PNG.
    - If on_returns is False: correlates sector-average AdjustedClose levels.
    - If on_returns is True : correlates sector-average daily returns of AdjustedClose.
    """
    df = prices_adj.merge(stock_list[["SecuritiesCode", "SectorName"]], on="SecuritiesCode", how="left")
    # Sector daily average series
    daily = (
        df.groupby(["Date", "SectorName"])["AdjustedClose"]
        .mean()
        .unstack("SectorName")
        .sort_index()
    )

    if on_returns:
        daily = daily.pct_change()
    corr = daily.corr()

    # Normalize any full-width chars in labels
    corr.index  = [unicodedata.normalize("NFKC", s) for s in corr.index]
    corr.columns = [unicodedata.normalize("NFKC", s) for s in corr.columns]

    # Matplotlib heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr, aspect="auto")
    plt.title("Sector Correlation Heatmap" + (" (Returns)" if on_returns else " (AdjustedClose)"))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def sector_features_dashboard(price_names: pd.DataFrame, out_html: Path, start_date: str = "2020-12-29"):
    """
    Reproduce Kelli-style Plotly dashboard for MA/EMA/Return/Volatility aggregated by sector (mean).
    Saves a single HTML file.
    """
    # Filter by start date
    price_names = price_names.copy()
    price_names = price_names[price_names["Date"] >= pd.to_datetime(start_date)]
    price_names = price_names.sort_values("Date")

    features = ["MovingAvg", "ExpMovingAvg", "Return", "Volatility"]
    names = ["Average", "Exp. Moving Average", "Period", "Volatility"]

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(
            "Adjusted Close Moving Average",
            "Exponential Moving Average",
            "Stock Return",
            "Stock Volatility",
        ),
    )

    sectors = price_names["SectorName"].dropna().unique().tolist()
    buttons = []

    for i, sector in enumerate(sectors):
        sector_df = price_names[price_names["SectorName"] == sector].copy()

        # The first two subplots include a baseline AdjustedClose mean (no period label)
        periods_top = [0, 10, 30, 50]  # include baseline
        periods_bottom = [10, 30, 50]

        # Colors/dashes (kept simple for clarity)
        vivid = px.colors.qualitative.Vivid
        bold = px.colors.qualitative.Bold

        # Row/col helper
        def add_trace(row, col, series, name, color, dash=None, showlegend=False, legendgroup=None, visible=None):
            fig.add_trace(
                go.Scatter(
                    x=series.index, y=series.values, mode="lines",
                    name=name, line=dict(width=2, dash=(dash if dash else "solid")),
                    marker=dict(color=color),
                    showlegend=showlegend, legendgroup=legendgroup,
                    visible=visible,
                ),
                row=row, col=col,
            )

        # Build traces
        for j, (feature, nm) in enumerate(zip(features, names)):
            row = 1 if j < 2 else 2
            col = 1 if j % 2 == 0 else 2
            if j < 2:
                periods = periods_top
                palette = vivid
                dashes = ["solid", "dash", "longdash", "dashdot"]
            else:
                periods = periods_bottom
                palette = bold[1:]
                dashes = ["solid", "dash", "longdash"]  # not used below

            for k, p in enumerate(periods):
                if (k == 0) and (j < 2):  # baseline AdjustedClose average
                    plot_data = sector_df.groupby("Date")["AdjustedClose"].mean()
                    name_label = "Adjusted Close"
                    dash = "solid"
                    color = palette[(k + 1) % len(palette)]
                else:
                    series_col = f"{feature}_{p}Day"
                    series = sector_df.groupby("Date")[series_col].mean()
                    # scale returns (%) for readability
                    if j >= 2 and "Return" in feature:
                        series = series * 100.0
                    name_label = f"{p}-day {nm}"
                    dash = ("solid" if j >= 2 else dashes[min(k, len(dashes) - 1)])
                    color = palette[(k + 1) % len(palette)]

                add_trace(
                    row=row, col=col,
                    series=plot_data if (k == 0 and j < 2) else series,
                    name=name_label,
                    color=color,
                    dash=dash if j < 2 else None,
                    showlegend=(j in [0, 2]),  # show legend only for first plot of each row
                    legendgroup=f"group{row}",
                    visible=(True if i == 0 else False),
                )

        # Visibility control for each sector (14 traces per sector)
        traces_per_sector = 14
        total_traces = traces_per_sector * len(sectors)
        visibility = [False] * total_traces
        start_idx = i * traces_per_sector
        for idx in range(start_idx, start_idx + traces_per_sector):
            visibility[idx] = True

        buttons.append(
            dict(
                label=sector,
                method="update",
                args=[{"visible": visibility}],
            )
        )

    fig.update_layout(
        title="Stock Price Moving Average, Return,<br>and Volatility by Sector",
        hovermode="x unified",
        height=800,
        width=1200,
        legend_title_text="Period",
        updatemenus=[
            dict(
                active=0, type="dropdown", buttons=buttons,
                xanchor="left", yanchor="bottom", y=1.105, x=0.01
            )
        ],
        margin=dict(t=150),
    )

    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main():
    parser = argparse.ArgumentParser(description="JPX Phase 1+2: EDA + Feature Engineering")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--out-dir", type=str, default="./artifacts", help="Path to output directory")
    parser.add_argument("--start-date", type=str, default="2020-12-29", help="Start date for dashboard (YYYY-MM-DD)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    # 1) Load
    train, supp, stock_list = read_datasets(data_dir)

    # 2) Combine
    prices = pd.concat([train, supp], ignore_index=True).sort_values(["SecuritiesCode", "Date"])
    prices.to_csv(out_dir / "combined_prices.csv", index=False)

    # 3) Adjusted Close
    prices_adj = adjust_price(prices)
    prices_adj.to_csv(out_dir / "prices_adjusted.csv", index=False)

    # 4) Features
    feat = create_features(prices_adj, col="AdjustedClose")

    # Drop rarely-used columns if present
    drop_cols = [c for c in ["RowId", "SupervisionFlag", "AdjustmentFactor", "CumulativeAdjustmentFactor", "Close"] if c in feat.columns]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    # 5) Merge names & sectors
    price_names = feat.merge(
        stock_list[["SecuritiesCode", "Name", "SectorName"]],
        on="SecuritiesCode", how="left"
    )

    # Save modeling-ready features
    price_names.to_csv(out_dir / "price_features.csv", index=False)

    # 6) EDA: Sector correlation heatmap
    corr_png = out_dir / "eda_sector_correlation.png"
    sector_correlation_plot(prices_adj, stock_list, corr_png, on_returns=CORR_ON_RETURNS)

    # 7) Dashboard
    dashboard_html = out_dir / "sector_features_dashboard.html"
    sector_features_dashboard(price_names, dashboard_html, start_date=args.start_date)

    print("Done.")
    print(f"- Combined prices:           {out_dir / 'combined_prices.csv'}")
    print(f"- Adjusted prices:           {out_dir / 'prices_adjusted.csv'}")
    print(f"- Feature dataset:           {out_dir / 'price_features.csv'}")
    print(f"- Sector correlation heatmap:{corr_png}")
    print(f"- Plotly dashboard:          {dashboard_html}")


if __name__ == "__main__":
    main()
