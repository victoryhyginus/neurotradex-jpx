#!/usr/bin/env python3
# JPX Kaggle "API Submission" style script
# - Trains a LightGBM model on historical stock_prices
# - In the iter_test loop, updates price history, recomputes features for the current date,
#   scores, ranks, and submits.
#
# Drop this into a Kaggle Notebook / Script and run.
# It expects the standard competition dataset path:
#   /kaggle/input/jpx-tokyo-stock-exchange-prediction/

import numpy as np
import pandas as pd
import lightgbm as lgb
import jpx_tokyo_market_prediction

# ---------------------
# Feature utilities
# ---------------------
def adjust_price(price: pd.DataFrame) -> pd.DataFrame:
    price = price.copy()
    price["Date"] = pd.to_datetime(price["Date"])

    def _one_sec(df):
        df = df.sort_values("Date", ascending=False).copy()
        df["CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        df["AdjustedClose"] = (df["CumulativeAdjustmentFactor"] * df["Close"]).round(1)
        df = df.sort_values("Date")
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        df["AdjustedClose"] = df["AdjustedClose"].ffill()
        return df

    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode", group_keys=False).apply(_one_sec)
    return price


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()
    periods = [5, 10, 20, 30, 50]
    g = df.groupby("SecuritiesCode")

    # Returns
    for p in periods:
        df[f"Return_{p}Day"] = g["AdjustedClose"].pct_change(p)

    # Simple MAs
    for p in periods:
        df[f"MovingAvg_{p}Day"] = g["AdjustedClose"].rolling(p).mean().reset_index(level=0, drop=True)

    # Exponential MAs
    for p in periods:
        df[f"ExpMovingAvg_{p}Day"] = g["AdjustedClose"].transform(lambda s, P=p: s.ewm(span=P, adjust=False).mean())

    # Volatility of log returns
    logret = g["AdjustedClose"].transform(lambda s: np.log(s).diff())
    for p in periods:
        df[f"Volatility_{p}Day"] = logret.groupby(df["SecuritiesCode"]).rolling(p).std().reset_index(level=0, drop=True)

    # Lagged OHLC for a bit of recency
    for c in ["Open", "High", "Low"]:
        if c in df.columns:
            df[f"{c}_lag1"] = g[c].shift(1)

    return df


def build_target(df: pd.DataFrame, col="AdjustedClose", horizon=1) -> pd.DataFrame:
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()
    df["Target"] = df.groupby("SecuritiesCode")[col].shift(-horizon) / df[col] - 1.0
    return df


def feature_columns(df: pd.DataFrame):
    prefixes = ("Return_", "MovingAvg_", "ExpMovingAvg_", "Volatility_")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    for c in ["Open_lag1", "High_lag1", "Low_lag1"]:
        if c in df.columns:
            cols.append(c)
    # Optional: include raw Volume which is often strong in Kelli's chart
    if "Volume" in df.columns:
        cols.append("Volume")
    return cols


# ---------------------
# Load historical data and train once
# ---------------------
DATA_DIR = "/kaggle/input/jpx-tokyo-stock-exchange-prediction"
train_prices = pd.read_csv(f"{DATA_DIR}/train_files/stock_prices.csv")
# Light warmup window to keep runtime sane (adjust if time allows)
# You can remove the date filter to use all history.
train_prices["Date"] = pd.to_datetime(train_prices["Date"])
# Use ~3 years of data for speed. Comment the next line to use all.
# train_prices = train_prices[train_prices["Date"] >= "2019-01-01"]

cols = ["Date","SecuritiesCode","Open","High","Low","Close","Volume","AdjustmentFactor"]
train_prices = train_prices[cols].copy()

adj = adjust_price(train_prices)
feat = create_features(adj)
feat = build_target(feat, col="AdjustedClose", horizon=1)

feat = feat.dropna(subset=["Target"]).reset_index(drop=True)
feats = feature_columns(feat)

X_train = feat[feats].fillna(0.0)
y_train = feat["Target"].values

params = dict(
    n_estimators=500,
    num_leaves=100,
    learning_rate=0.1,
    colsample_bytree=0.9,
    subsample=0.8,
    reg_alpha=0.4,
    random_state=21
)

gbm = lgb.LGBMRegressor(**params)
gbm.fit(X_train, y_train, feature_name=list(X_train.columns))

# ---------------------
# Inference loop (API)
# ---------------------
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

# We'll keep a growing buffer of raw prices to recompute features each day.
# Start it with the historical slice we trained on (last few months is enough).
df_price_raw = train_prices[train_prices["Date"] >= "2021-08-01"].copy()

for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    current_date = pd.to_datetime(prices["Date"].iloc[0])

    # Append today's raw prices and recompute adjusted & features
    df_price_raw = pd.concat([df_price_raw, prices[cols]], ignore_index=True)
    df_price = adjust_price(df_price_raw)
    all_feat = create_features(df_price)

    # Select rows for current_date and build the model's feature frame
    today_feat = all_feat[all_feat["Date"] == current_date].copy()
    X_today = today_feat.reindex(columns=feats).fillna(0.0)

    # Predict and rank
    pred = gbm.predict(X_today)
    ranks = pd.Series(pred).rank(method="first", ascending=False).astype(int) - 1

    # Fill the submission
    sample_prediction["Rank"] = ranks.values

    # Sanity checks required by the competition
    assert sample_prediction["Rank"].notna().all()
    assert sample_prediction["Rank"].min() == 0
    assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1

    # Submit
    env.predict(sample_prediction)
