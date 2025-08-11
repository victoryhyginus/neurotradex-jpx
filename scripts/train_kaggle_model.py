#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LightGBM for JPX Tokyo Stock Exchange Prediction using our v2 pipeline:
- AdjustedClose
- Kelli-style features (Returns, MAs, EMAs, Volatility) + OHLC lag1
- Optional horizon
Saves:
  - lgbm_jpx_model.pkl
  - feature_list.json
"""

import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------
# Feature engineering (same as submission)
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

    # Lagged OHLC
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
    if "Volume" in df.columns:
        cols.append("Volume")
    return cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="/kaggle/input/jpx-tokyo-stock-exchange-prediction")
    ap.add_argument("--start-date", type=str, default=None, help="e.g., 2019-01-01 to limit training window")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default="./model_artifacts")
    args = ap.parse_args()

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(
        Path(args.data_dir) / "train_files" / "stock_prices.csv",
        usecols=["Date","SecuritiesCode","Open","High","Low","Close","Volume","AdjustmentFactor"]
    )
    prices["Date"] = pd.to_datetime(prices["Date"])
    if args.start_date:
        prices = prices[prices["Date"] >= pd.to_datetime(args.start_date)].copy()

    adj = adjust_price(prices)
    feat = create_features(adj)
    feat = build_target(feat, col="AdjustedClose", horizon=args.horizon)
    feat = feat.dropna(subset=["Target"]).reset_index(drop=True)

    feats = feature_columns(feat)
    X = feat[feats].fillna(0.0)
    y = feat["Target"].values

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
    gbm.fit(X, y, feature_name=list(X.columns))

    importances = gbm.booster_.feature_importance(importance_type="gain")
    names = list(gbm.booster_.feature_name())
    fi = pd.Series(importances, index=names).sort_values(ascending=False)
    fi.to_csv(outdir / "feature_importance_gain.csv", header=["gain"])

    joblib.dump(gbm, outdir / "lgbm_jpx_model.pkl")
    (outdir / "feature_list.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"- Model: {outdir / 'lgbm_jpx_model.pkl'}")
    print(f"- Feature list: {outdir / 'feature_list.json'}")
    print(f"- Gain importances: {outdir / 'feature_importance_gain.csv'}")

if __name__ == "__main__":
    main()
