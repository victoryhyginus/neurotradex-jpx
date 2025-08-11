#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPX Phase 3 (v2): LightGBM + TimeSeriesSplit + JPX Spread Sharpe

- Loads Phase 2 features (price_features.csv)
- Builds next-day target by default (horizon=1)
- Per-date cross-sectional z-scoring of features (optional)
- Optional sector de-mean of target
- TimeSeriesSplit with large gap to reduce leakage
- LightGBMRegressor baseline (Kelli-style params)
- Computes JPX spread-return Sharpe (top 200 long, bottom 200 short, linear weights)
- Saves fold metrics, OOF predictions, feature importances, and plots

Usage
-----
python JPX_phase3_modeling_v2.py \
  --features-csv ./artifacts/price_features.csv \
  --out-dir ./artifacts_modeling_v2 \
  --horizon 1 \
  --n-splits 10 --gap 10000 \
  --portfolio-size 200 --toprank-weight-ratio 2 \
  --zscore-per-date True \
  --demean-by-sector False
"""

import argparse, json, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
plt.switch_backend("Agg")


def build_target(df: pd.DataFrame, col="AdjustedClose", horizon=1) -> pd.DataFrame:
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()
    df["Target"] = df.groupby("SecuritiesCode")[col].shift(-horizon) / df[col] - 1.0
    df["Target"].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def cross_sectional_zscore(df: pd.DataFrame, feature_cols):
    """Per-date z-score for each feature across stocks."""
    z = df.copy()
    for c in feature_cols:
        g = z.groupby("Date")[c]
        z[c] = (g.transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))).astype(float)
    return z


def de_mean_target_by_sector(df: pd.DataFrame):
    """Subtract sector mean target per date to focus on idiosyncratic signal."""
    if "SectorName" not in df.columns:
        return df
    out = df.copy()
    grp = out.groupby(["Date", "SectorName"])["Target"]
    out["Target"] = out["Target"] - grp.transform("mean")
    return out



def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    JPX evaluation metric: long top-N, short bottom-N (weighted), compute daily spread return Sharpe.
    Handles days with < portfolio_size stocks by shrinking the weights vector accordingly.
    Expects df with columns: Date, Rank, Target.
    """
    def _calc_spread_return_per_day(day_df, portfolio_size, toprank_weight_ratio):
        n = len(day_df)
        if n == 0:
            return 0.0
        k = int(min(portfolio_size, n))
        # Ensure ranks start at 0 and are consecutive for the slice we care about
        # (We don't hard-assert on max rank because some days may be filtered.)
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=k)
        sorted_up = day_df.sort_values(by="Rank")
        sorted_dn = day_df.sort_values(by="Rank", ascending=False)
        purchase = (sorted_up["Target"].head(k).values * weights).sum() / weights.mean()
        short = (sorted_dn["Target"].head(k).values * weights).sum() / weights.mean()
        return float(purchase - short)

    daily = df.groupby("Date").apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    return float(daily.mean() / (daily.std(ddof=0) + 1e-12))


def feature_columns(df: pd.DataFrame):
    tech_prefixes = ("Return_", "MovingAvg_", "ExpMovingAvg_", "Volatility_")
    cols = [c for c in df.columns if c.startswith(tech_prefixes)]
    # Optionally include lagged OHLC if present and already lagged
    for c in ["Open_lag1", "High_lag1", "Low_lag1"]:
        if c in df.columns:
            cols.append(c)
    # Keep only numeric
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def maybe_add_lagged_ohlc(df: pd.DataFrame):
    """If raw OHLC levels are present, add 1-day lagged versions and drop current-day to avoid lookahead."""
    out = df.sort_values(["SecuritiesCode", "Date"]).copy()
    for c in ["Open", "High", "Low"]:
        if c in out.columns:
            out[f"{c}_lag1"] = out.groupby("SecuritiesCode")[c].shift(1)
    return out


def plot_feature_importance(fi_df: pd.DataFrame, out_png: Path, topn=30):
    fi = fi_df.mean(axis=1).sort_values(ascending=False).head(topn)
    plt.figure(figsize=(8, 10))
    fi.iloc[::-1].plot(kind="barh")
    plt.title("LightGBM Feature Importance (avg over folds)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", type=str, default="./artifacts/price_features.csv")
    ap.add_argument("--out-dir", type=str, default="./artifacts_modeling_v2")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--gap", type=int, default=10000)
    ap.add_argument("--portfolio-size", type=int, default=200)
    ap.add_argument("--toprank-weight-ratio", type=float, default=2.0)
    ap.add_argument("--zscore-per-date", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--demean-by-sector", type=lambda x: str(x).lower() == "true", default=False)
    args = ap.parse_args()

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.features_csv, parse_dates=["Date"])
    df = maybe_add_lagged_ohlc(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()

    # Build target
    df = build_target(df, col="AdjustedClose", horizon=args.horizon)

    # Keep rows with observed target
    df = df[df["Target"].notna()].copy()

    # Build feature set
    feats = feature_columns(df)
    if len(feats) == 0:
        raise RuntimeError("No feature columns found. Ensure price_features.csv has engineered features.")

    # Optional z-score per date
    if args.zscore_per_date:
        df[feats] = cross_sectional_zscore(df[["Date"] + feats], feats)[feats]

    # Optional sector de-mean of target
    if args.demean_by_sector and "SectorName" in df.columns:
        df = de_mean_target_by_sector(df)

    # Prepare arrays
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    y = df["Target"].values
    X = df[feats].values

    # TSSplit
    tss = TimeSeriesSplit(n_splits=args.n_splits, gap=args.gap)

    fold_metrics = []
    oof_list = []
    feat_importance = pd.DataFrame(index=feats)

    params = {
        "n_estimators": 500,
        "num_leaves": 100,
        "learning_rate": 0.1,
        "colsample_bytree": 0.9,
        "subsample": 0.8,
        "reg_alpha": 0.4,
        "random_state": 21,
        "n_jobs": 4,
    }

    log_lines = []
    for fold, (tr_idx, va_idx) in enumerate(tss.split(X, y), 1):
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val, y_val = X[va_idx], y[va_idx]

        d_train = df.iloc[tr_idx]
        d_val = df.iloc[va_idx]

        # Train
        gbm = LGBMRegressor(**params)
        gbm.fit(X_train, y_train)

        # Predict and compute RMSE/MAE
        pred_val = gbm.predict(X_val)
        rmse = float(mean_squared_error(y_val, pred_val) ** 0.5)
        mae = float(mean_absolute_error(y_val, pred_val))

        # Rank per day on validation slice
        ranks = []
        dates_in_val = d_val["Date"].unique()
        for dt in dates_in_val:
            mask = d_val["Date"] == dt
            preds_day = pd.Series(pred_val[mask], index=d_val.index[mask])
            rank_day = preds_day.rank(method="first", ascending=False).astype(int) - 1
            ranks.append(rank_day)
        ranks = pd.concat(ranks).sort_index()

        fold_df = pd.DataFrame({
            "Date": d_val["Date"],
            "SecuritiesCode": d_val["SecuritiesCode"],
            "Target": y_val,
            "Pred": pred_val,
            "Rank": ranks.values
        }).sort_values(["Date", "SecuritiesCode"])

        sharpe = calc_spread_return_sharpe(fold_df, args.portfolio_size, args.toprank_weight_ratio)

        fold_metrics.append({"fold": fold, "rmse": rmse, "mae": mae, "sharpe": sharpe})
        oof_list.append(fold_df)

        # Feature importances
        feat_importance[f"Fold{fold}"] = gbm.feature_importances_

        log_lines.append(
            f"Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, Sharpe={sharpe:.4f}, "
            f"Train dates {d_train['Date'].min().date()}–{d_train['Date'].max().date()}, "
            f"Valid dates {d_val['Date'].min().date()}–{d_val['Date'].max().date()}"
        )

        del X_train, y_train, X_val, y_val, d_train, d_val, gbm, pred_val
        gc.collect()

    # Save logs and metrics
    (outdir / "cv_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    metrics = {
        "folds": fold_metrics,
        "mean": {
            "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "sharpe": float(np.mean([m["sharpe"] for m in fold_metrics])),
        },
        "std": {
            "rmse": float(np.std([m["rmse"] for m in fold_metrics], ddof=0)),
            "mae": float(np.std([m["mae"] for m in fold_metrics], ddof=0)),
            "sharpe": float(np.std([m["sharpe"] for m in fold_metrics], ddof=0)),
        },
    }
    (outdir / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save OOF predictions
    oof_df = pd.concat(oof_list, ignore_index=True)
    oof_df.to_csv(outdir / "oof_preds.csv", index=False)

    # Save feature importances
    feat_importance.to_csv(outdir / "feature_importance.csv")
    plot_feature_importance(feat_importance, outdir / "feature_importance.png")

    print("Done.")
    print(f"- CV metrics:      {outdir / 'cv_metrics.json'}")
    print(f"- OOF preds:       {outdir / 'oof_preds.csv'}")
    print(f"- Feature imp png: {outdir / 'feature_importance.png'}")
    print(f"- CV log:          {outdir / 'cv_log.txt'}")


if __name__ == "__main__":
    main()
