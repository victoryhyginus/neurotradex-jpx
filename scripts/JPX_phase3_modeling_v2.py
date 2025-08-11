#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPX Phase 3 (v2): LightGBM + TimeSeriesSplit + JPX Spread Sharpe
- Preserves feature names (DataFrame, not ndarray)
- Uses LightGBM booster gain importances (averaged across folds)
- Exports PNG + HTML + CSV
"""

import argparse, json, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor, log_evaluation
import matplotlib.pyplot as plt
plt.switch_backend("Agg")


def build_target(df: pd.DataFrame, col="AdjustedClose", horizon=1) -> pd.DataFrame:
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()
    df["Target"] = df.groupby("SecuritiesCode")[col].shift(-horizon) / df[col] - 1.0
    df["Target"].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def cross_sectional_zscore(df: pd.DataFrame, feature_cols):
    z = df.copy()
    for c in feature_cols:
        g = z.groupby("Date")[c]
        z[c] = (g.transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))).astype(float)
    return z


def de_mean_target_by_sector(df: pd.DataFrame):
    if "SectorName" not in df.columns:
        return df
    out = df.copy()
    grp = out.groupby(["Date", "SectorName"])["Target"]
    out["Target"] = out["Target"] - grp.transform("mean")
    return out


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    def _calc_spread_return_per_day(day_df, portfolio_size, toprank_weight_ratio):
        n = len(day_df)
        if n == 0:
            return 0.0
        k = int(min(portfolio_size, n))
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
    for c in ["Open_lag1", "High_lag1", "Low_lag1"]:
        if c in df.columns:
            cols.append(c)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def maybe_add_lagged_ohlc(df: pd.DataFrame):
    out = df.sort_values(["SecuritiesCode", "Date"]).copy()
    for c in ["Open", "High", "Low"]:
        if c in out.columns:
            out[f"{c}_lag1"] = out.groupby("SecuritiesCode")[c].shift(1)
    return out


def plot_feature_importance(fi_df: pd.DataFrame, out_png: Path, topn=30):
    avg = fi_df.mean(axis=1).sort_values(ascending=False).head(topn)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm, colors

    plt.figure(figsize=(8, 10))
    vals = avg.values[:]
    norm = colors.Normalize(vmin=vals.min(), vmax=vals.max())
    cmap = cm.get_cmap("plasma")
    for i, (v, name) in enumerate(zip(vals[::-1], avg.index[::-1])):
        plt.barh(i, v, color=cmap(norm(v)))
    plt.yticks(range(len(avg)), avg.index[::-1])
    plt.xlabel("Average Importance (gain)")
    plt.title("Overall Feature Importance")
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

    df = pd.read_csv(args.features_csv, parse_dates=["Date"])
    df = maybe_add_lagged_ohlc(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()

    df = build_target(df, col="AdjustedClose", horizon=args.horizon)
    df = df[df["Target"].notna()].copy()

    feats = feature_columns(df)
    if len(feats) == 0:
        raise RuntimeError("No feature columns found.")

    if args.zscore_per_date:
        df[feats] = cross_sectional_zscore(df[["Date"] + feats], feats)[feats]

    if args.demean_by_sector and "SectorName" in df.columns:
        df = de_mean_target_by_sector(df)

    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    y = df["Target"].values
    X = df[feats].copy()  # preserve names

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
        X_train, y_train = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[va_idx], y[va_idx]
        d_train = df.iloc[tr_idx]
        d_val = df.iloc[va_idx]

        gbm = LGBMRegressor(**params)
        # Remove 'verbose' (not supported in some versions); use callback to silence logs
        gbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[log_evaluation(period=0)],
            feature_name=list(X_train.columns)
        )

        pred_val = gbm.predict(X_val)
        rmse = float(mean_squared_error(y_val, pred_val) ** 0.5)
        mae = float(mean_absolute_error(y_val, pred_val))

        ranks = []
        for dt in d_val["Date"].unique():
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

        names = list(gbm.booster_.feature_name())
        gains = gbm.booster_.feature_importance(importance_type="gain")
        _s = pd.Series(gains, index=names)
        feat_importance[f"Fold{fold}"] = _s.reindex(feats).fillna(0).values

        log_lines.append(
            f"Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, Sharpe={sharpe:.4f}, "
            f"Train {d_train['Date'].min().date()}–{d_train['Date'].max().date()}, "
            f"Valid {d_val['Date'].min().date()}–{d_val['Date'].max().date()}"
        )

        del X_train, y_train, X_val, y_val, d_train, d_val, gbm, pred_val
        gc.collect()

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

    oof_df = pd.concat(oof_list, ignore_index=True)
    oof_df.to_csv(outdir / "oof_preds.csv", index=False)

    feat_importance.to_csv(outdir / "feature_importance.csv")

    avg_gain = feat_importance.mean(axis=1).sort_values(ascending=False)
    avg_gain.to_csv(outdir / "feature_importance_gain.csv", header=["avg_gain"])

    plot_feature_importance(feat_importance, outdir / "feature_importance.png")

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        vals = (avg_gain - avg_gain.min()) / (avg_gain.max() - avg_gain.min() + 1e-12)
        colors = px.colors.sequential.Plasma
        idx = (vals * (len(colors)-1)).round().astype(int).clip(0, len(colors)-1)
        bar_colors = [colors[i] for i in idx.values]

        fig = go.Figure(go.Bar(
            x=avg_gain.values[::-1],
            y=avg_gain.index[::-1],
            orientation="h",
            marker=dict(color=bar_colors[::-1]),
            hovertemplate="%{y}<br>Average Importance (gain) = %{x:.2f}<extra></extra>"
        ))
        fig.update_layout(
            title="Overall Feature Importance",
            xaxis_title="Average Importance (gain)",
            yaxis_title=None,
            template="plotly_white",
            height=800,
            margin=dict(l=160, r=40, t=60, b=40)
        )
        fig.write_html(outdir / "feature_importance.html", include_plotlyjs="cdn")
    except Exception as e:
        print(f"[warn] Could not write feature_importance.html: {e}")

    print("Done.")
    print(f"- CV metrics:              {outdir / 'cv_metrics.json'}")
    print(f"- OOF preds:               {outdir / 'oof_preds.csv'}")
    print(f"- Feature imp matrix CSV:  {outdir / 'feature_importance.csv'}")
    print(f"- Feature imp gain CSV:    {outdir / 'feature_importance_gain.csv'}")
    print(f"- Feature imp PNG:         {outdir / 'feature_importance.png'}")
    print(f"- Feature imp HTML:        {outdir / 'feature_importance.html'}")
    print(f"- CV log:                  {outdir / 'cv_log.txt'}")


if __name__ == "__main__":
    main()
