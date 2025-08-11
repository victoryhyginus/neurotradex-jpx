#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPX Phase 3: Modeling & Simple Backtest (Corrected)

- Loads Phase 2 features
- Builds H-day future return target
- Date-based split (train/val/test)
- Baselines: LogisticRegression (classification), Ridge (regression), optional XGBoost
- Simple top-K long backtest
- Robust NaN/Inf handling
"""

import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import matplotlib.pyplot as plt
plt.switch_backend("Agg")


def build_target(df: pd.DataFrame, col="AdjustedClose", horizon=5) -> pd.DataFrame:
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()
    # future return over horizon
    df["target_return"] = df.groupby("SecuritiesCode")[col].shift(-horizon) / df[col] - 1.0
    # sanitize any infs from division by zero BEFORE creating cls
    df["target_return"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["target_cls"] = (df["target_return"] > 0).astype("float")
    return df


def make_splits(df: pd.DataFrame, train_end: str, val_end: str, test_end: str):
    tr = df[df["Date"] <= pd.to_datetime(train_end)].copy()
    va = df[(df["Date"] > pd.to_datetime(train_end)) & (df["Date"] <= pd.to_datetime(val_end))].copy()
    te = df[(df["Date"] > pd.to_datetime(val_end)) & (df["Date"] <= pd.to_datetime(test_end))].copy()
    return tr, va, te


def numeric_feature_cols(df: pd.DataFrame):
    tech_prefixes = ("Return_", "MovingAvg_", "ExpMovingAvg_", "Volatility_")
    cols = [c for c in df.columns if c.startswith(tech_prefixes)]
    cols = ["AdjustedClose"] + cols
    # keep only numeric
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) ]


def drop_bad_targets(df: pd.DataFrame):
    return df[df["target_return"].notna()].copy()


def fit_classification(train, val, feature_cols):
    Xtr, ytr = train[feature_cols].values, train["target_cls"].values.astype(int)
    Xva, yva = val[feature_cols].values, val["target_cls"].values.astype(int)

    logit = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
    logit.fit(Xtr, ytr)

    yhat_va = logit.predict(Xva)
    try:
        yproba_va = logit.predict_proba(Xva)[:, 1]
    except Exception:
        s = logit.decision_function(Xva)
        yproba_va = (s - s.min()) / (s.max() - s.min() + 1e-9)

    metrics = dict(
        acc=float(accuracy_score(yva, yhat_va)),
        f1=float(f1_score(yva, yhat_va)),
        precision=float(precision_score(yva, yhat_va)),
        recall=float(recall_score(yva, yhat_va)),
        roc_auc=float(roc_auc_score(yva, yproba_va)),
    )
    models = {"logit": logit}

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", n_jobs=4, random_state=42
        )
        xgb.fit(Xtr, ytr)
        yhat_xgb = xgb.predict(Xva)
        yproba_xgb = xgb.predict_proba(Xva)[:, 1]
        metrics.update({
            "xgb_acc": float(accuracy_score(yva, yhat_xgb)),
            "xgb_f1": float(f1_score(yva, yhat_xgb)),
            "xgb_roc_auc": float(roc_auc_score(yva, yproba_xgb)),
        })
        models["xgb"] = xgb

    return models, metrics


def fit_regression(train, val, feature_cols):
    Xtr, ytr = train[feature_cols].values, train["target_return"].values
    Xva, yva = val[feature_cols].values, val["target_return"].values

    ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=42))])
    ridge.fit(Xtr, ytr)
    ypred_va = ridge.predict(Xva)

    metrics = dict(
        r2=float(r2_score(yva, ypred_va)),
        mae=float(mean_absolute_error(yva, ypred_va)),
        rmse=float((mean_squared_error(yva, ypred_va)) ** 0.5),
    )
    models = {"ridge": ridge}
    return models, metrics


def run_backtest(df_test: pd.DataFrame, scores: pd.Series, topk=20):
    scored = df_test[["Date", "SecuritiesCode", "target_return"]].copy()
    scored["score"] = scores.values
    picks = (
        scored.groupby("Date", as_index=False)
              .apply(lambda g: g.nlargest(topk, "score"))
              .reset_index(drop=True)
    )
    daily = picks.groupby("Date")["target_return"].mean().sort_index()
    cumret = (1.0 + daily).cumprod()
    return daily, cumret


def save_backtest_plot(cumret: pd.Series, out_png: Path, title="Cumulative Return (Top-K Long)"):
    plt.figure(figsize=(10,5))
    plt.plot(cumret.index, cumret.values)
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(description="JPX Phase 3 Modeling + Backtest")
    ap.add_argument("--features-csv", type=str, default="./artifacts/price_features.csv")
    ap.add_argument("--out-dir", type=str, default="./artifacts_modeling")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-end", type=str, default="2021-12-31")
    ap.add_argument("--val-end", type=str, default="2022-03-31")
    ap.add_argument("--test-end", type=str, default="2022-06-24")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv, parse_dates=["Date"])
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()

    # Basic sanitation of features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["AdjustedClose"] = df.groupby("SecuritiesCode")["AdjustedClose"].ffill()
    # Zero prices can cause inf in returns; nudge zeros
    df["AdjustedClose"].replace(0.0, np.nan, inplace=True)
    df["AdjustedClose"].ffill(inplace=True)
    df.fillna(0.0, inplace=True)

    # Build target AFTER sanitation
    df = build_target(df, col="AdjustedClose", horizon=args.horizon)

    # Date splits
    tr, va, te = make_splits(df, args.train_end, args.val_end, args.test_end)
    # Drop rows where target is missing (end of each ticker series, etc.)
    tr, va, te = drop_bad_targets(tr), drop_bad_targets(va), drop_bad_targets(te)

    feature_cols = numeric_feature_cols(df)

    # Train
    cls_models, cls_metrics = fit_classification(tr, va, feature_cols)
    reg_models, reg_metrics = fit_regression(tr, va, feature_cols)

    # Test predictions
    Xte = te[feature_cols].values
    yproba_te = cls_models["logit"].predict_proba(Xte)[:, 1]
    ypred_te_reg = reg_models["ridge"].predict(Xte)

    te_out = te[["Date", "SecuritiesCode", "target_return"]].copy()
    te_out["proba_logit"] = yproba_te
    if "xgb" in cls_models:
        te_out["proba_xgb"] = cls_models["xgb"].predict_proba(Xte)[:, 1]
    te_out["pred_ridge"] = ypred_te_reg
    te_out.to_csv(outdir / "preds_test.csv", index=False)

    # Backtests
    daily_logit, cum_logit = run_backtest(te, pd.Series(yproba_te, index=te.index), topk=args.topk)
    save_backtest_plot(cum_logit, outdir / "backtest_logit.png", title=f"Cumulative Return (Top-{args.topk} Long, Logit)")

    daily_ridge, cum_ridge = run_backtest(te, pd.Series(ypred_te_reg, index=te.index), topk=args.topk)
    save_backtest_plot(cum_ridge, outdir / "backtest_ridge.png", title=f"Cumulative Return (Top-{args.topk} Long, Ridge)")

    # Save metrics
    out_metrics = {
        "classification_val": cls_metrics,
        "regression_val": reg_metrics,
        "backtest": {
            "logit": {"days": int(len(daily_logit)), "final_cumret": float(cum_logit.iloc[-1]) if len(cum_logit)>0 else None},
            "ridge": {"days": int(len(daily_ridge)), "final_cumret": float(cum_ridge.iloc[-1]) if len(cum_ridge)>0 else None},
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(out_metrics, indent=2))

    print("Done.")
    print(f"- Features:         {args.features_csv}")
    print(f"- Predictions:      {outdir / 'preds_test.csv'}")
    print(f"- Metrics:          {outdir / 'metrics.json'}")
    print(f"- Backtest (logit): {outdir / 'backtest_logit.png'}")
    print(f"- Backtest (ridge): {outdir / 'backtest_ridge.png'}")


if __name__ == "__main__":
    main()
