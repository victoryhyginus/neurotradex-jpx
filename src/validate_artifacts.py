import pandas as pd, numpy as np, pathlib as pl

data = pl.Path("./artifacts")

combo = pd.read_csv(data/"combined_prices.csv", parse_dates=["Date"])
adj   = pd.read_csv(data/"prices_adjusted.csv", parse_dates=["Date"])
feat  = pd.read_csv(data/"price_features.csv", parse_dates=["Date"])

def ok(name, cond): print(f"[{name}] {'OK' if cond else 'FAIL'}")

# 1) combined_prices.csv checks
ok("combined has rows", len(combo)>0)
ok("combined has required cols", {"Date","SecuritiesCode","Close","AdjustmentFactor"} <= set(combo.columns))
print("combined date range:", combo["Date"].min(), "→", combo["Date"].max())
print("combined tickers:", combo["SecuritiesCode"].nunique())

# 2) prices_adjusted.csv checks
ok("adjusted has AdjustedClose", "AdjustedClose" in adj.columns)
# Should be forward-filled; allow NaN only at series starts
null_ac = adj["AdjustedClose"].isna().sum()
print("AdjustedClose NaNs:", null_ac)

# 3) price_features.csv checks
periods = [5,10,20,30,50]
req_cols = {"Date","SecuritiesCode","AdjustedClose"} | \
           {f"{k}_{p}Day" for p in periods for k in ["Return","MovingAvg","ExpMovingAvg","Volatility"]}
ok("features has required cols", req_cols <= set(feat.columns))
print("features rows:", len(feat), "tickers:", feat["SecuritiesCode"].nunique())

# Spot-check: no all-null feature columns
allnull = [c for c in req_cols if c in feat.columns and feat[c].isna().all()]
print("all-null feature cols:", allnull or "None")

# Optional: check sector merge happened (if present)
sect_ok = {"Name","SectorName"} <= set(feat.columns)
ok("features has sector info", sect_ok)
if sect_ok:
    print("unique sectors:", feat["SectorName"].nunique())
