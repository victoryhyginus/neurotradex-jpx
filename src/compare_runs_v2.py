import json, pathlib as pl
for name in ["artifacts_modeling_v2",
             "artifacts_modeling_v2_sectorDM",
             "artifacts_modeling_v2_h5"]:
    p = pl.Path(name) / "cv_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        print(f"{name}: Sharpe mean {m['mean']['sharpe']:.3f} ± {m['std']['sharpe']:.3f} | "
              f"RMSE {m['mean']['rmse']:.6f} | MAE {m['mean']['mae']:.6f}")
