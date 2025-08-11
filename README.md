# NeuroTradeX JPX

Clean, Kaggle-compliant repository for the JPX Tokyo Stock Exchange Prediction project.

## Structure
- src/ – Core Python modules
- scripts/ – Entry-point scripts (train / inference / submission)
- notebooks/ – Jupyter notebooks for EDA & experiments
- reports/ – Analysis results, charts, and metrics
- images/ – Visuals for README and reports
- models/ – Model artifacts (empty; not versioned)
- data/ – Placeholder for Kaggle data (not committed)
- submissions/ – Saved competition submissions (small CSVs only)

## Setup
1) Create venv:  python -m venv .venv
2) Activate:     source .venv/bin/activate   (Windows: .venv\Scripts\activate)
3) Install:      pip install -r requirements.txt

## Data (not included)
Place Kaggle files locally like:
data/jpx-tokyo-stock-exchange-prediction/
  - train_files/stock_prices.csv
  - supplemental_files/stock_prices.csv
  - example_test_files/stock_prices.csv
  - example_test_files/sample_submission.csv

## Quick start
- Train:  python scripts/train_kaggle_model.py --base data/jpx-tokyo-stock-exchange-prediction
- Submit: python scripts/jpx_submission.py --base data/jpx-tokyo-stock-exchange-prediction --out submissions/submission.csv

## License
See LICENSE.
