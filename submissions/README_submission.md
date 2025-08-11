JPX Kaggle Submission – Step-by-step (v2 pipeline)
====================================================

Files:
- train_kaggle_model.py  — trains LightGBM using our v2 feature engineering; saves:
    - lgbm_jpx_model.pkl
    - feature_list.json
- jpx_submission.py      — inference-only script for Kaggle that loads the model and runs the API loop.
- This README

A) Train your model
-------------------
Option 1: Train on Kaggle
- Create a new Notebook under the JPX competition.
- Attach the competition dataset (auto-attached in competition notebooks).
- Upload train_kaggle_model.py (Add data → Upload).
- Run, for example:
  !python /kaggle/input/<your-upload-folder>/train_kaggle_model.py \
    --data-dir /kaggle/input/jpx-tokyo-stock-exchange-prediction \
    --start-date 2019-01-01 \
    --horizon 1 \
    --out-dir /kaggle/working

- After training finishes, artifacts will be in /kaggle/working:
    lgbm_jpx_model.pkl, feature_list.json, feature_importance_gain.csv

- Click "Save Version" → "Save & Create Dataset" to publish these files as a Kaggle Dataset.
  Example dataset slug: "my-jpx-model"

Option 2: Train locally
- Download the Kaggle dataset locally with the same folder layout.
- Run:
  python train_kaggle_model.py --data-dir /path/to/jpx-dataset --start-date 2019-01-01 --horizon 1 --out-dir ./model_artifacts
- Upload lgbm_jpx_model.pkl and feature_list.json to Kaggle as a **new Dataset** (e.g., "my-jpx-model").

B) Submit on Kaggle
-------------------
- Open a new Notebook under the competition.
- Attach:
  1) Competition dataset: "jpx-tokyo-stock-exchange-prediction"
  2) Your model dataset: the one you just published (e.g., "my-jpx-model")
  3) Upload jpx_submission.py (or paste its content into a cell)
- Edit jpx_submission.py to set:
    MODEL_DIR = "/kaggle/input/my-jpx-model"
- Run:
  !python /kaggle/input/<your-upload-folder>/jpx_submission.py
- If it finishes without errors, click "Submit to Competition".

Tips
----
- Keep feature engineering identical between train and submission.
- Do not change FEATS order; we save it as feature_list.json and reindex at inference.
- Reduce runtime by raising --start-date (e.g., 2020-01-01) if needed.
- The Kaggle API provides only the required inputs; you only output Ranks via env.predict().
