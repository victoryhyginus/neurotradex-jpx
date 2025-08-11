import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import os
import joblib

# === Config ===
DATA_PATH = '../data/processed/stock_prices_features.csv'
MODEL_PATH = '../models/random_forest_model.pkl'
CONF_MATRIX_IMG = 'confusion_matrix_rf.png'

# === Load Data ===
print(f"✅ Loaded dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
print(f"✅ Dataset shape: {df.shape}")

# === Fix 1: Proper Signal Labeling by Date ===
df['Signal'] = df.groupby('Date')['Target'].transform(lambda x: (x.rank(pct=True) > 0.9).astype(int))

# === Features & Target ===
feature_cols = [
    'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume',
    'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag',
    'Return', 'MA_5', 'MA_10', 'Volatility_5', 'Volatility_10',
    'Close/Open', 'High/Low', 'Lag_1', 'Lag_3', 'Lag_5'
]

X = df[feature_cols]
y = df['Signal']

print(f"📊 Class balance:\n{y.value_counts().rename('count')}")

# Filter to recent data only
df = df[df['Date'] > '2020-12-23']

# Optional: drop any rows with missing Target
df = df[df['Target'].notna()]

print(f"📆 Filtered to post-2020-12-23 data. New shape: {df.shape}")
print(f"❓ Missing values in Target: {df['Target'].isna().sum()}")


# === Model Training ===
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"📈 Cross-validated Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Fit model
model.fit(X, y)

# Predict & Report
y_pred = model.predict(X)
print("\n📋 Classification Report:")
print(classification_report(y, y_pred))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(CONF_MATRIX_IMG)
print(f"✅ Confusion matrix saved as '{CONF_MATRIX_IMG}'")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Trained model saved to '{MODEL_PATH}'")