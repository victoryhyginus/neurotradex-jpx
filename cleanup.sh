#!/bin/bash
set -e  # stop if any command fails

echo "📂 Moving README_SUBMISSION.txt to submissions/"
mkdir -p submissions
if [ -f scripts/README_SUBMISSION.txt ]; then
    mv scripts/README_SUBMISSION.txt submissions/README_submission.md
fi

echo "📂 Moving HTML and MD reports from scripts to reports/"
mkdir -p reports
mv scripts/*.html reports/ 2>/dev/null || true
mv scripts/*.md reports/ 2>/dev/null || true

echo "🧹 Removing large CSVs from submissions/"
find submissions -type f -name "*.csv" ! -name "sample_submission.csv" -delete

echo "🗂 Creating tiny Kaggle-compliant sample submission"
if [ ! -f submissions/sample_submission.csv ]; then
    cat > submissions/sample_submission.csv <<'EOF'
Date,SecuritiesCode,Rank
2021-12-06,1301,0
2021-12-06,1332,1
2021-12-06,1333,2
EOF
fi

if [ ! -f submissions/README_submission.md ]; then
    cat > submissions/README_submission.md <<'EOF'
# Submissions

This folder stores *examples only*. Do **not** commit real competition predictions.

- `sample_submission.csv` is a tiny demo file.
- Generate real submissions locally with:

```bash
python scripts/jpx_submission.py

