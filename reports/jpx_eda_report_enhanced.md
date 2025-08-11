
# 📊 JPX Tokyo Stock Exchange — Exploratory Data Analysis Report

This report presents key insights from the JPX dataset provided in the Kaggle competition. The goal is to understand market dynamics, stock behaviors, and uncover patterns that may inform predictive modeling.

---

## 1️⃣ Dataset Overview

- Total Records: **2,332,531**
- Columns: 12
- Unique Securities: **2,000**
- Date Range: **2017-01-04** to **2021-12-03**

### 🔍 Columns
- `Date`, `SecuritiesCode`, `Open`, `High`, `Low`, `Close`, `Volume`
- `AdjustmentFactor`, `ExpectedDividend`, `SupervisionFlag`, `Target`

---

## 2️⃣ Missing Values Summary

| Column            | Missing Count |
|-------------------|---------------|
| Open / High / Low / Close | 7,608 |
| ExpectedDividend  | 2,313,666 |
| Target            | 238 |

*Most stocks have no expected dividend values, which limits some dividend-based modeling.*

---

## 3️⃣ Market-Wide Metrics Over Time

📈 Interactive chart showing **average return**, **closing price**, and **volume**:

![Market Overview](jpx_eda_chart.html)

---

## 4️⃣ Top 5 Most Active Stocks (by Volume)

Top 5 stocks: `8411`, `8306`, `6502`, `4689`, `8604`

### 🔄 Volume Over Time
![Volume](top5_volume_over_time.html)

### 💵 Closing Price Over Time
![Price](top5_close_price_over_time.html)

### 📊 Daily Returns
![Returns](top5_daily_return.html)

### 📉 Rolling 30-Day Volatility
![Volatility](top5_rolling_volatility.html)

---

## ✅ Key Observations

- Stocks like `8411` and `8306` dominate in trading volume — ideal for liquidity-driven strategies.
- Volatility varies significantly across these top stocks, revealing potential for risk-adjusted return models.
- Missing values are manageable except for `ExpectedDividend`.

---

## 🧭 Next Steps

- Labeling and generating trading signals
- Feature engineering (momentum, volatility, regime signals)
- Model selection and backtesting

---

📂 **Generated Automatically — Ready for GitHub/Docs**


---

## 📈 Interactive Visualizations

### JPX Market Overview Chart
<iframe src="jpx_eda_chart.html" width="100%" height="500px" frameBorder="0"></iframe>

### Top 5 Stocks by Volume Over Time
<iframe src="top5_volume_over_time.html" width="100%" height="500px" frameBorder="0"></iframe>

### Top 5 Stocks: Close Price Over Time
<iframe src="top5_close_price_over_time.html" width="100%" height="500px" frameBorder="0"></iframe>

### Top 5 Stocks: Daily Returns
<iframe src="top5_daily_return.html" width="100%" height="500px" frameBorder="0"></iframe>

### Top 5 Stocks: Rolling Volatility
<iframe src="top5_rolling_volatility.html" width="100%" height="500px" frameBorder="0"></iframe>
