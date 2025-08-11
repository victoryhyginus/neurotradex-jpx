
# 📊 NeuroTradeX — EDA Visual Insight Report

This report summarizes key visualizations from the JPX Tokyo Stock Exchange dataset and interprets potential trading insights.

---

## 1️⃣ Volume Over Time – Top 5 Most Active Stocks

![Volume Over Time](images/vol-over-time-top5.png)

**Insight**:  
Stock `7272` dominates in trading volume, far surpassing the others. It may be a highly liquid or actively traded security, suitable for high-frequency or momentum strategies.

---

## 2️⃣ Supervision Flag Distribution

![Supervision Flag](images/sup-flag-distribution.png)

**Insight**:  
Most stocks are not under supervision. This means the dataset mostly includes standard, unrestricted stocks. No immediate regulatory filtering is necessary.

---

## 3️⃣ Price & Volatility — Stock 1301

![Price and Volatility](images/price-volatility-stock1301.png)

**Insight**:  
This stock shows a smooth upward price trend with minimal volatility. This could signal strong momentum or stable investor interest.

---

## 4️⃣ Heatmap of Daily Returns – Top 10 Traded Stocks

![Heatmap Daily Returns](images/heatmap-daily-returns.png)

**Insight**:  
Several securities show significantly higher daily returns. This can be used to spot short-term trading opportunities or inform a volatility-based strategy.

---

## 5️⃣ Close Price Over Time — Stock 1301

![Close Price](images/close-price-over-time-stock1301.png)

**Insight**:  
Visualizes a linear uptrend in close price — helpful for confirming moving average strategy signals or bullish momentum.

---

## ✅ Conclusion

These plots provide solid exploratory insights. Next steps may include:

- Feature engineering (volume, volatility, MA signals)
- Signal-based model or rule-based backtest
- API enrichment (dividends, earnings, news)

You’re ready to move toward modeling or signal generation!
