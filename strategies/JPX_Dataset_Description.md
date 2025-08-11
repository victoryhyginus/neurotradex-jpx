
# 📦 Dataset Description — JPX Tokyo Stock Exchange Prediction

This dataset contains historic data for a variety of Japanese stocks and options. Your challenge is to **predict the future returns of the stocks**.

As historic stock prices are not confidential, this is a **forecasting competition** using the **time series API**. The data for the public leaderboard period is included as part of the competition dataset. Expect to see many people submitting perfect submissions for fun. Accordingly:

- The **public leaderboard** is mainly for testing and convenience.
- The **real evaluation** happens in the **forecasting phase** using future real market data.

---

## 📁 Files Overview

- **`stock_prices.csv`**  
  The core file of interest. Includes the daily closing price for each stock and the **target column** to be predicted.

- **`options.csv`**  
  Data on various stock market options. These may implicitly reflect future market predictions, although they’re **not directly scored**.

- **`secondary_stock_prices.csv`**  
  Contains data for **less-liquid securities** not in the core dataset. These aren't scored but may be useful for **market context**.

- **`trades.csv`**  
  Summary of **weekly trading volumes**, useful for understanding investor activity.

- **`financials.csv`**  
  Results from **quarterly earnings reports** — valuable for fundamental analysis.

- **`stock_list.csv`**  
  Maps `SecuritiesCode` to **company names** and includes **industry classifications**.

---

## 📂 Folders Overview

- **`data_specifications/`**  
  Contains definitions and documentation for each column.

- **`jpx_tokyo_market_prediction/`**  
  Contains files needed to run the **time series API**. The API is efficient: it delivers all rows in under 5 minutes and uses < 0.5 GB of memory.

- **`train_files/`**  
  Main training data covering historical periods.

- **`supplemental_files/`**  
  Contains dynamic training data updates — added during the competition (e.g., May, June) and once at the start of the **forecasting phase**.

- **`example_test_files/`**  
  Mimics the **public test phase**. Contains:
  - All columns (except `Target`)
  - Sample submission format
  - Lets you simulate API submissions offline  
  You can derive `Target` from the `Close` price by calculating returns across two days.

---

## 📝 Notes
- The dataset is rich and complex, suitable for both **technical indicators** and **fundamental analysis**.
- Use the provided data thoughtfully to create robust and **time-aware features**.

