# 🌡️ Temperature Prediction using Random Forest Surrogate Model

This project uses a **Random Forest Regressor** to predict daily temperature using **lag-based features** (time series forecasting). The goal is to train a surrogate model that captures temporal patterns in temperature data.

---

## 📁 Dataset

- The data is loaded from `dailytemp.csv`.
- It contains at least one column:
  - `temp`: the daily temperature readings.

---

## 🧠 Model Description

- Model: `RandomForestRegressor` from `scikit-learn`
- Features: Lagged temperature values (`lag_1`, `lag_2`, `lag_3`)
- Target: Current temperature

---

## 🛠️ Steps Performed

1. Read temperature data using `pandas`
2. Create **lagged features** using past 3 days' temperatures
3. Split data into 80% training and 20% testing
4. Train `RandomForestRegressor`
5. Predict temperature on the test set
6. Plot and compare **actual vs predicted** temperatures
7. Compute **R² score** (explained variance)

---

## 📊 Result

- The model shows good alignment between predicted and actual values.
- Example output R² score:
  
R² Score (Accuracy): 90.12%
