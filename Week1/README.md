# Peak-Hour-Electricity-Spikes
Analyzed hourly electricity consumption and predicted weekly evening peak loads using linear regression.

---

## 🔹 Project Objectives

- Analyze hourly electricity consumption.
- Apply smoothing and trend analysis to highlight daily and weekly patterns.
- Aggregate evening consumption (17:00–21:00) into weekly peaks.
- Predict weekly evening peak loads using linear regression.

---

## 🔹 Dataset & Preprocessing

**Dataset:** Hourly electricity consumption and generation data  
**Link:** [Kaggle](https://www.kaggle.com/datasets/stefancomanita/hourly-electricity-consumption-and-production)  

**Steps:**

1. Imported CSV data and converted `DateTime` to datetime type.
2. Sorted data chronologically.
3. Extracted time-based features: `hour`, `weekday`, `week`, `year`.
4. Checked for missing values and explored basic statistics.
5. Applied Exponential Moving Average (EMA) over 24 hours to smooth short-term fluctuations.
6. Filtered evening hours (17:00–21:00) to focus on peak periods.
7. Aggregated weekly evening consumption and weekly average generation from various energy sources (Nuclear, Wind, Solar, etc.).

---

## 🔹 Feature Engineering

Constructed weekly feature set including:

- Average weekly consumption
- Mean weekly generation from different energy sources  

**Target variable:** Weekly evening peak consumption

---

## 🔹 Modeling

- Linear Regression used to predict weekly evening peaks based on weekly features.
- Predicted peaks stored in `predicted_peak` column.

**Evaluation Metrics:**

| Metric | Value |
|--------|-------|
| MAE    | 147 MW |
| RMSE   | ~228 MW (~29% of natural variability) |
| R²     | 0.915 |

> Interpretation: MAE is very low relative to mean peak (~7920 MW), and R² shows that most variation is explained by the model.

---

## 🔹 Smoothing & Trend Analysis

- EMA-smoothed hourly consumption reveals clear daily and weekly patterns.
- Weekly aggregation highlights evening peak trends.
- Linear regression captures trends based on consumption and generation features.

---

## 🔹 Visualization

**Hourly Consumption Plot:**

- Raw hourly consumption (opacity 0.3)
- EMA-smoothed 24-hour consumption line

**Weekly Peak Prediction Plot:**

- Actual evening peaks (markers)
- Predicted evening peaks (line)

---

## 🔹 Comparison with Other Forecasting Methods

| Method | Pros | Cons |
|--------|------|------|
| Traditional Load Forecasting | Simple, historical patterns | Limited adaptability, manual tuning |
| Rule-Based Peak Estimation | Interpretable, easy | Cannot adapt to evolving patterns, ignores renewables |
| ARIMA/SARIMA | Captures seasonality | Requires stationarity, less flexible with exogenous features |
| Machine Learning (LSTM, XGBoost) | High accuracy for nonlinear patterns | High computational cost, less interpretable |
| **This Project** | Transparent, low-cost, near real-time | Slightly lower predictive power compared to complex ML models |

> This system emphasizes interpretability, transparency, and real-time usability while integrating renewable generation features.

---

## 🔹 Key Insights

- Smoothed consumption highlights daily and weekly behavior patterns.
- Linear regression predicts weekly evening peaks accurately.
- Model performance is excellent: RMSE < 30% of standard deviation, R² = 0.915.
- Provides actionable insights for energy management at campus/dorm level.

---
