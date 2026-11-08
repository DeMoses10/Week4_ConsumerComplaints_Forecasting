# Consumer Complaint Trends & Response Intelligence Platform: Model Comparison & Validation Report

**Document Title:** Comparative Model Development, Rolling-Origin Cross-Validation & Benchmark Report  
**Project:** Consumer Complaint Trends and Response Intelligence Platform  
**Track:** Machine Learning  
**Programme:** AnalystLab Africa Experience Lab (Week 4)  
**Author:** Machine Learning Engineering Intern  
**Date:** November 2026  

---

## 1. Executive Summary & Success Criteria

To establish an evidence-based forecasting prototype, eight forecasting architectures were benchmarked across two rigorous validation schemes: a fixed out-of-sample holdout test (January–July 2023) and multi-origin rolling-window cross-validation (17 historical origins, 6-month horizons).

### Pre-Defined Engineering Success Criteria:
1. **Outperform Operational Baselines:** The candidate model must improve upon the Seasonal Naive baseline by at least **20% in Mean Absolute Error (MAE)** and achieve a **Weighted Absolute Percentage Error (WAPE) $< 15\%$** on out-of-sample data.
2. **Horizon Stability:** The model must maintain consistent error rates across multi-step forecast horizons ($h = 1$ to $h = 6$ months) without exponential divergence.
3. **Statistical Soundness:** Residuals from the champion model must pass the Ljung-Box test for white noise ($p > 0.05$), confirming that systematic temporal dependencies have been properly captured.
4. **Product-Level Extensibility:** The modeling architecture must demonstrate capability to forecast disaggregated product-level complaint flows for specialized operational queues.

---

## 2. Models Evaluated & Architecture Specifications

### 2.1 Baseline Forecasters
- **Naive (Persistence):** Assumes complaint volume remains constant from the preceding month: $\hat{y}_{t+h} = y_t$. Serves as the fundamental lower bound.
- **Seasonal Naive:** Assumes complaint volume replicates the exact observation from the same month in the prior year: $\hat{y}_{t+h} = y_{t+h-12}$.
- **3-Month Simple Moving Average (SMA-3):** Predicts the arithmetic mean of the preceding 3 months: $\hat{y}_{t+h} = \frac{1}{3} \sum_{i=1}^3 y_{t-i+1}$.

### 2.2 Classical Statistical Time-Series Models
- **Holt-Winters Exponential Smoothing:** Multiplicative/additive formulation with additive trend and 12-month additive seasonal factors optimized via maximum likelihood.
- **SARIMAX $(1, 1, 1) \times (1, 1, 0)_{12}$:** Seasonal AutoRegressive Integrated Moving Average incorporating first-order non-seasonal differencing and annual seasonal autoregression.

### 2.3 Machine Learning Recursive Forecasters
- **Ridge Regressor (L2 Regularization):** Linear combination of 17 engineered lag, rolling, and cyclical features with $L_2$ shrinkage parameter $\alpha = 5.0$.
- **Random Forest Regressor:** Non-linear ensemble of 100 decision trees (`max_depth=6`, `random_state=42`) with recursive multi-step lag updates.
- **HistGradientBoostingRegressor:** Histogram-binned gradient boosted decision trees (`max_iter=100`, `max_depth=4`, `l2_regularization=2.0`).

---

## 3. Quantitative Benchmark: Out-of-Sample Holdout (2023 Test Period)

The following table summarizes model performance on the 7-month holdout test set (January 2023 to July 2023), evaluated against 8,464 actual incoming complaints:

| Rank | Model Architecture | Test MAE | Test RMSE | **Test WAPE (%)** | Test MAPE (%) | Test sMAPE (%) | Test $R^2$ | Architectural Classification |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Random Forest Regressor** | **145.17** | 241.68 | **12.08%** | **10.25%** | **11.22%** | -0.0741 | Machine Learning Ensemble *(Champion)* |
| **2** | **3-Month Moving Average** | **149.19** | 242.27 | **12.41%** | 10.62% | 11.58% | -0.0793 | Simple Rolling Baseline |
| **3** | **Naive (Persistence)** | **150.43** | 240.07 | **12.51%** | 10.81% | 11.69% | -0.0597 | Simple Persistence Baseline |
| **4** | **SARIMAX $(1,1,1)\times(1,1,0)_{12}$**| **156.14** | **220.98** | **12.99%** | 11.82% | 12.10% | **0.1020** | Classical Statistical Time Series |
| **5** | **Holt-Winters ExpSmoothing** | **163.10** | 232.99 | **13.57%** | 12.61% | 12.71% | 0.0018 | Classical Exponential Smoothing |
| **6** | **Ridge Regressor (L2)** | **183.45** | 243.03 | **15.26%** | 14.35% | 14.44% | -0.0861 | Regularized Linear ML |
| **7** | **Seasonal Naive (Lag-12)** | **197.29** | 290.61 | **16.41%** | 14.36% | 16.37% | -0.5530 | Seasonal Baseline |
| **8** | **HistGradientBoosting** | **202.74** | 296.84 | **16.87%** | 14.63% | 16.75% | -0.6203 | Gradient Boosted Trees |

---

## 4. Multi-Origin Rolling-Window Cross-Validation (6-Month Horizons)

To evaluate model resilience across changing historical periods, expanding-window rolling-origin cross-validation was conducted across **17 distinct forecast origins** (minimum train window: 36 months, horizon: 6 months, step size: 2 months):

| Rank | Model Architecture | CV MAE | CV RMSE | **CV WAPE (%)** | CV MAPE (%) | CV sMAPE (%) | CV $R^2$ | Number of Origins |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Naive (Persistence)** | **96.04** | **120.49** | **9.75%** | **9.53%** | **9.90%** | **0.2462** | 17 |
| **2** | **3-Month Moving Average** | **100.60** | **125.33** | **10.21%** | 10.07% | 10.63% | 0.1845 | 17 |
| **3** | **Random Forest Regressor** | **104.81** | **139.22** | **10.64%** | 10.60% | 11.44% | -0.0064 | 17 |
| **4** | **Holt-Winters ExpSmoothing** | **130.98** | 164.30 | **13.30%** | 13.44% | 13.87% | -0.4015 | 17 |
| **5** | **SARIMAX $(1,1,1)\times(1,1,0)_{12}$**| **136.53** | 170.57 | **13.86%** | 14.10% | 14.72% | -0.5105 | 17 |
| **6** | **Seasonal Naive (Lag-12)** | **167.61** | 208.14 | **17.01%** | 16.66% | 19.10% | -1.2493 | 17 |
| **7** | **HistGradientBoosting** | **191.37** | 217.06 | **19.43%** | 19.30% | 22.03% | -1.4461 | 17 |
| **8** | **Ridge Regressor (L2)** | **325.83** | 1164.39 | **33.07%** | 37.64% | 19.12% | -69.3920 | 17 |

---

## 5. Technical Insights & Comparative Interpretation

### 5.1 Why Random Forest Excelled
`Random Forest Regressor` achieved the lowest error on the out-of-sample 2023 holdout:
- **Test MAE of 145.17 complaints** and a **WAPE of 12.08%** (compared to 197.29 / 16.41% for Seasonal Naive, representing a **26.4% error reduction**).
- Unlike linear models that extrapolate trend indices linearly into infinity, decision tree ensembles perform piecewise-constant approximations. They bound their forecasts to reasonable historical extremes, preventing explosive runaway forecasts.
- The ensemble successfully integrated short-term momentum (`diff_lag1_lag2`), quarterly moving averages, and cyclical month encodings.

### 5.2 Strengths of SARIMAX
`SARIMAX (1,1,1)x(1,1,0)_12` achieved the **lowest Test RMSE (220.98)** and was the only model with a **positive test-set $R^2$ (0.1020)**. 
- The seasonal autoregressive parameter properly captured the recurring annual cyclicality in tax refund season (March) and year-end banking reconciliation (November–December).
- It provides a rigorous, mathematically sound statistical alternative to machine learning tree ensembles.

### 5.3 Baseline Competitiveness
The simple persistence baseline (`Naive`) and `3-Month Moving Average` demonstrated strong performance (CV WAPE ~9.8% to 10.2%). This occurs because monthly consumer complaints exhibit strong month-to-month inertia: consumer grievance rates rarely drop by 50% or double overnight during business-as-usual operations. However, baseline models cannot incorporate leading multi-product signals or accommodate structural regulatory changes.

---

## 6. Disaggregated Product-Level Forecasting Performance

To address the assignment mandate of investigating forecasting by financial product, separate recursive forecasters were trained for Bank of America's four largest product lines:

| Financial Product Category | Historical Total Volume | Test Period Mean / Mo | Forecast MAE | Forecast RMSE | **Forecast WAPE (%)** | Forecast sMAPE (%) | Operational Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Credit reporting, repair services...**| 7,710 (12.3%) | 170.7 complaints | **20.41** | **26.23** | **11.96%** | **12.47%** | **High Precision:** Highly consistent dispute volumes. |
| **Credit card or prepaid card** | 16,197 (25.9%) | 314.4 complaints | **46.78** | **81.31** | **14.88%** | **13.66%** | **Solid Accuracy:** Well-behaved transactional card claims. |
| **Checking or savings account** | 24,814 (39.7%) | 540.6 complaints | **131.66** | **192.34** | **24.36%** | **24.70%** | **Moderate Variance:** Affected by July 2023 fee penalty surge. |
| **Mortgage** | 6,601 (10.6%) | 46.1 complaints | **55.96** | **58.13** | **121.28%** | **75.99%** | **Structural Drift:** Rising interest rates reduced mortgage originations from 2021 to 2023. |

---

## 7. Model Selection & Deployment Artifacts

`Random Forest Regressor` was chosen as the **Machine Learning Champion**, supported by `SARIMAX` as the **Statistical Challenger**:
- The model was refit on the complete 75-month historical series (through July 2023).
- Model parameters, preprocessing logic, and historical buffers were serialized to:
  `models/complaint_volume_forecaster.joblib`
- Specialized product models were serialized to:
  `models/product_forecasters.joblib`
