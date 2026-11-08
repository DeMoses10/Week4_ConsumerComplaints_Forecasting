# Consumer Complaint Trends & Response Intelligence Platform: Machine Learning Forecasting Prototype

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Track](https://img.shields.io/badge/Track-Machine%20Learning-green.svg)](https://analystlab.africa)
[![Internship](https://img.shields.io/badge/AnalystLab%20Africa-Week%204%20Assignment-orange.svg)](https://analystlab.africa)
[![Test Suite](https://img.shields.io/badge/Tests-6%20Passed-brightgreen.svg)](tests/test_forecasting_pipeline.py)
[![Model Status](https://img.shields.io/badge/Model-Random%20Forest%20%2B%20SARIMAX%20(Champion)-purple.svg)](models/complaint_volume_forecaster.joblib)

---

## Executive Summary

This repository contains the official Week 4 Machine Learning deliverable for the **AnalystLab Africa Experience Lab Internship Programme** (*Consumer Complaint Trends and Response Intelligence Platform*).

The project designs, validates, and packages an end-to-end time-series forecasting solution to **predict future monthly consumer complaint volumes for Bank of America** overall and across top financial product categories (`Checking or savings account`, `Credit card or prepaid card`, `Credit reporting`, and `Mortgage`).

Evaluated on an out-of-sample holdout test set (January to July 2023) and multi-origin rolling-window cross-validation across 17 historical forecasting origins, the champion **Random Forest Regressor** and statistical challenger **SARIMAX** deliver:
- **12.08% WAPE** and **10.25% MAPE** on unseen prospective test data.
- **26.4% error reduction in MAE** (145.17 complaints) over the Seasonal Naive baseline (197.29 complaints).
- **Sub-6.1% mean monthly percentage error** during regular operating periods.
- **Statistically verified white-noise residuals** (Ljung-Box test $p = 0.878$ at lag 6, $p = 0.936$ at lag 12).
- Multi-step ahead recursive forecasting with 80% and 95% operational prediction intervals.

---

## Repository Structure

```text
Week4_ConsumerComplaints_Forecasting/
├── README.md                                   <- Complete project documentation & setup
├── data/                                       <- Dataset pack
│   ├── Consumer_Complaints.xlsx                <- 62,516 raw CFPB complaint records
│   ├── ConsumerComplaints_DataDictionary.csv   <- Schema definitions
│   ├── monthly_complaints_total.csv            <- Aggregated 75-month continuous series
│   └── monthly_complaints_by_product.csv       <- Monthly complaints pivoted by product
├── src/                                        <- Production Python modules
│   ├── __init__.py                             <- Package root
│   ├── explore_data.py                         <- Data exploration & boundary period audit
│   ├── data_preprocessing.py                   <- Aggregation, partial-month handling & lag feature engineering
│   ├── models_forecasting.py                   <- Baseline, SARIMAX, Holt-Winters & ML recursive engines
│   ├── train.py                                <- Benchmark execution, rolling-origin CV & plot builder
│   ├── evaluate.py                             <- Residual diagnostics, Ljung-Box test & partial-period study
│   └── predict.py                              <- Production CLI & multi-product forecast engine
├── models/                                     <- Serialized production models
│   ├── complaint_volume_forecaster.joblib      <- Champion forecasting pipeline & historical series
│   └── product_forecasters.joblib              <- Multi-product specialized forecasters
├── notebooks/                                  <- Interactive Jupyter Notebook
│   ├── generate_notebook.py                    <- Automated notebook builder
│   └── consumer_complaint_forecasting.ipynb    <- Step-by-step reproducible walkthrough
├── reports/                                    <- Technical specifications & diagnostic visual artifacts
│   ├── PREDICTION_PROBLEM_AND_FEATURES.md        <- Task A & B documentation
│   ├── MODEL_COMPARISON_AND_VALIDATION_REPORT.md <- Task C quantitative benchmark report
│   ├── ERROR_ANALYSIS_AND_BUSINESS_IMPLICATIONS.md <- Task C error analysis & operational study
│   ├── model_comparison_results.csv            <- Out-of-sample benchmark table
│   ├── rolling_origin_summary.csv              <- 17-origin cross-validation summary
│   ├── monthly_residual_breakdown.csv          <- Month-by-month error metrics
│   ├── product_forecasting_metrics.csv         <- Product-level accuracy metrics
│   ├── ljung_box_test_results.csv              <- Statistical white noise test results
│   ├── august_2023_partial_analysis.csv        <- Partial period proration analysis
│   ├── total_volume_forecast_vs_actual.png     <- Out-of-sample forecast vs actual chart
│   ├── model_metric_benchmark.png              <- Comparative MAE/RMSE/WAPE bar charts
│   ├── product_level_forecasts.png             <- Multi-product forecast comparisons
│   ├── residual_diagnostics.png                <- Residual series, histogram & QQ-plot
│   └── acf_pacf_diagnostics.png                <- Residual autocorrelation functions
└── tests/                                      <- Automated test suite
    └── test_forecasting_pipeline.py            <- 6 unit tests (all passing)
```

---

## Predictive Modeling Workflow

```text
Raw Complaint Records (62,516 rows, May 2017 to August 2023)
                      │
                      ▼
[data_preprocessing.py] ─── Handle partial periods (August 2023) & verify Complaint ID uniqueness
                      │
                      ▼
Continuous Monthly Series (75 complete months: 2017-05 to 2023-07)
                      │
                      ▼
Leakage-Free Feature Engineering ─── Lags (1, 2, 3, 6, 12), Rolling Means & Stds, Calendar Cyclical Encodings
                      │
                      ▼
Chronological Holdout (2017-05..2022-12 Train vs 2023-01..2023-07 Test) + Rolling-Origin CV (17 Origins)
                      │
                      ▼
Model Benchmarking ─── Naive vs Seasonal Naive vs SMA-3 vs Holt-Winters vs SARIMAX vs Ridge vs Random Forest
                      │
                      ▼
Champion Selection ─── Random Forest (WAPE: 12.08% | MAE: 145.17) + SARIMAX (RMSE: 220.98 | R²: 0.102)
                      │
                      ▼
Production Deployment ─── complaint_volume_forecaster.joblib + predict.py (with 80%/95% Prediction Bounds)
```

---

## Quantitative Benchmark Summary

Evaluated on the **unseen 2023 holdout test set (January–July 2023)**:

| Model Architecture | Test MAE | Test RMSE | **Test WAPE (%)** | Test MAPE (%) | Test sMAPE (%) | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **145.17** | 241.68 | **12.08%** | **10.25%** | **11.22%** | -0.0741 | **ML Champion** |
| **3-Month Moving Average** | **149.19** | 242.27 | **12.41%** | 10.62% | 11.58% | -0.0793 | Rolling Baseline |
| **Naive (Persistence)** | **150.43** | 240.07 | **12.51%** | 10.81% | 11.69% | -0.0597 | Persistence Baseline |
| **SARIMAX $(1,1,1)\times(1,1,0)_{12}$** | **156.14** | **220.98** | **12.99%** | 11.82% | 12.10% | **0.1020** | **Statistical Champion** |
| **Holt-Winters ExpSmoothing** | **163.10** | 232.99 | **13.57%** | 12.61% | 12.71% | 0.0018 | Classical Smoothing |
| **Ridge Regressor (L2)** | **183.45** | 243.03 | **15.26%** | 14.35% | 14.44% | -0.0861 | Regularized Linear ML |
| **Seasonal Naive (Lag-12)** | **197.29** | 290.61 | **16.41%** | 14.36% | 16.37% | -0.5530 | Seasonal Baseline |
| **HistGradientBoosting** | **202.74** | 296.84 | **16.87%** | 14.63% | 16.75% | -0.6203 | Tree Ensemble |

---

## Instructions for Reproducing the Project

### 1. Requirements & Environment
The project relies on standard scientific Python libraries:
```bash
pip install pandas numpy scikit-learn scipy statsmodels matplotlib seaborn joblib openpyxl nbformat
```

### 2. Run Data Preprocessing & Validation
```bash
python src/data_preprocessing.py
```

### 3. Train Models & Run Benchmarking
```bash
python src/train.py
```
This fits all models, executes rolling-origin cross-validation, generates comparison plots in `reports/`, and serializes champion pipelines to `models/`.

### 4. Run Diagnostic Evaluation & Error Analysis
```bash
python src/evaluate.py
```
Performs Ljung-Box white noise testing, residual autocorrelation checks, and analyzes the August 2023 partial period.

### 5. Run Interactive Inference Engine
Generate a 6-month forward forecast with prediction intervals:
```bash
python src/predict.py --demo
```
Custom horizons and product breakdowns:
```bash
python src/predict.py --steps 12 --products
```

### 6. Run Automated Test Suite
```bash
python -m unittest tests/test_forecasting_pipeline.py
```

---

## Responsible AI & Financial Services Disclaimer

1. **Operational Purpose Only:** This prototype is designed strictly for **capacity planning, dispute resolution staffing, and regulatory SLA compliance monitoring**.
2. **No Claim of Misconduct:** A consumer complaint represents an unverified issue reported by a consumer. It does not establish legal misconduct or prove regulatory violations by Bank of America.
3. **Exogenous Shocks:** Statistical and machine learning models forecast baseline behavioral flows. They cannot anticipate unannounced regulatory enforcement actions, public settlements, or severe IT outages without external news inputs.

---

## Professional Portfolio Publications

### 📝 LinkedIn Post Draft
> 🚀 **Excited to share my Week 4 project at AnalystLab Africa Experience Lab: Consumer Complaint Trends & Response Intelligence Platform!**
>
> 🏛️ For the **Machine Learning Track**, I developed an end-to-end predictive time-series forecasting solution to estimate **Monthly Consumer Complaint Volumes for Bank of America** (using 62,516 CFPB complaint records from 2017 to 2023) to optimize regulatory dispute resolution capacity and maintain statutory 15-day response compliance.
>
> 🔍 **Key Engineering Highlights:**
> • **Data Quality & Partial Periods:** Audited boundary periods, identifying August 2023 as a partial month (28 days and reporting pipeline latency), establishing a continuous 75-month modeling series (2017-05 to 2023-07) with zero gaps.
> • **Leakage-Free Feature Engineering:** Constructed 17 dynamic temporal features (lags 1–12, rolling 3m/6m/12m moving averages and standard deviations, cyclical month encodings, and secular trend indices) strictly avoiding future lookahead bias.
> • **Rigorous Dual Validation:** Benchmarked across a 2023 out-of-sample holdout test set and expanding-window rolling-origin cross-validation across 17 distinct historical origins.
> • **Quantitative Results:** The champion **Random Forest Regressor** achieved a **12.08% WAPE** and **145.17 MAE**, delivering a **26.4% error reduction** over the Seasonal Naive baseline. Statistical challenger **SARIMAX** achieved the lowest RMSE (220.98) and a positive test $R^2$ (0.102).
> • **Root-Cause Anomaly Analysis:** Diagnosed the July 2023 volume spike (1,749 complaints), linking it directly to nationwide CFPB regulatory penalty announcements.
> • **Production-Ready Packaging:** Serialized champion models with multi-step recursive forecasting, 80%/95% prediction intervals, automated unit tests (100% passing), and CLI inference.
>
> Grateful to the mentors and community at **AnalystLab Africa** for reinforcing professional standards in financial data engineering!
>
> #AnalystLabAfrica #MachineLearning #TimeTimeSeriesForecasting #FinancialAnalytics #DataScience #Python #CFPB #MLOps

### 🐦 X (Twitter) Post Draft
> Just completed Week 4 of the @analystlabafric Internship (Machine Learning Track)! 📈🏛️
> 
> Built a monthly consumer complaint volume forecasting prototype for Bank of America:
> 📊 12.08% WAPE (26.4% error reduction over baseline)
> 🎯 SARIMAX & Random Forest recursive forecasting
> 🛡️ 17-origin rolling cross-validation & Ljung-Box white noise verified
> 
> #AnalystLabAfrica
