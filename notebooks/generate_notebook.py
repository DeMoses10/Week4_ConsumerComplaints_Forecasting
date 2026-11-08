"""
Builds and saves the complete Jupyter Notebook for Week 4 Machine Learning Forecasting.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Header & Executive Overview
cells.append(nbf.v4.new_markdown_cell("""# AnalystLab Africa Experience Lab - Week 4
## Machine Learning Track: Consumer Complaint Volume Forecasting Prototype
**Project Title:** Consumer Complaint Trends and Response Intelligence Platform  
**Target:** Forecasting Bank of America Monthly Consumer Complaints (2017–2023)  
**Programme:** AnalystLab Africa Experience Lab Internship  
**Author:** Machine Learning Engineering Intern  
**Date:** November 2026  

---

### Executive Problem Statement
Financial institutions receive thousands of consumer complaints regarding checking accounts, credit cards, mortgages, and payment systems forwarded by the Consumer Financial Protection Bureau (CFPB). Managing customer advocacy teams, regulatory response timelines, and legal compliance requires forecasting future complaint volumes with sufficient lead time.

Inability to anticipate complaint volumes leads to:
1. Operational bottlenecks and breaches of the statutory 15-day company response window.
2. Inefficient staffing and resource allocation across customer dispute resolution teams.
3. Lack of early warning for systemic product issues and escalating customer friction.

This notebook builds, evaluates, and tests an end-to-end, leak-free time-series forecasting solution to **estimate future monthly complaint volumes overall and across key financial products**.
"""))

# Cell 2: Imports & Environment
cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

# Add src to path
sys.path.append(os.path.abspath("../src"))
from data_preprocessing import (
    load_raw_complaints,
    clean_and_validate_complaints,
    aggregate_monthly_series,
    build_time_series_features,
    get_feature_column_names
)
from models_forecasting import (
    calculate_metrics,
    NaivePersistenceForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    HoltWintersForecaster,
    SARIMAXForecaster,
    MLRecursiveForecaster,
    rolling_origin_evaluation
)
from predict import ComplaintForecaster

print("Environment and packages loaded successfully.")
"""))

# Cell 3: Data Ingestion & Quality Assessment
cells.append(nbf.v4.new_markdown_cell("""## 1. Raw Dataset Ingestion & Quality Assessment
We inspect 62,516 historical complaint records for Bank of America spanning May 2017 to August 2023.
"""))

cells.append(nbf.v4.new_code_cell("""data_dir = "../data"
raw_df = load_raw_complaints(data_dir)
print(f"Raw Complaints Shape: {raw_df.shape}")
print(f"Unique Complaint IDs: {raw_df['Complaint ID'].nunique()}")
display(raw_df.head(3))
"""))

# Cell 4: Cleaning & Aggregation
cells.append(nbf.v4.new_markdown_cell("""## 2. Time-Series Aggregation & Partial-Month Analysis
### Boundary Periods Handling
- **May 2017:** Covers full 31 calendar days (May 1 to May 31, 744 complaints).
- **August 2023:** Cutoff on August 28 (28 days, reporting cutoff lag, 717 complaints).
For reliable continuous modeling, we utilize the **75 complete months (2017-05 to 2023-07)** and evaluate August 2023 separately for proration.
"""))

cells.append(nbf.v4.new_code_cell("""clean_df = clean_and_validate_complaints(raw_df)
monthly_total, monthly_prods, metadata = aggregate_monthly_series(clean_df, exclude_partial_last_month=True)

print("Dataset Aggregation Metadata:")
for k, v in metadata.items():
    print(f"  {k}: {v}")

display(monthly_total.head(5))
"""))

# Cell 5: Time Series Visual Exploration
cells.append(nbf.v4.new_markdown_cell("""## 3. Historical Complaint Volume Trends
We plot total monthly complaint volume and top financial product trajectories over time.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Total Monthly Volume
axes[0].plot([str(p) for p in monthly_total.index], monthly_total['complaint_count'].values, color='#1f77b4', marker='o', lw=2)
axes[0].set_title("Bank of America: Total Monthly Consumer Complaints (2017–2023)", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Complaints Count")

# Top 4 Products
top_prods = ['Checking or savings account', 'Credit card or prepaid card', 'Credit reporting, credit repair services, or other personal consumer reports', 'Mortgage']
for prod in top_prods:
    axes[1].plot([str(p) for p in monthly_prods.index], monthly_prods[prod].values, label=prod[:30], lw=1.8)

axes[1].set_title("Monthly Complaints by Top Financial Products", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Complaints Count")
axes[1].set_xlabel("Year-Month")
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

# Cell 6: Leakage-Free Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Leakage-Free Feature Engineering
To prevent future lookahead leakage, all lag features ($y_{t-1}, y_{t-2}, y_{t-3}, y_{t-6}, y_{t-12}$) and rolling statistics (3-month, 6-month, 12-month means and standard deviations) are strictly shifted by at least 1 month.
"""))

cells.append(nbf.v4.new_code_cell("""df_features = build_time_series_features(monthly_total)
feature_cols = get_feature_column_names()

print(f"Constructed {len(feature_cols)} features.")
display(df_features[feature_cols].tail(4))
"""))

# Cell 7: Model Benchmarking
cells.append(nbf.v4.new_markdown_cell("""## 5. Model Development & Comparative Benchmark
We evaluate 8 forecasting architectures on the **2023 holdout test set (2023-01 to 2023-07)**:
1. Baseline: Naive (Persistence)
2. Baseline: Seasonal Naive (Year-over-Year)
3. Baseline: 3-Month Moving Average
4. Statistical: Holt-Winters Exponential Smoothing
5. Statistical: SARIMAX (1,1,1)x(1,1,0)_12
6. ML: Ridge Regression (L2)
7. ML: Random Forest Regressor
8. ML: HistGradientBoostingRegressor
"""))

cells.append(nbf.v4.new_code_cell("""benchmark_csv = "../reports/model_comparison_results.csv"
if os.path.exists(benchmark_csv):
    df_bench = pd.read_csv(benchmark_csv)
    display(df_bench)
else:
    print("Run train.py to generate benchmarks.")
"""))

# Cell 8: Diagnostic Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 6. Forecast vs Actual Comparison & Benchmark Charts
"""))

cells.append(nbf.v4.new_code_cell("""# Display generated diagnostic figures
from IPython.display import Image, display

display(Image(filename="../reports/total_volume_forecast_vs_actual.png"))
display(Image(filename="../reports/model_metric_benchmark.png"))
display(Image(filename="../reports/product_level_forecasts.png"))
"""))

# Cell 9: Error Analysis & Ljung-Box Test
cells.append(nbf.v4.new_markdown_cell("""## 7. Residual Error Analysis & White Noise Diagnostics
We inspect test-set residuals month by month and evaluate the Ljung-Box test for residual autocorrelation.
"""))

cells.append(nbf.v4.new_code_cell("""res_breakdown = pd.read_csv("../reports/monthly_residual_breakdown.csv")
display(res_breakdown)

lb_results = pd.read_csv("../reports/ljung_box_test_results.csv")
print("\nLjung-Box Test for Autocorrelation in Residuals (p > 0.05 indicates white noise):")
display(lb_results)
"""))

# Cell 10: Production Inference Engine
cells.append(nbf.v4.new_markdown_cell("""## 8. Production Inference Prototype & Operational Alerts
We load the serialized production model artifact and generate 6-month forward projections with prediction intervals and operational workload flags.
"""))

cells.append(nbf.v4.new_code_cell("""forecaster = ComplaintForecaster("../models/complaint_volume_forecaster.joblib")
future_forecast = forecaster.generate_forecast(steps=6)
display(future_forecast)

product_forecast = forecaster.generate_product_forecasts(steps=6)
display(product_forecast)
"""))

# Cell 11: Responsible Use & Limitations
cells.append(nbf.v4.new_markdown_cell("""## 9. Model Limitations & Responsible Use Safeguards
1. **Operational Planning Only:** Forecasts represent complaint inflows for staffing and SLA management. They do not prove misconduct or regulatory guilt.
2. **Exogenous Shocks:** Statistical and ML models cannot anticipate unannounced regulatory fines, media coverage, or class-action litigation (e.g., the July 2023 surge).
3. **Continuous Monitoring:** Review model tracking metrics monthly and retrain whenever the Population Stability Index (PSI) exceeds 0.25.
"""))

nb.cells = cells

notebook_path = r"C:\Users\HAPPY\.gemini\antigravity\scratch\Week4_ConsumerComplaints_Forecasting\notebooks\consumer_complaint_forecasting.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter notebook written successfully to {notebook_path}")
