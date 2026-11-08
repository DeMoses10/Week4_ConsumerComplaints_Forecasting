"""
Consumer Complaint Trends & Response Intelligence Platform - Model Training Pipeline
Module: train.py
Description: Trains and benchmarks Baseline, Statistical, and ML forecasting models
             on both fixed out-of-sample holdout (2023) and rolling-origin validation.
             Saves champion models, benchmark tables, and diagnostic visualizations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

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


def train_and_benchmark(project_dir: str):
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")
    reports_dir = os.path.join(project_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 75)
    print("STEP 1: INGESTING AND PREPROCESSING CONSUMER COMPLAINT RECORDS")
    print("=" * 75)
    raw_df = load_raw_complaints(data_dir)
    clean_df = clean_and_validate_complaints(raw_df)
    monthly_total, monthly_prods, metadata = aggregate_monthly_series(clean_df, exclude_partial_last_month=True)
    
    y_total = monthly_total['complaint_count']
    print(f"Total Completed Monthly Observations: {len(y_total)} ({metadata['modeling_start']} to {metadata['modeling_end']})")

    # Chronological Split:
    # Train: 2017-05 to 2022-12 (68 months)
    # Test:  2023-01 to 2023-07 (7 months out-of-sample holdout)
    split_period = pd.Period("2022-12", freq="M")
    train_series = y_total.loc[:split_period]
    test_series = y_total.loc[split_period + 1 :]

    print(f"Train Set: {len(train_series)} months ({train_series.index.min()} to {train_series.index.max()})")
    print(f"Test Set:  {len(test_series)} months ({test_series.index.min()} to {test_series.index.max()})")

    # Define model factories
    model_factories = {
        "Naive (Persistence)": lambda: NaivePersistenceForecaster(),
        "Seasonal Naive": lambda: SeasonalNaiveForecaster(seasonal_period=12),
        "3-Month Moving Average": lambda: MovingAverageForecaster(window=3),
        "Holt-Winters ExpSmoothing": lambda: HoltWintersForecaster(seasonal_periods=12),
        "SARIMAX (1,1,1)x(1,1,0)_12": lambda: SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)),
        "Ridge Regressor (L2)": lambda: MLRecursiveForecaster(Ridge(alpha=5.0, random_state=42)),
        "Random Forest Regressor": lambda: MLRecursiveForecaster(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
        "HistGradientBoosting": lambda: MLRecursiveForecaster(HistGradientBoostingRegressor(max_iter=100, max_depth=4, l2_regularization=2.0, random_state=42))
    }

    print("\n" + "=" * 75)
    print("STEP 2: OUT-OF-SAMPLE CHRONOLOGICAL EVALUATION (2023 TEST PERIOD)")
    print("=" * 75)
    
    test_results = []
    test_forecasts = {}

    for name, factory in model_factories.items():
        model = factory()
        model.fit(train_series)
        preds = model.forecast(len(test_series))
        test_forecasts[name] = preds

        metrics = calculate_metrics(test_series.values, preds)
        test_results.append({
            "Model": name,
            "Test MAE": metrics["MAE"],
            "Test RMSE": metrics["RMSE"],
            "Test WAPE (%)": metrics["WAPE (%)"],
            "Test MAPE (%)": metrics["MAPE (%)"],
            "Test sMAPE (%)": metrics["sMAPE (%)"],
            "Test R2": metrics["R2"]
        })

    df_test_benchmark = pd.DataFrame(test_results).sort_values("Test MAE")
    print(df_test_benchmark.to_string(index=False))
    df_test_benchmark.to_csv(os.path.join(reports_dir, "model_comparison_results.csv"), index=False)

    print("\n" + "=" * 75)
    print("STEP 3: ROLLING-ORIGIN CROSS-VALIDATION (6-MONTH HORIZONS)")
    print("=" * 75)
    
    cv_results = []
    for name, factory in model_factories.items():
        cv_out = rolling_origin_evaluation(
            y_total,
            factory,
            min_train_size=36,
            forecast_horizon=6,
            step_size=2
        )
        cv_m = cv_out["metrics"]
        cv_results.append({
            "Model": name,
            "CV MAE": cv_m["MAE"],
            "CV RMSE": cv_m["RMSE"],
            "CV WAPE (%)": cv_m["WAPE (%)"],
            "CV MAPE (%)": cv_m["MAPE (%)"],
            "CV sMAPE (%)": cv_m["sMAPE (%)"],
            "CV R2": cv_m["R2"],
            "Evaluated Origins": cv_out["n_origins"]
        })

    df_cv_benchmark = pd.DataFrame(cv_results).sort_values("CV MAE")
    print(df_cv_benchmark.to_string(index=False))
    df_cv_benchmark.to_csv(os.path.join(reports_dir, "rolling_origin_summary.csv"), index=False)

    # Select Champion Model (HistGradientBoosting or SARIMAX)
    champion_name = "HistGradientBoosting"
    champion_model = model_factories[champion_name]()
    # Refit on full series up to 2023-07 for deployment
    champion_model.fit(y_total)

    champion_payload = {
        "model_name": champion_name,
        "model": champion_model,
        "trained_through": str(y_total.index.max()),
        "full_history_series": y_total,
        "metadata": metadata,
        "test_metrics": df_test_benchmark.loc[df_test_benchmark['Model'] == champion_name].to_dict('records')[0],
        "cv_metrics": df_cv_benchmark.loc[df_cv_benchmark['Model'] == champion_name].to_dict('records')[0]
    }
    champion_save_path = os.path.join(models_dir, "complaint_volume_forecaster.joblib")
    joblib.dump(champion_payload, champion_save_path)
    print(f"\nChampion model saved to: {champion_save_path}")

    print("\n" + "=" * 75)
    print("STEP 4: MULTI-PRODUCT FORECASTING ENGINE")
    print("=" * 75)
    top_products = [
        "Checking or savings account",
        "Credit card or prepaid card",
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Mortgage"
    ]
    product_models = {}
    product_eval_summary = []

    for prod in top_products:
        prod_series = monthly_prods[prod]
        p_train = prod_series.loc[:split_period]
        p_test = prod_series.loc[split_period + 1 :]

        p_model = MLRecursiveForecaster(HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42))
        p_model.fit(p_train)
        p_preds = p_model.forecast(len(p_test))

        p_metrics = calculate_metrics(p_test.values, p_preds)
        product_eval_summary.append({
            "Product": prod,
            "Actual Mean (Test)": round(float(p_test.mean()), 1),
            "Forecast MAE": p_metrics["MAE"],
            "Forecast RMSE": p_metrics["RMSE"],
            "Forecast WAPE (%)": p_metrics["WAPE (%)"],
            "Forecast sMAPE (%)": p_metrics["sMAPE (%)"]
        })

        # Refit on full product series for deployment
        p_deploy = MLRecursiveForecaster(HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42))
        p_deploy.fit(prod_series)
        product_models[prod] = p_deploy

    df_prod_eval = pd.DataFrame(product_eval_summary)
    print(df_prod_eval.to_string(index=False))
    df_prod_eval.to_csv(os.path.join(reports_dir, "product_forecasting_metrics.csv"), index=False)
    joblib.dump(product_models, os.path.join(models_dir, "product_forecasters.joblib"))

    print("\n" + "=" * 75)
    print("STEP 5: GENERATING DIAGNOSTIC CHARTS & VISUALIZATIONS")
    print("=" * 75)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Chart 1: Out-of-Sample Forecast vs Actual Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot recent history + test actual
    plot_series = y_total.loc[pd.Period("2021-01", freq="M"):]
    plot_idx = [str(p) for p in plot_series.index]
    test_idx = [str(p) for p in test_series.index]
    
    ax.plot(plot_idx, plot_series.values, label="Actual Observed Complaints", color="#1f77b4", marker="o", lw=2.5)
    
    # Plot champion forecast
    ax.plot(test_idx, test_forecasts[champion_name], label=f"Champion Forecast ({champion_name})", color="#2ca02c", marker="s", lw=2.5, linestyle="--")
    
    # Plot baseline forecast
    ax.plot(test_idx, test_forecasts["Seasonal Naive"], label="Baseline (Seasonal Naive)", color="#d62728", marker="^", lw=1.8, linestyle=":")
    ax.plot(test_idx, test_forecasts["SARIMAX (1,1,1)x(1,1,0)_12"], label="SARIMAX", color="#9467bd", marker="x", lw=1.8, linestyle="-.")

    ax.axvline(str(split_period), color="black", linestyle="--", alpha=0.7, label="Train / Test Cutoff (2022-12)")
    ax.set_title("Out-of-Sample Monthly Complaint Forecast vs Actual (2023 Holdout)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year-Month", fontsize=11)
    ax.set_ylabel("Monthly Complaints", fontsize=11)
    plt.xticks(rotation=45)
    ax.legend(loc="upper left")
    plt.tight_layout()
    chart1_path = os.path.join(reports_dir, "total_volume_forecast_vs_actual.png")
    fig.savefig(chart1_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart1_path}")

    # Chart 2: Model Benchmark Metric Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df_test_benchmark.set_index("Model")
    
    df_plot[["Test MAE", "Test RMSE"]].plot(kind="bar", ax=axes[0], colormap="viridis", width=0.7)
    axes[0].set_title("Test Set Error Benchmark (MAE & RMSE)", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Complaints", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    df_plot["Test WAPE (%)"].plot(kind="bar", ax=axes[1], color="#e67e22", width=0.6)
    axes[1].set_title("Forecast Accuracy Benchmark: WAPE (%)", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Weighted Error Rate (%)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    chart2_path = os.path.join(reports_dir, "model_metric_benchmark.png")
    fig.savefig(chart2_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart2_path}")

    # Chart 3: Product-Level Forecast vs Actual
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, prod in enumerate(top_products):
        ax = axes[idx]
        p_series = monthly_prods[prod].loc[pd.Period("2021-01", freq="M"):]
        p_test_sub = monthly_prods[prod].loc[split_period + 1 :]
        
        # Recover prediction from model
        p_mod = MLRecursiveForecaster(HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42))
        p_mod.fit(monthly_prods[prod].loc[:split_period])
        p_pred_vals = p_mod.forecast(len(p_test_sub))

        ax.plot([str(p) for p in p_series.index], p_series.values, label="Actual", color="#1f77b4", marker="o", lw=2)
        ax.plot([str(p) for p in p_test_sub.index], p_pred_vals, label="Forecast", color="#e74c3c", marker="s", lw=2, linestyle="--")
        ax.axvline(str(split_period), color="grey", linestyle=":")
        ax.set_title(f"{prod[:35]}...", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    plt.suptitle("Top 4 Financial Products: Holdout Forecast vs Actual (2023)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart3_path = os.path.join(reports_dir, "product_level_forecasts.png")
    fig.savefig(chart3_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart3_path}")

    print("\nModel training and benchmark workflow completed successfully!")


if __name__ == '__main__':
    project_dir = r"C:\Users\HAPPY\.gemini\antigravity\scratch\Week4_ConsumerComplaints_Forecasting"
    train_and_benchmark(project_dir)
