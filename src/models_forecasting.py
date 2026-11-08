"""
Consumer Complaint Trends & Response Intelligence Platform - Forecasting Engine
Module: models_forecasting.py
Description: Implements statistical and machine learning forecasting architectures:
             - Baselines: Naive, Seasonal Naive, 3-Month Moving Average
             - Classical Time Series: Holt-Winters Exponential Smoothing, SARIMAX
             - Machine Learning: Ridge Regression, Random Forest, HistGradientBoosting
             Includes recursive multi-step forecasting and rolling-origin validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_preprocessing import build_time_series_features, get_feature_column_names


# =====================================================================
# EVALUATION METRICS
# =====================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive time-series forecasting accuracy metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Prevent negative forecasts for complaint counts
    y_pred = np.clip(y_pred, 0.0, None)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Weighted Absolute Percentage Error (WAPE)
    sum_actual = float(np.sum(y_true))
    wape = float((np.sum(np.abs(y_true - y_pred)) / (sum_actual + 1e-8)) * 100.0)

    # Mean Absolute Percentage Error (MAPE)
    mask = y_true > 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = 0.0

    # Symmetric Mean Absolute Percentage Error (sMAPE)
    denom = np.abs(y_true) + np.abs(y_pred)
    smape_mask = denom > 0
    if np.any(smape_mask):
        smape = float(np.mean(200.0 * np.abs(y_true[smape_mask] - y_pred[smape_mask]) / denom[smape_mask]))
    else:
        smape = 0.0

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE (%)": round(wape, 2),
        "MAPE (%)": round(mape, 2),
        "sMAPE (%)": round(smape, 2),
        "R2": round(r2, 4)
    }


# =====================================================================
# BASELINE FORECASTERS
# =====================================================================

class NaivePersistenceForecaster:
    """Predicts the most recently observed value: y_hat_{t+h} = y_t."""
    def __init__(self):
        self.last_value_ = None

    def fit(self, y_train: pd.Series):
        self.last_value_ = float(y_train.iloc[-1])
        return self

    def forecast(self, steps: int) -> np.ndarray:
        return np.full(steps, self.last_value_)


class SeasonalNaiveForecaster:
    """Predicts the value from the same month last year: y_hat_{t+h} = y_{t+h-12}."""
    def __init__(self, seasonal_period: int = 12):
        self.period = seasonal_period
        self.history_ = None

    def fit(self, y_train: pd.Series):
        self.history_ = y_train.to_numpy(dtype=float)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        preds = []
        hist = list(self.history_)
        for _ in range(steps):
            pred = hist[-self.period] if len(hist) >= self.period else hist[-1]
            preds.append(pred)
            hist.append(pred)
        return np.array(preds)


class MovingAverageForecaster:
    """Predicts the mean of the last k observed values: y_hat_{t+h} = mean(y_{t-k+1:t})."""
    def __init__(self, window: int = 3):
        self.window = window
        self.rolling_mean_ = None

    def fit(self, y_train: pd.Series):
        self.rolling_mean_ = float(y_train.iloc[-self.window:].mean())
        return self

    def forecast(self, steps: int) -> np.ndarray:
        return np.full(steps, self.rolling_mean_)


# =====================================================================
# STATISTICAL TIME SERIES MODELS
# =====================================================================

class HoltWintersForecaster:
    """Holt-Winters Exponential Smoothing with additive trend and seasonality."""
    def __init__(self, seasonal_periods: int = 12):
        self.seasonal_periods = seasonal_periods
        self.model_fit_ = None

    def fit(self, y_train: pd.Series):
        # Convert index to numeric or timestamp
        y_vals = y_train.astype(float).to_numpy()
        try:
            model = ExponentialSmoothing(
                y_vals,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated"
            )
            self.model_fit_ = model.fit(optimized=True)
        except Exception:
            # Fallback to simple additive trend if seasonal fitting encounters convergence issues
            model = ExponentialSmoothing(y_vals, trend="add", initialization_method="estimated")
            self.model_fit_ = model.fit(optimized=True)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        preds = self.model_fit_.forecast(steps)
        return np.clip(preds, 0.0, None)


class SARIMAXForecaster:
    """Seasonal ARIMA Model for monthly complaint series."""
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit_ = None

    def fit(self, y_train: pd.Series):
        y_vals = y_train.astype(float).to_numpy()
        try:
            model = SARIMAX(
                y_vals,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit_ = model.fit(disp=False)
        except Exception:
            # Fallback to non-seasonal ARIMA
            model = SARIMAX(
                y_vals,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit_ = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        preds = self.model_fit_.forecast(steps)
        return np.clip(preds, 0.0, None)


# =====================================================================
# MACHINE LEARNING RECURSIVE FORECASTER
# =====================================================================

class MLRecursiveForecaster:
    """
    Wraps scikit-learn regressors for recursive multi-step forecasting.
    Dynamically recalculates lag and rolling statistics at each forecast horizon.
    """
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.feature_cols = get_feature_column_names()
        self.history_series_ = None

    def fit(self, y_train: pd.Series):
        self.history_series_ = y_train.copy()
        
        # Build features for training
        df_feat = build_time_series_features(y_train.to_frame('complaint_count'), target_col='complaint_count')
        # Drop rows with NaNs caused by the longest lag (lag 12)
        df_train = df_feat.dropna(subset=self.feature_cols)
        
        X = df_train[self.feature_cols]
        y = df_train['complaint_count']
        
        self.estimator.fit(X, y)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """Generates h-step recursive forecasts updating lags at each step."""
        sim_history = self.history_series_.copy()
        preds = []

        last_period = sim_history.index[-1]
        
        for step in range(1, steps + 1):
            next_period = last_period + step
            # Create a placeholder row
            temp_series = sim_history.copy()
            temp_series.loc[next_period] = 0.0  # Temporary placeholder
            
            # Recompute features
            df_feat = build_time_series_features(temp_series.to_frame('complaint_count'), target_col='complaint_count')
            
            # Extract feature row for the next step
            X_step = df_feat.loc[[next_period], self.feature_cols]
            pred_val = float(self.estimator.predict(X_step)[0])
            pred_val = max(0.0, pred_val)
            
            preds.append(pred_val)
            # Update history with predicted value for subsequent steps
            sim_history.loc[next_period] = pred_val

        return np.array(preds)


# =====================================================================
# ROLLING-ORIGIN VALIDATION (TIME-SERIES CROSS VALIDATION)
# =====================================================================

def rolling_origin_evaluation(
    y_series: pd.Series,
    model_factory,
    min_train_size: int = 36,
    forecast_horizon: int = 6,
    step_size: int = 1
) -> Dict[str, Any]:
    """
    Executes expanding window rolling-origin validation:
    Trains on 0..t, forecasts t+1..t+h, steps forward by step_size.
    """
    n = len(y_series)
    all_actuals = []
    all_preds = []
    origin_records = []

    for end_idx in range(min_train_size, n - forecast_horizon + 1, step_size):
        train_sub = y_series.iloc[:end_idx]
        test_sub = y_series.iloc[end_idx : end_idx + forecast_horizon]

        model = model_factory()
        model.fit(train_sub)
        preds = model.forecast(forecast_horizon)

        all_actuals.extend(test_sub.values)
        all_preds.extend(preds)

        origin_records.append({
            "origin_period": str(train_sub.index[-1]),
            "horizon_actuals": test_sub.values.tolist(),
            "horizon_preds": preds.tolist(),
            "origin_mae": float(mean_absolute_error(test_sub.values, preds))
        })

    metrics = calculate_metrics(np.array(all_actuals), np.array(all_preds))
    return {
        "metrics": metrics,
        "n_origins": len(origin_records),
        "details": origin_records
    }
