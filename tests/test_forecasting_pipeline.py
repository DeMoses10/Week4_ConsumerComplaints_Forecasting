"""
Consumer Complaint Trends & Response Intelligence Platform - Automated Test Suite
Module: test_forecasting_pipeline.py
Description: Unittest suite validating data integrity, leak-free feature construction,
             model training, forecasting interfaces, and artifact loading.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

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
    MLRecursiveForecaster
)
from predict import ComplaintForecaster
from sklearn.linear_model import Ridge


class TestComplaintForecastingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        cls.data_dir = os.path.join(cls.project_dir, "data")
        cls.model_path = os.path.join(cls.project_dir, "models", "complaint_volume_forecaster.joblib")
        
        # Load small sample for fast tests
        cls.raw_df = load_raw_complaints(cls.data_dir)
        cls.clean_df = clean_and_validate_complaints(cls.raw_df)
        cls.monthly_total, cls.monthly_prods, cls.meta = aggregate_monthly_series(cls.clean_df, exclude_partial_last_month=True)
        cls.y_series = cls.monthly_total['complaint_count']

    def test_raw_dataset_schema_and_uniqueness(self):
        """Verifies candidate primary key uniqueness and core date fields."""
        self.assertEqual(len(self.raw_df), 62516)
        self.assertEqual(self.raw_df['Complaint ID'].nunique(), 62516)
        self.assertIn('Date submitted', self.raw_df.columns)
        self.assertIn('Product', self.raw_df.columns)

    def test_time_series_continuity_and_partial_month(self):
        """Ensures complete 75-month continuous series from 2017-05 to 2023-07 with zero gaps."""
        self.assertEqual(len(self.monthly_total), 75)
        self.assertEqual(str(self.monthly_total.index.min()), "2017-05")
        self.assertEqual(str(self.monthly_total.index.max()), "2023-07")
        
        # Confirm August 2023 was separated as partial
        self.assertEqual(self.meta["partial_august_actual"], 717)
        self.assertGreater(self.meta["partial_august_prorated"], 717)

    def test_leak_free_feature_engineering(self):
        """Ensures that lag features at index t are strictly based on prior values <= t-1."""
        df_feats = build_time_series_features(self.monthly_total)
        feat_cols = get_feature_column_names()

        for col in feat_cols:
            self.assertIn(col, df_feats.columns)

        # Check that lag_1 at index i equals complaint_count at index i-1
        self.assertEqual(df_feats['lag_1'].iloc[1], self.monthly_total['complaint_count'].iloc[0])
        self.assertEqual(df_feats['lag_12'].iloc[12], self.monthly_total['complaint_count'].iloc[0])

    def test_forecasting_models_interface(self):
        """Tests that all forecasting models fit on history and generate non-negative forecasts."""
        train_sub = self.y_series.iloc[:50]
        horizon = 6

        models = [
            NaivePersistenceForecaster(),
            SeasonalNaiveForecaster(seasonal_period=12),
            MovingAverageForecaster(window=3),
            HoltWintersForecaster(seasonal_periods=12),
            SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
            MLRecursiveForecaster(Ridge(alpha=1.0))
        ]

        for mod in models:
            mod.fit(train_sub)
            preds = mod.forecast(horizon)
            self.assertEqual(len(preds), horizon, f"Failed length for {mod.__class__.__name__}")
            self.assertTrue(np.all(preds >= 0), f"Negative prediction for {mod.__class__.__name__}")

    def test_metrics_calculation(self):
        """Validates accuracy calculation (MAE, RMSE, WAPE, sMAPE)."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])
        m = calculate_metrics(y_true, y_pred)

        self.assertAlmostEqual(m["MAE"], 6.67, places=1)
        self.assertGreater(m["WAPE (%)"], 0.0)
        self.assertLess(m["WAPE (%)"], 10.0)

    def test_production_forecaster_artifact(self):
        """Ensures serialized champion model loads and serves production forecasts."""
        self.assertTrue(os.path.exists(self.model_path), f"Missing model artifact at {self.model_path}")
        forecaster = ComplaintForecaster(self.model_path)
        
        forecast_df = forecaster.generate_forecast(steps=6)
        self.assertEqual(len(forecast_df), 6)
        self.assertIn("Forecast_Complaints", forecast_df.columns)
        self.assertIn("Lower_80%", forecast_df.columns)
        self.assertIn("Upper_80%", forecast_df.columns)
        self.assertIn("Operational_Status", forecast_df.columns)
        
        # Upper bounds must be >= forecast
        self.assertTrue((forecast_df["Upper_80%"] >= forecast_df["Forecast_Complaints"]).all())
        self.assertTrue((forecast_df["Lower_80%"] <= forecast_df["Forecast_Complaints"]).all())


if __name__ == '__main__':
    unittest.main()
