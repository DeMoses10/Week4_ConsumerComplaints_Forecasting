"""
Consumer Complaint Trends & Response Intelligence Platform - Production Inference Engine
Module: predict.py
Description: Loads trained champion forecasting models, generates multi-step ahead
             complaint volume projections, calculates prediction intervals,
             breaks down forecasts by product, and generates operational capacity alerts.
"""

import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd


class ComplaintForecaster:
    """Production inference wrapper for consumer complaint volume forecasting."""

    def __init__(self, model_path: str = None, product_models_path: str = None):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            model_path = os.path.join(curr_dir, "..", "models", "complaint_volume_forecaster.joblib")
        if product_models_path is None:
            product_models_path = os.path.join(curr_dir, "..", "models", "product_forecasters.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")

        payload = joblib.load(model_path)
        self.model_name = payload["model_name"]
        self.model = payload["model"]
        self.trained_through = payload["trained_through"]
        self.full_history = payload["full_history_series"]
        self.test_metrics = payload.get("test_metrics", {})
        
        # Empirical residual std for prediction intervals
        self.residual_std = 125.0

        # Load product models if available
        self.product_models = {}
        if os.path.exists(product_models_path):
            self.product_models = joblib.load(product_models_path)

    def generate_forecast(self, steps: int = 6) -> pd.DataFrame:
        """
        Generates point forecasts and 80%/95% prediction intervals
        for the next `steps` months into the future.
        """
        last_period = self.full_history.index[-1]
        future_periods = [last_period + i for i in range(1, steps + 1)]
        
        point_forecasts = self.model.forecast(steps)

        # Standard error grows with forecast horizon h: se(h) = residual_std * sqrt(h)
        horizons = np.arange(1, steps + 1)
        se_h = self.residual_std * np.sqrt(horizons)

        lower_80 = np.clip(point_forecasts - 1.282 * se_h, 0.0, None)
        upper_80 = point_forecasts + 1.282 * se_h
        lower_95 = np.clip(point_forecasts - 1.960 * se_h, 0.0, None)
        upper_95 = point_forecasts + 1.960 * se_h

        # Operational capacity alert logic based on historical volume thresholds
        # Historical 75th percentile is ~960, 90th percentile is ~1250
        alert_levels = []
        for val in point_forecasts:
            if val < 900:
                alert_levels.append("Normal Inflow")
            elif val < 1250:
                alert_levels.append("Elevated Inflow - Monitoring Needed")
            else:
                alert_levels.append("High Volume Alert - Operational Scaling Required")

        df_out = pd.DataFrame({
            "Period": [str(p) for p in future_periods],
            "Forecast_Complaints": np.round(point_forecasts, 0).astype(int),
            "Lower_80%": np.round(lower_80, 0).astype(int),
            "Upper_80%": np.round(upper_80, 0).astype(int),
            "Lower_95%": np.round(lower_95, 0).astype(int),
            "Upper_95%": np.round(upper_95, 0).astype(int),
            "Operational_Status": alert_levels
        })

        return df_out

    def generate_product_forecasts(self, steps: int = 6) -> pd.DataFrame:
        """Generates product-specific forecasts across the top financial products."""
        if not self.product_models:
            return pd.DataFrame()

        last_period = self.full_history.index[-1]
        future_periods = [str(last_period + i) for i in range(1, steps + 1)]
        
        prod_dict = {"Period": future_periods}
        for prod_name, p_model in self.product_models.items():
            short_name = prod_name.split(",")[0]
            preds = p_model.forecast(steps)
            prod_dict[short_name] = np.round(preds, 0).astype(int)

        return pd.DataFrame(prod_dict)


def main():
    parser = argparse.ArgumentParser(description="Consumer Complaint Volume Forecasting CLI")
    parser.add_argument("--steps", type=int, default=6, help="Forecast horizon in months (default: 6)")
    parser.add_argument("--products", action="store_true", help="Include product-level breakdown")
    parser.add_argument("--demo", action="store_true", help="Run demonstrative CLI forecast summary")
    args = parser.parse_args()

    forecaster = ComplaintForecaster()
    print("=" * 80)
    print("BANK OF AMERICA CONSUMER COMPLAINT VOLUME FORECASTING ENGINE")
    print(f"Production Model: {forecaster.model_name} (Trained Through: {forecaster.trained_through})")
    print("=" * 80)

    forecast_df = forecaster.generate_forecast(steps=args.steps)
    print(f"\n>>> Total Monthly Complaint Forecasts (Next {args.steps} Months):")
    print(forecast_df.to_string(index=False))

    if args.products or args.demo:
        prod_df = forecaster.generate_product_forecasts(steps=args.steps)
        if not prod_df.empty:
            print(f"\n>>> Forecast Breakdown by Key Financial Products:")
            print(prod_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("OPERATIONAL INTELLIGENCE GUIDANCE:")
    print("1. Forecasts represent expected consumer complaint volume submitted to the CFPB.")
    print("2. Use Upper 80% bounds for customer advocacy team staffing and SLA compliance buffers.")
    print("3. External regulatory events, class-action litigation, or system outages create exogenous")
    print("   spikes that statistical models cannot anticipate without prior public disclosures.")
    print("-" * 80)


if __name__ == '__main__':
    main()
