"""
Consumer Complaint Trends & Response Intelligence Platform - Time Series Preprocessing
Module: data_preprocessing.py
Description: Ingests raw CFPB Bank of America complaint records, performs schema validation,
             analyzes reporting periods, handles partial months, aggregates into monthly counts,
             and constructs leak-free time-series features.
"""

import os
import numpy as np
import pandas as pd


def load_raw_complaints(data_dir: str) -> pd.DataFrame:
    """Loads raw complaints from the supplied Excel workbook."""
    excel_path = os.path.join(data_dir, "Consumer_Complaints.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Missing complaints workbook at {excel_path}")
    
    df = pd.read_excel(excel_path, sheet_name="Data")
    return df


def clean_and_validate_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates schema, checks uniqueness, cleans dates, and computes forwarding intervals.
    """
    df_clean = df.copy()

    # 1. Uniqueness check
    assert df_clean['Complaint ID'].nunique() == len(df_clean), "Duplicate Complaint IDs detected!"

    # 2. Date parsing
    df_clean['Date submitted'] = pd.to_datetime(df_clean['Date submitted'])
    df_clean['Date received'] = pd.to_datetime(df_clean['Date received'])

    # 3. CFPB Complaint-Forwarding Interval (days)
    df_clean['forwarding_interval_days'] = (
        df_clean['Date received'] - df_clean['Date submitted']
    ).dt.total_seconds() / 86400.0

    # 4. Standardize text categories
    df_clean['Product'] = df_clean['Product'].fillna('Unknown Product').astype(str).str.strip()
    df_clean['Issue'] = df_clean['Issue'].fillna('Unknown Issue').astype(str).str.strip()
    df_clean['Submitted via'] = df_clean['Submitted via'].fillna('Web').astype(str).str.strip()

    # 5. Calendar periods
    df_clean['YearMonth'] = df_clean['Date submitted'].dt.to_period('M')
    df_clean['Year'] = df_clean['Date submitted'].dt.year
    df_clean['Month'] = df_clean['Date submitted'].dt.month

    return df_clean


def aggregate_monthly_series(df_clean: pd.DataFrame, exclude_partial_last_month: bool = True):
    """
    Aggregates complaints into monthly total and product-level time series.
    Note on partial months:
    - May 2017: contains full 31 calendar days (May 1 to May 31, 744 complaints).
    - August 2023: cutoff on August 28 (28 days, reporting cut-off lag, 717 complaints).
    If exclude_partial_last_month=True, August 2023 is excluded from the primary modeling series
    (covering 75 complete continuous months: 2017-05 to 2023-07).
    """
    # Total monthly complaints
    monthly_total = df_clean.groupby('YearMonth').size().rename('complaint_count').to_frame()
    monthly_total.index.name = 'Period'

    # Product-level monthly complaints
    monthly_products = df_clean.pivot_table(
        index='YearMonth',
        columns='Product',
        values='Complaint ID',
        aggfunc='count',
        fill_value=0
    )
    monthly_products.index.name = 'Period'

    # Record August 2023 partial stats
    partial_august_actual = monthly_total.loc[pd.Period('2023-08', freq='M'), 'complaint_count']
    partial_august_prorated = round(partial_august_actual * (31.0 / 28.0), 1)

    if exclude_partial_last_month and pd.Period('2023-08', freq='M') in monthly_total.index:
        monthly_total_modeling = monthly_total.drop(pd.Period('2023-08', freq='M'))
        monthly_products_modeling = monthly_products.drop(pd.Period('2023-08', freq='M'))
    else:
        monthly_total_modeling = monthly_total.copy()
        monthly_products_modeling = monthly_products.copy()

    # Ensure continuous frequency index
    full_idx = pd.period_range(
        start=monthly_total_modeling.index.min(),
        end=monthly_total_modeling.index.max(),
        freq='M'
    )
    monthly_total_modeling = monthly_total_modeling.reindex(full_idx, fill_value=0)
    monthly_products_modeling = monthly_products_modeling.reindex(full_idx, fill_value=0)

    metadata = {
        "total_records": len(df_clean),
        "total_months_raw": len(monthly_total),
        "complete_months_count": len(monthly_total_modeling),
        "modeling_start": str(monthly_total_modeling.index.min()),
        "modeling_end": str(monthly_total_modeling.index.max()),
        "partial_august_actual": int(partial_august_actual),
        "partial_august_prorated": float(partial_august_prorated)
    }

    return monthly_total_modeling, monthly_products_modeling, metadata


def build_time_series_features(df_ts: pd.DataFrame, target_col: str = 'complaint_count') -> pd.DataFrame:
    """
    Constructs leak-free lag and temporal features for machine learning forecasting.
    All lag and rolling features at time t are computed strictly from values <= t-1.
    """
    df_feat = df_ts.copy()
    y = df_feat[target_col]

    # 1. Autoregressive Lag Features (strictly lagged by 1 or more periods)
    for lag in [1, 2, 3, 6, 12]:
        df_feat[f'lag_{lag}'] = y.shift(lag)

    # 2. Rolling Window Statistics (shifted by 1 to prevent future leakage)
    df_feat['rolling_mean_3m'] = y.shift(1).rolling(window=3, min_periods=1).mean()
    df_feat['rolling_std_3m'] = y.shift(1).rolling(window=3, min_periods=1).std().fillna(0.0)
    df_feat['rolling_mean_6m'] = y.shift(1).rolling(window=6, min_periods=1).mean()
    df_feat['rolling_std_6m'] = y.shift(1).rolling(window=6, min_periods=1).std().fillna(0.0)
    df_feat['rolling_mean_12m'] = y.shift(1).rolling(window=12, min_periods=1).mean()

    # 3. Momentum & Relative Differences
    df_feat['diff_lag1_lag2'] = df_feat['lag_1'] - df_feat['lag_2']
    df_feat['diff_lag1_lag12'] = df_feat['lag_1'] - df_feat['lag_12']
    df_feat['ratio_lag1_to_3m'] = df_feat['lag_1'] / (df_feat['rolling_mean_3m'] + 1e-5)

    # 4. Calendar & Seasonal Features
    if isinstance(df_feat.index, pd.PeriodIndex):
        dt_idx = df_feat.index.to_timestamp()
    else:
        dt_idx = pd.to_datetime(df_feat.index)

    df_feat['month'] = dt_idx.month
    df_feat['quarter'] = dt_idx.quarter
    df_feat['year'] = dt_idx.year

    # Cyclical Sine/Cosine encoding for month
    df_feat['sin_month'] = np.sin(2 * np.pi * df_feat['month'] / 12.0)
    df_feat['cos_month'] = np.cos(2 * np.pi * df_feat['month'] / 12.0)

    # 5. Deterministic Trend Step
    df_feat['trend_index'] = np.arange(len(df_feat))

    return df_feat


def get_feature_column_names():
    """Returns list of feature column names for machine learning models."""
    return [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3m', 'rolling_std_3m',
        'rolling_mean_6m', 'rolling_std_6m',
        'rolling_mean_12m',
        'diff_lag1_lag2', 'diff_lag1_lag12', 'ratio_lag1_to_3m',
        'sin_month', 'cos_month', 'quarter', 'trend_index'
    ]


if __name__ == '__main__':
    data_dir = r"C:\Users\HAPPY\.gemini\antigravity\scratch\Week4_ConsumerComplaints_Forecasting\data"
    print("Loading raw complaints...")
    raw_df = load_raw_complaints(data_dir)
    print("Cleaning and validating schema...")
    clean_df = clean_and_validate_complaints(raw_df)
    
    monthly_total, monthly_prods, meta = aggregate_monthly_series(clean_df, exclude_partial_last_month=True)
    print("\nDataset Metadata:", meta)
    
    # Save processed series
    total_csv_path = os.path.join(data_dir, "monthly_complaints_total.csv")
    prods_csv_path = os.path.join(data_dir, "monthly_complaints_by_product.csv")
    monthly_total.to_csv(total_csv_path)
    monthly_prods.to_csv(prods_csv_path)
    print(f"Saved total monthly counts to: {total_csv_path}")
    print(f"Saved product monthly counts to: {prods_csv_path}")

    # Build features
    df_features = build_time_series_features(monthly_total)
    feat_cols = get_feature_column_names()
    print(f"\nConstructed {len(feat_cols)} time-series features.")
    print("Sample feature values (last 3 rows):")
    print(df_features[feat_cols].tail(3))
