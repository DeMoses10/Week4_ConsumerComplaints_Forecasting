import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\HAPPY\.gemini\antigravity\scratch\Week4_ConsumerComplaints_Forecasting\data"

def inspect():
    dict_path = os.path.join(DATA_DIR, "ConsumerComplaints_DataDictionary.csv")
    if os.path.exists(dict_path):
        print("=== DATA DICTIONARY ===")
        df_dict = pd.read_csv(dict_path)
        print(df_dict.to_string())
        print("\n")

    excel_path = os.path.join(DATA_DIR, "Consumer_Complaints.xlsx")
    print("=== READING EXCEL DATA ===")
    df = pd.read_excel(excel_path, sheet_name="Data")
    print(f"Data Shape: {df.shape}")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nUnique Complaint IDs:", df['Complaint ID'].nunique())
    print("Duplicates count:", df.duplicated().sum())

    # Date inspections
    df['Date submitted'] = pd.to_datetime(df['Date submitted'])
    df['Date received'] = pd.to_datetime(df['Date received'])
    print("\nDate submitted range:", df['Date submitted'].min(), "to", df['Date submitted'].max())
    print("Date received range:", df['Date received'].min(), "to", df['Date received'].max())

    # Daily distribution near boundaries
    print("\nEarliest submitted dates:")
    print(df['Date submitted'].dt.date.value_counts().sort_index().head(10))
    print("\nLatest submitted dates:")
    print(df['Date submitted'].dt.date.value_counts().sort_index().tail(10))

    # Forwarding interval
    df['forwarding_interval_days'] = (df['Date received'] - df['Date submitted']).dt.total_seconds() / 86400.0
    print("\nForwarding Interval (days) summary:")
    print(df['forwarding_interval_days'].describe())

    # Product breakdown
    print("\nComplaints by Product:")
    print(df['Product'].value_counts())

    # Monthly aggregation
    df['YearMonth'] = df['Date submitted'].dt.to_period('M')
    monthly = df.groupby('YearMonth').size()
    print("\nMonthly Complaints (first 10 months):")
    print(monthly.head(10))
    print("\nMonthly Complaints (last 10 months):")
    print(monthly.tail(10))
    print(f"Total months in range: {len(monthly)}")

    # Check for missing months in continuous date range
    all_months = pd.period_range(start=df['Date submitted'].min().to_period('M'),
                                 end=df['Date submitted'].max().to_period('M'),
                                 freq='M')
    missing_months = [m for m in all_months if m not in monthly.index]
    print(f"Missing months count: {len(missing_months)}")

if __name__ == '__main__':
    inspect()
