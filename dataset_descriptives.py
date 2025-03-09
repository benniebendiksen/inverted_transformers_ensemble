import pandas as pd
import os
import numpy as np


def print_df_stats(df):
    # Global statistics
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    # Show distribution of data types
    dtype_counts = df.dtypes.value_counts()
    print("\nData type distribution:")
    for dtype, count in dtype_counts.items():
        print(f"- {dtype}: {count} columns")

    # Total missing values
    total_missing = df.isnull().sum().sum()
    print(f"\nTotal missing values across all columns: {total_missing}")

    # Time range if timestamp exists
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"\nTime range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")
        except:
            print("\nCouldn't convert timestamp to datetime.")

    # Per-column statistics
    print("\n" + "=" * 80)
    print("DETAILED COLUMN STATISTICS")
    print("=" * 80)

    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"  Data type: {df[col].dtype}")

        # Missing values
        missing = df[col].isnull().sum()
        missing_pct = missing / len(df) * 100
        print(f"  Missing values: {missing} ({missing_pct:.2f}%)")

        # Range of values
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            print(f"  Range: {df[col].min()} to {df[col].max()}")
            print(f"  Mean: {df[col].mean()}")
            print(f"  Median: {df[col].median()}")
            print(f"  Std dev: {df[col].std()}")

            # Check for zero values
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"  Zero values: {zero_count} ({zero_count / len(df) * 100:.2f}%)")

            # Check for potential outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outlier_count > 0:
                print(f"  Potential outliers: {outlier_count} ({outlier_count / len(df) * 100:.2f}%)")

        elif pd.api.types.is_string_dtype(df[col].dtype):
            # For string columns
            unique_count = df[col].nunique()
            print(f"  Unique values: {unique_count}")
            if unique_count <= 10:  # Only show value counts for columns with few unique values
                print("  Value counts:")
                for val, count in df[col].value_counts().head(10).items():
                    print(f"    - {val}: {count}")
            else:
                print(
                    f"  Most common value: {df[col].value_counts().index[0]} ({df[col].value_counts().iloc[0]} occurrences)")

        elif pd.api.types.is_datetime64_dtype(df[col].dtype):
            # For datetime columns
            print(f"  Range: {df[col].min()} to {df[col].max()}")
            print(f"  Time span: {df[col].max() - df[col].min()}")


if __name__ == "__main__":
    # Define the file path
    file_path = '/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_us_historical_data/btcusdc_15m_historical.csv'
    # Check if the file exists before proceeding
    if os.path.exists(file_path):
        print(f"File: {os.path.basename(file_path)}")
        # Read the CSV file
        df = pd.read_csv(file_path)
    else:
        raise Exception(f"Error: The file {file_path} does not exist. Present working directory: {os.getcwd()}")
    print_df_stats(df)
