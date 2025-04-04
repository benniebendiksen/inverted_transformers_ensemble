import pandas as pd
import numpy as np
from datetime import datetime
import os
import csv


def convert_timestamp(timestamp_str):
    """Convert human-readable timestamp to Unix timestamp"""
    try:
        return int(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp())
    except ValueError:
        print(f"Warning: Could not parse timestamp {timestamp_str}")
        return 0


def main():
    """
    Consolidates 12h BTC historical data with 4h features data.

    The script:
    1. Aligns the starting time point of the 12h file with a timestamp of the 4h file
    2. Writes the corresponding records as a single record into a third csv file
    3. Moves the 12h file pointer one record while moving the 4h file pointer three records
       to maintain alignment (12 hours = 3 × 4 hours)
    4. Proceeds until either csv runs out of data
    5. Provides detailed analysis of input and output datasets
    """
    print("=" * 60)
    print("BTC HISTORICAL DATA CONSOLIDATION (12h + 4h)")
    print("=" * 60)

    # Define input and output file paths
    input_12h_file = 'binance_futures_historical_data/btcusdt_12h_historical.csv'
    input_4h_file = 'binance_futures_historical_data/btc_usdt_4h_features.csv'
    output_file = 'binance_futures_historical_data/btcusdt_12h_4h_consolidated.csv'

    print(f"Starting data consolidation process:")
    print(f"12h data source: {input_12h_file}")
    print(f"4h data source: {input_4h_file}")
    print(f"Output destination: {output_file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the CSV files
    try:
        df_12h = pd.read_csv(input_12h_file)
        df_4h = pd.read_csv(input_4h_file)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Print detailed structure information about the 12h dataset
    print("\n=== 12H DATASET STRUCTURE ===")
    print(f"Shape: {df_12h.shape} (rows, columns)")
    print(f"Columns: {', '.join(df_12h.columns)}")
    print("Data types:")
    for col, dtype in df_12h.dtypes.items():
        print(f"  - {col}: {dtype}")
    print("Sample data (first 3 rows):")
    print(df_12h.head(3))

    # Print detailed structure information about the 4h dataset
    print("\n=== 4H DATASET STRUCTURE ===")
    print(f"Shape: {df_4h.shape} (rows, columns)")
    print(f"Columns: {', '.join(df_4h.columns)}")
    print("Data types:")
    for col, dtype in df_4h.dtypes.items():
        print(f"  - {col}: {dtype}")
    print("Sample data (first 3 rows):")
    print(df_4h.head(3))

    # Verify we have the necessary columns
    if 'timestamp' not in df_12h.columns:
        print("Error: 'timestamp' column not found in 12h data")
        return

    if 'time' not in df_4h.columns:
        print("Error: 'time' column not found in 4h data")
        return

    print(f"Loaded 12h data: {len(df_12h)} rows")
    print(f"Loaded 4h data: {len(df_4h)} rows")

    # Convert 12h timestamps to Unix format for comparison
    df_12h['unix_timestamp'] = pd.to_datetime(df_12h['timestamp']).apply(lambda x: int(x.timestamp()))

    # Check if 'time' is already in Unix timestamp format (integer)
    if df_4h['time'].dtype == 'int64' or df_4h['time'].dtype == 'int32':
        print("Note: 'time' column in 4h dataset appears to be already in Unix timestamp format")
    else:
        # Try to convert from datetime string
        try:
            df_4h['unix_time'] = pd.to_datetime(df_4h['time']).apply(lambda x: int(x.timestamp()))
            print("Converted 4h time column to unix timestamp")
            df_4h['time'] = df_4h['unix_time']
            df_4h.drop('unix_time', axis=1, inplace=True)
        except Exception as e:
            print(f"Warning: Could not convert 4h time to datetime: {e}")
            print("Assuming time is already in Unix timestamp format")

    # Get the first timestamp from each file
    first_time_12h = df_12h['unix_timestamp'].iloc[0]
    first_time_4h = df_4h['time'].iloc[0]

    print(f"First 12h timestamp: {df_12h['timestamp'].iloc[0]} ({first_time_12h})")
    readable_4h_time = datetime.fromtimestamp(first_time_4h).strftime('%Y-%m-%d %H:%M:%S')
    print(f"First 4h timestamp: {readable_4h_time} ({first_time_4h})")

    # Print time range information for both datasets
    print("\n=== DATASET TIME RANGES ===")
    print(f"12h dataset time range:")
    print(f"  - Start: {df_12h['timestamp'].iloc[0]} ({df_12h['unix_timestamp'].iloc[0]})")
    print(f"  - End: {df_12h['timestamp'].iloc[-1]} ({df_12h['unix_timestamp'].iloc[-1]})")
    print(f"  - Duration: {(df_12h['unix_timestamp'].iloc[-1] - df_12h['unix_timestamp'].iloc[0]) / 3600:.1f} hours")

    print(f"4h dataset time range:")
    start_time_4h = datetime.fromtimestamp(df_4h['time'].iloc[0]).strftime('%Y-%m-%d %H:%M:%S')
    end_time_4h = datetime.fromtimestamp(df_4h['time'].iloc[-1]).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  - Start: {start_time_4h} ({df_4h['time'].iloc[0]})")
    print(f"  - End: {end_time_4h} ({df_4h['time'].iloc[-1]})")
    print(f"  - Duration: {(df_4h['time'].iloc[-1] - df_4h['time'].iloc[0]) / 3600:.1f} hours")

    # Determine which file starts later and use that as starting point
    start_idx_12h = 0
    start_idx_4h = 0

    if first_time_12h > first_time_4h:
        # If 12h data starts later, find the corresponding point in 4h data
        time_diff = first_time_12h - first_time_4h
        hours_diff = time_diff / 3600
        records_diff = int(hours_diff / 4)  # How many 4h records to skip

        start_idx_4h = records_diff
        # Ensure we're at a proper starting point (divisible by 3)
        start_idx_4h = (start_idx_4h // 3) * 3

        print(f"12h data starts later by approximately {hours_diff:.1f} hours ({records_diff} 4h records)")
        print(f"Starting at 12h index: {start_idx_12h}, 4h index: {start_idx_4h}")
    else:
        # If 4h data starts later, find the corresponding point in 12h data
        time_diff = first_time_4h - first_time_12h
        hours_diff = time_diff / 3600
        records_diff = int(hours_diff / 12)  # How many 12h records to skip

        start_idx_12h = records_diff

        print(f"4h data starts later by approximately {hours_diff:.1f} hours ({records_diff} 12h records)")
        print(f"Starting at 12h index: {start_idx_12h}, 4h index: {start_idx_4h}")

    # Verify our starting indices are valid
    if start_idx_12h >= len(df_12h):
        print(f"Error: Calculated 12h start index {start_idx_12h} exceeds available data")
        return

    if start_idx_4h >= len(df_4h):
        print(f"Error: Calculated 4h start index {start_idx_4h} exceeds available data")
        return

    # Verify the first column name from the 12h file
    first_column_name = df_12h.columns[0]
    print(f"First column in 12h data: '{first_column_name}'")

    # Create the output file with all columns
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)

        # Create combined header for output
        header_12h = list(df_12h.columns)
        header_4h = list(df_4h.columns)

        # Remove the temporary unix_timestamp column from header
        if 'unix_timestamp' in header_12h:
            header_12h.remove('unix_timestamp')

        # Create the output header: all 12h columns, then 4h features for each batch
        output_header = header_12h.copy()

        # Track unique and duplicated columns for reporting
        duplicated_columns = []
        feature_columns = []

        # Add the 4h data columns for each of the three 4-hour periods
        for batch in range(1, 4):
            for col in header_4h:
                if col not in ['time', 'open', 'high', 'low', 'close']:  # Skip duplicated price data columns
                    if col not in feature_columns:
                        feature_columns.append(col)
                    output_header.append(f'4h_batch{batch}_{col}')
                elif col not in ['time'] and col not in duplicated_columns:
                    duplicated_columns.append(col)

        # Print column analysis information
        print("\n=== COLUMN ANALYSIS ===")
        print(f"12h dataset columns: {len(header_12h)}")
        print(f"4h dataset columns: {len(header_4h)}")
        print(f"Duplicate price columns being skipped: {len(duplicated_columns)}")
        if duplicated_columns:
            print(f"  - Duplicates: {', '.join(duplicated_columns)}")
        print(f"4h feature columns being added (× 3 batches): {len(feature_columns)}")
        if feature_columns:
            print(f"  - Features: {', '.join(feature_columns)}")
        print(f"Total columns in output: {len(output_header) + 4}")  # +4 for timestamp columns

        # Add reference timestamp columns (at the end, to preserve the original 12h first column)
        output_header.append('12h_unix_timestamp')
        for i in range(1, 4):
            output_header.append(f'4h_batch{i}_unix_timestamp')

        # Write the header
        writer.writerow(output_header)

        # Verify that the first column in output matches the first column in 12h data
        if output_header[0] == first_column_name:
            print(f"✓ Output file first column '{output_header[0]}' matches 12h data")
        else:
            print(
                f"! Warning: Output file first column '{output_header[0]}' does not match 12h data '{first_column_name}'")
            print("  Continuing anyway...")

        # Initialize counters
        idx_12h = start_idx_12h
        idx_4h = start_idx_4h
        records_written = 0

        # Process records until we reach the end of either file
        while idx_12h < len(df_12h) and idx_4h + 2 < len(df_4h):
            # Get the current 12h record
            row_12h = df_12h.iloc[idx_12h]

            # Get the three consecutive 4h records
            row_4h_1 = df_4h.iloc[idx_4h]
            row_4h_2 = df_4h.iloc[idx_4h + 1]
            row_4h_3 = df_4h.iloc[idx_4h + 2]

            # Create the merged record
            merged_row = []

            # Add all columns from the 12h record (except unix_timestamp)
            # This ensures the first column of the output file matches the 12h dataset exactly
            for col in header_12h:
                if col != 'unix_timestamp':
                    merged_row.append(row_12h[col])

            # Add the non-OHLC columns from each 4h batch
            for row_4h in [row_4h_1, row_4h_2, row_4h_3]:
                for col in header_4h:
                    if col not in ['time', 'open', 'high', 'low', 'close']:
                        merged_row.append(row_4h[col])

            # Add the reference timestamps
            merged_row.append(row_12h['unix_timestamp'])
            merged_row.append(row_4h_1['time'])
            merged_row.append(row_4h_2['time'])
            merged_row.append(row_4h_3['time'])

            # Write the merged record
            writer.writerow(merged_row)
            records_written += 1

            # Move to the next records: 1 record in 12h data, 3 records in 4h data
            idx_12h += 1
            idx_4h += 3

            # Print progress every 100 records
            if records_written % 100 == 0 and records_written > 0:
                print(f"Processed {records_written} merged records...")

    print(f"Consolidation complete! {records_written} merged records written to {output_file}")
    print(f"Final 12h index: {idx_12h} of {len(df_12h)}")
    print(f"Final 4h index: {idx_4h} of {len(df_4h)}")

    # Report any unused records
    if idx_12h < len(df_12h):
        print(f"Note: {len(df_12h) - idx_12h} records from the 12h file were not processed")

    if idx_4h < len(df_4h):
        print(f"Note: {len(df_4h) - idx_4h} records from the 4h file were not processed")

    # Load and analyze the output file to provide summary information
    try:
        df_output = pd.read_csv(output_file)

        print("\n=== OUTPUT DATASET SUMMARY ===")
        print(f"Shape: {df_output.shape} (rows, columns)")
        print(f"Columns: {len(df_output.columns)}")
        print("Time range:")
        if 'timestamp' in df_output.columns:
            timestamp_col = 'timestamp'
            # Display timestamp information if timestamp column exists
            print(f"  - Start: {df_output[timestamp_col].iloc[0]}")
            print(f"  - End: {df_output[timestamp_col].iloc[-1]}")
        elif '12h_unix_timestamp' in df_output.columns:
            timestamp_col = '12h_unix_timestamp'
            # Convert unix timestamp to readable format for display
            first_ts = datetime.fromtimestamp(df_output[timestamp_col].iloc[0])
            last_ts = datetime.fromtimestamp(df_output[timestamp_col].iloc[-1])
            print(f"  - Start: {first_ts} ({df_output[timestamp_col].iloc[0]})")
            print(f"  - End: {last_ts} ({df_output[timestamp_col].iloc[-1]})")
            print(
                f"  - Duration: {(df_output[timestamp_col].iloc[-1] - df_output[timestamp_col].iloc[0]) / 3600:.1f} hours")

        print("Column categories:")
        # Categorize columns by their source
        base_12h_cols = [col for col in df_output.columns
                         if not col.startswith('4h_batch') and col != '12h_unix_timestamp']
        feature_4h_cols = [col for col in df_output.columns if
                           col.startswith('4h_batch') and 'unix_timestamp' not in col]
        reference_cols = [col for col in df_output.columns if 'unix_timestamp' in col]

        print(f"  - Base 12h columns: {len(base_12h_cols)}")
        print(f"  - 4h feature columns: {len(feature_4h_cols)}")
        print(f"  - Reference timestamp columns: {len(reference_cols)}")

        # Sample data
        print("\nSample data (first 3 rows):")
        # Format sample data to show first few columns and timestamps
        sample_cols = base_12h_cols[:5] + ['12h_unix_timestamp', '4h_batch1_unix_timestamp']
        if len(sample_cols) < len(df_output.columns):
            print(df_output[sample_cols].head(3))
            print(f"[...and {len(df_output.columns) - len(sample_cols)} more columns]")
        else:
            print(df_output.head(3))

        # Data quality check
        null_counts = df_output.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if not columns_with_nulls.empty:
            print("\nColumns with NULL values:")
            for col, count in columns_with_nulls.items():
                print(f"  - {col}: {count} NULLs ({count / len(df_output) * 100:.2f}%)")
        else:
            print("\nNo NULL values found in the output dataset.")

    except Exception as e:
        print(f"\nError analyzing output file: {e}")


if __name__ == "__main__":
    main()