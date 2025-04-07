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
    Consolidates BTC historical data from three timeframes: 4h, 12h, and 1d.

    The script:
    1. Uses 12h data as the reference timeframe
    2. For each 12h record, aligns with 3 consecutive 4h records (12h = 3 × 4h)
    3. For each pair of consecutive 12h records, aligns with 1 daily record (1d = 2 × 12h)
    4. Provides detailed analysis of input and output datasets

    Alignment:
    - Each daily record is repeated for 2 consecutive 12h records
    - Each 12h record contains features from 3 consecutive 4h periods
    """
    print("=" * 80)
    print("BTC HISTORICAL DATA CONSOLIDATION (4h + 12h + 1d)")
    print("=" * 80)

    # Define input and output file paths
    input_12h_file = 'binance_futures_historical_data/btcusd_12h_python_processed_bitsap.csv'
    input_4h_file = 'binance_futures_historical_data/btcusd_4h_features_bitsap_2.csv'
    input_1d_file = 'binance_futures_historical_data/btcusd_1d_features_bitsap.csv'
    output_file = 'binance_futures_historical_data/btcusd_12h_4h_1d_bitsap.csv'

    print(f"Starting data consolidation process:")
    print(f"12h data source: {input_12h_file}")
    print(f"4h data source: {input_4h_file}")
    print(f"1d data source: {input_1d_file}")
    print(f"Output destination: {output_file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the CSV files
    try:
        df_12h = pd.read_csv(input_12h_file)
        df_4h = pd.read_csv(input_4h_file)
        df_1d = pd.read_csv(input_1d_file)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Print information about each dataset
    print("\n=== DATASET INFORMATION ===")
    print(f"12h dataset: {len(df_12h)} rows, {len(df_12h.columns)} columns")
    print(f"4h dataset: {len(df_4h)} rows, {len(df_4h.columns)} columns")
    print(f"1d dataset: {len(df_1d)} rows, {len(df_1d.columns)} columns")

    # Verify we have the necessary timestamp columns
    if 'timestamp' not in df_12h.columns:
        print("Error: 'timestamp' column not found in 12h data")
        return

    if 'time' not in df_4h.columns:
        print("Error: 'time' column not found in 4h data")
        return

    if 'time' not in df_1d.columns:
        print("Error: 'time' column not found in 1d data")
        return

    # Convert 12h timestamps to Unix format for comparison
    if pd.api.types.is_integer_dtype(df_12h['timestamp']):
        df_12h['unix_timestamp'] = df_12h['timestamp']
        print("Timestamp column appears to be already in Unix format, using directly")
    else:
        df_12h['unix_timestamp'] = pd.to_datetime(df_12h['timestamp']).apply(lambda x: int(x.timestamp()))
        print("Converted timestamp column to Unix format")

    # Check 4h time format and convert if needed
    if df_4h['time'].dtype != 'int64' and df_4h['time'].dtype != 'int32':
        try:
            df_4h['time'] = pd.to_datetime(df_4h['time']).apply(lambda x: int(x.timestamp()))
            print("Converted 4h time column to unix timestamp")
        except Exception as e:
            print(f"Warning: Could not convert 4h time to timestamp: {e}")
            return

    # Check 1d time format and convert if needed
    if df_1d['time'].dtype != 'int64' and df_1d['time'].dtype != 'int32':
        try:
            df_1d['time'] = pd.to_datetime(df_1d['time']).apply(lambda x: int(x.timestamp()))
            print("Converted 1d time column to unix timestamp")
        except Exception as e:
            print(f"Warning: Could not convert 1d time to timestamp: {e}")
            return

    # Display time ranges for each dataset
    print("\n=== DATASET TIME RANGES ===")

    # 12h dataset
    print(f"12h dataset time range:")
    print(f"  - Start: {df_12h['timestamp'].iloc[0]} ({df_12h['unix_timestamp'].iloc[0]})")
    print(f"  - End: {df_12h['timestamp'].iloc[-1]} ({df_12h['unix_timestamp'].iloc[-1]})")
    print(
        f"  - Duration: {(df_12h['unix_timestamp'].iloc[-1] - df_12h['unix_timestamp'].iloc[0]) / (24 * 3600):.1f} days")

    # 4h dataset
    print(f"4h dataset time range:")
    start_time_4h = datetime.fromtimestamp(df_4h['time'].iloc[0]).strftime('%Y-%m-%d %H:%M:%S')
    end_time_4h = datetime.fromtimestamp(df_4h['time'].iloc[-1]).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  - Start: {start_time_4h} ({df_4h['time'].iloc[0]})")
    print(f"  - End: {end_time_4h} ({df_4h['time'].iloc[-1]})")
    print(f"  - Duration: {(df_4h['time'].iloc[-1] - df_4h['time'].iloc[0]) / (24 * 3600):.1f} days")

    # 1d dataset
    print(f"1d dataset time range:")
    start_time_1d = datetime.fromtimestamp(df_1d['time'].iloc[0]).strftime('%Y-%m-%d %H:%M:%S')
    end_time_1d = datetime.fromtimestamp(df_1d['time'].iloc[-1]).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  - Start: {start_time_1d} ({df_1d['time'].iloc[0]})")
    print(f"  - End: {end_time_1d} ({df_1d['time'].iloc[-1]})")
    print(f"  - Duration: {(df_1d['time'].iloc[-1] - df_1d['time'].iloc[0]) / (24 * 3600):.1f} days")

    # Determine the starting indices for all datasets based on timestamp alignment
    # We'll use the latest start time among all datasets as our reference
    start_time = max(
        df_12h['unix_timestamp'].iloc[0],
        df_4h['time'].iloc[0],
        df_1d['time'].iloc[0]
    )

    # Convert start time to human-readable format for display
    start_time_readable = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\nUsing common start time: {start_time_readable} ({start_time})")

    # Find starting indices for each dataset
    start_idx_12h = 0
    start_idx_4h = 0
    start_idx_1d = 0

    # Find starting index for 12h data
    if df_12h['unix_timestamp'].iloc[0] < start_time:
        # Find the first 12h record on or after start_time
        for idx, time in enumerate(df_12h['unix_timestamp']):
            if time >= start_time:
                start_idx_12h = idx
                break

    # Find starting index for 4h data
    if df_4h['time'].iloc[0] < start_time:
        # Find the first 4h record on or after start_time
        for idx, time in enumerate(df_4h['time']):
            if time >= start_time:
                # Ensure we're at a proper starting point (divisible by 3 for 4h data)
                start_idx_4h = (idx // 3) * 3
                break

    # Find starting index for 1d data
    if df_1d['time'].iloc[0] < start_time:
        # Find the first 1d record on or after start_time
        for idx, time in enumerate(df_1d['time']):
            if time >= start_time:
                start_idx_1d = idx
                break

    print(f"Starting indices - 12h: {start_idx_12h}, 4h: {start_idx_4h}, 1d: {start_idx_1d}")

    # Verify our starting indices are valid
    if start_idx_12h >= len(df_12h):
        print(f"Error: Calculated 12h start index {start_idx_12h} exceeds available data")
        return

    if start_idx_4h + 2 >= len(df_4h):
        print(f"Error: Calculated 4h start index {start_idx_4h} leaves insufficient records")
        return

    if start_idx_1d >= len(df_1d):
        print(f"Error: Calculated 1d start index {start_idx_1d} exceeds available data")
        return

    # Create combined header for output
    header_12h = list(df_12h.columns)
    header_4h = list(df_4h.columns)
    header_1d = list(df_1d.columns)

    # Remove temporary timestamp column from 12h header
    if 'unix_timestamp' in header_12h:
        header_12h.remove('unix_timestamp')

    # Create the output header
    output_header = header_12h.copy()

    # Add 4h features for each of the three 4-hour periods
    for batch in range(1, 4):
        for col in header_4h:
            if col not in ['time', 'open', 'high', 'low', 'close']:  # Skip duplicated price data columns
                output_header.append(f'4h_batch{batch}_{col}')

    # Add 1d features
    for col in header_1d:
        if col not in ['time', 'open', 'high', 'low', 'close']:  # Skip duplicated price data columns
            output_header.append(f'1d_{col}')

    # Add reference timestamp columns
    output_header.append('12h_unix_timestamp')
    for i in range(1, 4):
        output_header.append(f'4h_batch{i}_unix_timestamp')
    output_header.append('1d_unix_timestamp')

    print(f"\nOutput will contain {len(output_header)} columns")

    # Create the output file
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)

        # Write the header
        writer.writerow(output_header)

        # Initialize counters and trackers
        idx_12h = start_idx_12h
        idx_4h = start_idx_4h
        idx_1d = start_idx_1d
        records_written = 0

        # Track the current 1d row and update counter (for every 2 12h records)
        current_1d_row = df_1d.iloc[idx_1d] if idx_1d < len(df_1d) else None
        current_1d_counter = 0

        # Process records until we reach the end of any required dataset
        while (idx_12h < len(df_12h) and
               idx_4h + 2 < len(df_4h) and
               current_1d_row is not None):

            # Get the current 12h record
            row_12h = df_12h.iloc[idx_12h]

            # Get the three consecutive 4h records
            row_4h_1 = df_4h.iloc[idx_4h]
            row_4h_2 = df_4h.iloc[idx_4h + 1]
            row_4h_3 = df_4h.iloc[idx_4h + 2]

            # Create the merged record
            merged_row = []

            # Add all columns from the 12h record (except unix_timestamp)
            for col in header_12h:
                if col != 'unix_timestamp':
                    merged_row.append(row_12h[col])

            # Add the non-OHLC columns from each 4h batch
            for row_4h in [row_4h_1, row_4h_2, row_4h_3]:
                for col in header_4h:
                    if col not in ['time', 'open', 'high', 'low', 'close']:
                        merged_row.append(row_4h[col])

            # Add the non-OHLC columns from the current 1d record
            for col in header_1d:
                if col not in ['time', 'open', 'high', 'low', 'close']:
                    merged_row.append(current_1d_row[col])

            # Add the reference timestamps
            merged_row.append(row_12h['unix_timestamp'])
            merged_row.append(row_4h_1['time'])
            merged_row.append(row_4h_2['time'])
            merged_row.append(row_4h_3['time'])
            merged_row.append(current_1d_row['time'])

            # Write the merged record
            writer.writerow(merged_row)
            records_written += 1

            # Move to the next 12h record
            idx_12h += 1

            # Move to the next three 4h records
            idx_4h += 3

            # Update 1d record counter and potentially move to next 1d record
            current_1d_counter += 1
            if current_1d_counter >= 2:  # Move to next 1d record after every 2 12h records
                current_1d_counter = 0
                idx_1d += 1
                if idx_1d < len(df_1d):
                    current_1d_row = df_1d.iloc[idx_1d]
                else:
                    current_1d_row = None

            # Print progress every 100 records
            if records_written % 100 == 0 and records_written > 0:
                print(f"Processed {records_written} merged records...")

    print(f"\n=== CONSOLIDATION COMPLETE ===")
    print(f"Total records written: {records_written}")
    print(f"Final indices - 12h: {idx_12h}/{len(df_12h)}, 4h: {idx_4h}/{len(df_4h)}, 1d: {idx_1d}/{len(df_1d)}")

    # Report any unused records
    if idx_12h < len(df_12h):
        print(f"Note: {len(df_12h) - idx_12h} records from the 12h file were not processed")

    if idx_4h < len(df_4h):
        print(f"Note: {len(df_4h) - idx_4h} records from the 4h file were not processed")

    if idx_1d < len(df_1d):
        print(f"Note: {len(df_1d) - idx_1d} records from the 1d file were not processed")

    # Validate the output file
    try:
        df_output = pd.read_csv(output_file)

        print("\n=== OUTPUT VALIDATION ===")
        print(f"Output file contains {len(df_output)} rows and {len(df_output.columns)} columns")

        # Check for missing values
        null_counts = df_output.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            print("\nColumns with NULL values:")
            for col, count in columns_with_nulls.items():
                print(f"  - {col}: {count} NULLs ({count / len(df_output) * 100:.2f}%)")
        else:
            print("\nNo NULL values found in the output dataset.")

        print("\nConsolidation successfully completed!")

    except Exception as e:
        print(f"\nError validating output file: {e}")


if __name__ == "__main__":
    main()