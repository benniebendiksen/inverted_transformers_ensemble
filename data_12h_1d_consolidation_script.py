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
    Consolidates 12h BTC historical data with 1d features data.

    The script:
    1. Aligns the starting time point of the 12h file with a timestamp of the 1d file
    2. Writes corresponding records as a single record into a third csv file
    3. Moves the 12h file pointer two records while moving the 1d file pointer one record
       to maintain alignment (24 hours = 2 × 12 hours)
    4. Proceeds until either csv runs out of data
    """
    # Define input and output file paths
    input_12h_file = 'binance_futures_historical_data/btcusdt_12h_historical.csv'
    input_1d_file = 'binance_futures_historical_data/btc_usdt_1d_features.csv'
    output_file = 'binance_futures_historical_data/btcusdt_consolidated_1d.csv'

    print(f"Starting data consolidation process:")
    print(f"12h data source: {input_12h_file}")
    print(f"1d data source: {input_1d_file}")
    print(f"Output destination: {output_file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the CSV files
    try:
        df_12h = pd.read_csv(input_12h_file)
        df_1d = pd.read_csv(input_1d_file)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Verify we have the necessary columns
    if 'timestamp' not in df_12h.columns:
        print("Error: 'timestamp' column not found in 12h data")
        return

    if 'time' not in df_1d.columns:
        print("Error: 'time' column not found in 1d data")
        return

    print(f"Loaded 12h data: {len(df_12h)} rows")
    print(f"Loaded 1d data: {len(df_1d)} rows")

    # Convert 12h timestamps to Unix format for comparison
    df_12h['unix_timestamp'] = pd.to_datetime(df_12h['timestamp']).apply(lambda x: int(x.timestamp()))

    # Get the first timestamp from each file
    first_time_12h = df_12h['unix_timestamp'].iloc[0]
    first_time_1d = df_1d['time'].iloc[0]

    print(f"First 12h timestamp: {df_12h['timestamp'].iloc[0]} ({first_time_12h})")
    print(f"First 1d timestamp: {datetime.fromtimestamp(first_time_1d)} ({first_time_1d})")

    # Determine which file starts later and use that as starting point
    start_idx_12h = 0
    start_idx_1d = 0

    if first_time_12h > first_time_1d:
        # If 12h data starts later, find the corresponding point in 1d data
        time_diff = first_time_12h - first_time_1d
        days_diff = time_diff / (24 * 3600)  # Convert seconds to days
        records_diff = int(days_diff)  # How many 1d records to skip

        start_idx_1d = records_diff

        print(f"12h data starts later by approximately {days_diff:.1f} days ({records_diff} 1d records)")
        print(f"Starting at 12h index: {start_idx_12h}, 1d index: {start_idx_1d}")
    else:
        # If 1d data starts later, find the corresponding point in 12h data
        time_diff = first_time_1d - first_time_12h
        hours_diff = time_diff / 3600  # Convert seconds to hours
        records_diff = int(hours_diff / 12)  # How many 12h records to skip

        # Ensure we start at an even index for 12h (to maintain proper 2:1 alignment)
        start_idx_12h = records_diff + (records_diff % 2)

        print(f"1d data starts later by approximately {hours_diff:.1f} hours ({records_diff} 12h records)")
        print(f"Starting at 12h index: {start_idx_12h}, 1d index: {start_idx_1d}")

    # Verify our starting indices are valid
    if start_idx_12h >= len(df_12h):
        print(f"Error: Calculated 12h start index {start_idx_12h} exceeds available data")
        return

    if start_idx_1d >= len(df_1d):
        print(f"Error: Calculated 1d start index {start_idx_1d} exceeds available data")
        return

    # Create the output file with all columns
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)

        # Create combined header for output
        header_12h = list(df_12h.columns)
        header_1d = list(df_1d.columns)

        # Remove the temporary unix_timestamp column from header
        if 'unix_timestamp' in header_12h:
            header_12h.remove('unix_timestamp')

        # Create the output header: all 12h columns, then 1d features
        output_header = header_12h.copy()

        # Add the 1d data columns (excluding duplicated OHLC and time columns)
        for col in header_1d:
            if col not in ['time', 'open', 'high', 'low', 'close']:  # Skip duplicated price data columns
                output_header.append(f'1d_{col}')

        # Add reference timestamp columns
        output_header.append('12h_unix_timestamp')
        output_header.append('1d_unix_timestamp')

        # Write the header
        writer.writerow(output_header)

        # Initialize counters
        idx_12h = start_idx_12h
        idx_1d = start_idx_1d
        records_written = 0

        # Process records until we reach the end of either file
        # For each 1d record, we'll use 2 consecutive 12h records
        while idx_12h + 1 < len(df_12h) and idx_1d < len(df_1d):
            # Get the 1d record
            row_1d = df_1d.iloc[idx_1d]

            # Process two consecutive 12h records with the same 1d record
            for offset in range(2):
                if idx_12h + offset >= len(df_12h):
                    print(f"Reached end of 12h data at index {idx_12h + offset}")
                    break

                # Get the current 12h record
                row_12h = df_12h.iloc[idx_12h + offset]

                # Create the merged record
                merged_row = []

                # Add all columns from the 12h record (except unix_timestamp)
                for col in header_12h:
                    if col != 'unix_timestamp':
                        merged_row.append(row_12h[col])

                # Add the non-OHLC columns from the 1d record
                for col in header_1d:
                    if col not in ['time', 'open', 'high', 'low', 'close']:
                        merged_row.append(row_1d[col])

                # Add the reference timestamps
                merged_row.append(row_12h['unix_timestamp'])
                merged_row.append(row_1d['time'])

                # Write the merged record
                writer.writerow(merged_row)
                records_written += 1

            # Move to the next records: 2 records in 12h data, 1 record in 1d data
            idx_12h += 2
            idx_1d += 1

            # Print progress every 100 records
            if records_written % 100 == 0:
                print(f"Processed {records_written} merged records...")

    print(f"Consolidation complete! {records_written} merged records written to {output_file}")
    print(f"Final 12h index: {idx_12h} of {len(df_12h)}")
    print(f"Final 1d index: {idx_1d} of {len(df_1d)}")

    # Report any unused records
    if idx_12h < len(df_12h):
        print(f"Note: {len(df_12h) - idx_12h} records from the 12h file were not processed")

    if idx_1d < len(df_1d):
        print(f"Note: {len(df_1d) - idx_1d} records from the 1d file were not processed")


if __name__ == "__main__":
    main()