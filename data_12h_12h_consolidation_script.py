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
    Consolidates two 12h BTC historical datasets.

    The script:
    1. Aligns the starting time point of both 12h files by comparing timestamps
    2. Writes the corresponding records as a single record into a third csv file
    3. Proceeds row by row until either csv runs out of data
    4. Provides detailed analysis of input and output datasets
    """
    print("=" * 60)
    print("BTC HISTORICAL DATA CONSOLIDATION")
    print("=" * 60)
    # Define input and output file paths
    input_12h_file_1 = 'binance_futures_historical_data/btcusdt_12h_historical.csv'
    input_12h_file_2 = 'binance_futures_historical_data/btc_usdt_12h_features.csv'
    output_file = 'binance_futures_historical_data/btcusdt_12h_12h_consolidated.csv'

    print(f"Starting data consolidation process:")
    print(f"First 12h data source: {input_12h_file_1}")
    print(f"Second 12h data source: {input_12h_file_2}")
    print(f"Output destination: {output_file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the CSV files
    try:
        df_12h_1 = pd.read_csv(input_12h_file_1)
        df_12h_2 = pd.read_csv(input_12h_file_2)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Print detailed structure information about the first dataset
    print("\n=== FIRST DATASET STRUCTURE ===")
    print(f"Shape: {df_12h_1.shape} (rows, columns)")
    print(f"Columns: {', '.join(df_12h_1.columns)}")
    print("Data types:")
    for col, dtype in df_12h_1.dtypes.items():
        print(f"  - {col}: {dtype}")
    print("Sample data (first 3 rows):")
    print(df_12h_1.head(3))

    # Print detailed structure information about the second dataset
    print("\n=== SECOND DATASET STRUCTURE ===")
    print(f"Shape: {df_12h_2.shape} (rows, columns)")
    print(f"Columns: {', '.join(df_12h_2.columns)}")
    print("Data types:")
    for col, dtype in df_12h_2.dtypes.items():
        print(f"  - {col}: {dtype}")
    print("Sample data (first 3 rows):")
    print(df_12h_2.head(3))

    # Verify we have the necessary columns
    if 'timestamp' not in df_12h_1.columns:
        print("Error: 'timestamp' column not found in first 12h dataset")
        return

    # Check for timestamp column in second dataset (might be 'timestamp' or 'time')
    timestamp_col_2 = None
    if 'timestamp' in df_12h_2.columns:
        timestamp_col_2 = 'timestamp'
    elif 'time' in df_12h_2.columns:
        timestamp_col_2 = 'time'
        print("Note: Using 'time' column as timestamp in second dataset")

    if timestamp_col_2 is None:
        print("Error: Neither 'timestamp' nor 'time' column found in second 12h dataset")
        return

    print(f"Loaded first 12h dataset: {len(df_12h_1)} rows")
    print(f"Loaded second 12h dataset: {len(df_12h_2)} rows")

    # Convert both 12h timestamps to Unix format for comparison
    df_12h_1['unix_timestamp'] = pd.to_datetime(df_12h_1['timestamp']).apply(lambda x: int(x.timestamp()))

    # For second dataset, handle both 'timestamp' and 'time' columns
    if timestamp_col_2 == 'timestamp':
        # If it's a regular timestamp, convert it like the first dataset
        df_12h_2['unix_timestamp'] = pd.to_datetime(df_12h_2[timestamp_col_2]).apply(lambda x: int(x.timestamp()))
    else:  # timestamp_col_2 is 'time'
        # Check if 'time' is already in Unix timestamp format (integer)
        if df_12h_2[timestamp_col_2].dtype == 'int64' or df_12h_2[timestamp_col_2].dtype == 'int32':
            print("Note: 'time' column appears to be already in Unix timestamp format")
            df_12h_2['unix_timestamp'] = df_12h_2[timestamp_col_2]
        else:
            # Try to convert from datetime string
            try:
                df_12h_2['unix_timestamp'] = pd.to_datetime(df_12h_2[timestamp_col_2]).apply(
                    lambda x: int(x.timestamp()))
            except Exception as e:
                print(f"Warning: Error converting 'time' to Unix timestamp: {e}")
                print("Assuming 'time' is already in Unix timestamp format")
                df_12h_2['unix_timestamp'] = df_12h_2[timestamp_col_2]

    # Get the first timestamp from each file
    first_time_12h_1 = df_12h_1['unix_timestamp'].iloc[0]
    first_time_12h_2 = df_12h_2['unix_timestamp'].iloc[0]

    print(f"First dataset timestamp: {df_12h_1['timestamp'].iloc[0]} ({first_time_12h_1})")
    # Use the detected timestamp column name for the second dataset
    if 'timestamp' in df_12h_2.columns:
        print(f"Second dataset timestamp: {df_12h_2['timestamp'].iloc[0]} ({first_time_12h_2})")
    else:
        # For Unix timestamp, convert to readable format
        readable_time = datetime.fromtimestamp(df_12h_2[timestamp_col_2].iloc[0]).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Second dataset timestamp: {readable_time} ({first_time_12h_2})")

    # Print time range information for both datasets
    print("\n=== DATASET TIME RANGES ===")
    print(f"First dataset time range:")
    print(f"  - Start: {df_12h_1['timestamp'].iloc[0]} ({df_12h_1['unix_timestamp'].iloc[0]})")
    print(f"  - End: {df_12h_1['timestamp'].iloc[-1]} ({df_12h_1['unix_timestamp'].iloc[-1]})")
    print(
        f"  - Duration: {(df_12h_1['unix_timestamp'].iloc[-1] - df_12h_1['unix_timestamp'].iloc[0]) / 3600:.1f} hours")

    print(f"Second dataset time range:")
    # Handle different column names for timestamp
    second_dataset_time_col = timestamp_col_2

    # If time column is Unix timestamp, convert to readable format for display
    if df_12h_2[second_dataset_time_col].dtype == 'int64' or df_12h_2[second_dataset_time_col].dtype == 'int32':
        start_time = datetime.fromtimestamp(df_12h_2[second_dataset_time_col].iloc[0]).strftime('%Y-%m-%d %H:%M:%S')
        end_time = datetime.fromtimestamp(df_12h_2[second_dataset_time_col].iloc[-1]).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - Start: {start_time} ({df_12h_2['unix_timestamp'].iloc[0]})")
        print(f"  - End: {end_time} ({df_12h_2['unix_timestamp'].iloc[-1]})")
    else:
        print(f"  - Start: {df_12h_2[second_dataset_time_col].iloc[0]} ({df_12h_2['unix_timestamp'].iloc[0]})")
        print(f"  - End: {df_12h_2[second_dataset_time_col].iloc[-1]} ({df_12h_2['unix_timestamp'].iloc[-1]})")

    print(
        f"  - Duration: {(df_12h_2['unix_timestamp'].iloc[-1] - df_12h_2['unix_timestamp'].iloc[0]) / 3600:.1f} hours")

    # Determine which file starts later and use that as starting point
    start_idx_12h_1 = 0
    start_idx_12h_2 = 0

    if first_time_12h_1 > first_time_12h_2:
        # If first dataset starts later, find the corresponding point in second dataset
        # Search for matching or closest timestamp in dataset 2
        for i, row in df_12h_2.iterrows():
            if row['unix_timestamp'] >= first_time_12h_1:
                start_idx_12h_2 = i
                break

        print(f"First dataset starts later")
        print(f"Starting at first dataset index: {start_idx_12h_1}, second dataset index: {start_idx_12h_2}")
    else:
        # If second dataset starts later, find the corresponding point in first dataset
        for i, row in df_12h_1.iterrows():
            if row['unix_timestamp'] >= first_time_12h_2:
                start_idx_12h_1 = i
                break

        print(f"Second dataset starts later")
        print(f"Starting at first dataset index: {start_idx_12h_1}, second dataset index: {start_idx_12h_2}")

    # Verify our starting indices are valid
    if start_idx_12h_1 >= len(df_12h_1):
        print(f"Error: Calculated first dataset start index {start_idx_12h_1} exceeds available data")
        return

    if start_idx_12h_2 >= len(df_12h_2):
        print(f"Error: Calculated second dataset start index {start_idx_12h_2} exceeds available data")
        return

    # Show information about start records
    print("\n=== STARTING RECORD INFORMATION ===")
    print(f"First dataset starting record (index {start_idx_12h_1}):")
    if start_idx_12h_1 < len(df_12h_1):
        start_record_1 = df_12h_1.iloc[start_idx_12h_1]
        print(f"  - Timestamp: {start_record_1['timestamp']} ({start_record_1['unix_timestamp']})")
        # Display a few key columns if they exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in start_record_1:
                print(f"  - {col}: {start_record_1[col]}")

    print(f"Second dataset starting record (index {start_idx_12h_2}):")
    if start_idx_12h_2 < len(df_12h_2):
        start_record_2 = df_12h_2.iloc[start_idx_12h_2]
        # Handle different timestamp column names and formats
        if timestamp_col_2 == 'timestamp':
            print(f"  - Timestamp: {start_record_2[timestamp_col_2]} ({start_record_2['unix_timestamp']})")
        else:  # timestamp_col_2 is 'time'
            start_time_readable = datetime.fromtimestamp(start_record_2[timestamp_col_2]).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - Time: {start_time_readable} ({start_record_2['unix_timestamp']})")
        # Display a few key columns if they exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in start_record_2:
                print(f"  - {col}: {start_record_2[col]}")

    # Verify the first column name from the first 12h file
    first_column_name = df_12h_1.columns[0]
    print(f"First column in first 12h dataset: '{first_column_name}'")

    # Create the output file with all columns
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)

        # Create combined header for output
        header_12h_1 = list(df_12h_1.columns)
        header_12h_2 = list(df_12h_2.columns)

        # Remove the temporary unix_timestamp column from headers
        if 'unix_timestamp' in header_12h_1:
            header_12h_1.remove('unix_timestamp')
        if 'unix_timestamp' in header_12h_2:
            header_12h_2.remove('unix_timestamp')

        # Create the output header: all columns from first dataset, then unique columns from second dataset
        output_header = header_12h_1.copy()

        # Add columns from second dataset (excluding duplicates like timestamp)
        duplicate_columns = []
        unique_columns = []

        for col in header_12h_2:
            # Skip the timestamp column (whether it's 'timestamp' or 'time')
            if col == timestamp_col_2 or col == 'unix_timestamp':
                continue

            if col not in header_12h_1:  # Skip duplicated columns
                unique_columns.append(col)
                output_header.append(f'dataset2_{col}')
            else:  # Track duplicates
                duplicate_columns.append(col)

        print("\n=== COLUMN ANALYSIS ===")
        print(f"First dataset columns: {len(header_12h_1)}")
        print(f"Second dataset columns: {len(header_12h_2)}")
        print(f"Duplicate columns being skipped: {len(duplicate_columns)}")
        if duplicate_columns:
            print(f"  - Duplicates: {', '.join(duplicate_columns)}")
        print(f"Unique columns from second dataset being added: {len(unique_columns)}")
        if unique_columns:
            print(f"  - Unique: {', '.join(unique_columns)}")
        print(f"Total columns in output: {len(output_header) + 2}")  # +2 for timestamp columns

        # Add reference timestamp columns at the end
        output_header.append('dataset1_unix_timestamp')
        output_header.append('dataset2_unix_timestamp')

        # Write the header
        writer.writerow(output_header)

        # Verify that the first column in output matches the first column in the first 12h dataset
        if output_header[0] == first_column_name:
            print(f"✓ Output file first column '{output_header[0]}' matches first 12h dataset")
        else:
            print(
                f"! Warning: Output file first column '{output_header[0]}' does not match first 12h dataset '{first_column_name}'")
            print("  Continuing anyway...")

        # Initialize counters
        idx_12h_1 = start_idx_12h_1
        idx_12h_2 = start_idx_12h_2
        records_written = 0
        tolerance_seconds = 60  # Allow for small timestamp differences

        # Dictionary to store columns from second dataset
        second_dataset_columns = []

        # Map the column names to column objects
        for col in header_12h_2:
            if col != timestamp_col_2 and col != 'unix_timestamp' and col not in header_12h_1:
                second_dataset_columns.append(col)

        # Process records until we reach the end of either file
        while idx_12h_1 < len(df_12h_1) and idx_12h_2 < len(df_12h_2):
            # Get the current records
            row_12h_1 = df_12h_1.iloc[idx_12h_1]
            row_12h_2 = df_12h_2.iloc[idx_12h_2]

            time_diff = abs(row_12h_1['unix_timestamp'] - row_12h_2['unix_timestamp'])

            # If timestamps are close enough (within tolerance), merge the rows
            if time_diff <= tolerance_seconds:
                # Create the merged record
                merged_row = []

                # Add all columns from the first 12h record (except unix_timestamp)
                for col in header_12h_1:
                    if col != 'unix_timestamp':
                        merged_row.append(row_12h_1[col])

                # Add unique columns from the second 12h record
                for col in second_dataset_columns:
                    merged_row.append(row_12h_2[col])

                # Add the reference timestamps
                merged_row.append(row_12h_1['unix_timestamp'])
                merged_row.append(row_12h_2['unix_timestamp'])

                # Write the merged record
                writer.writerow(merged_row)
                records_written += 1

                # Move to the next records
                idx_12h_1 += 1
                idx_12h_2 += 1

            # If first dataset timestamp is earlier, advance first dataset index
            elif row_12h_1['unix_timestamp'] < row_12h_2['unix_timestamp']:
                idx_12h_1 += 1

            # If second dataset timestamp is earlier, advance second dataset index
            else:
                idx_12h_2 += 1

            # Print progress every 100 records
            if records_written % 100 == 0 and records_written > 0:
                print(f"Processed {records_written} merged records...")

    print(f"Consolidation complete! {records_written} merged records written to {output_file}")
    print(f"Final first dataset index: {idx_12h_1} of {len(df_12h_1)}")
    print(f"Final second dataset index: {idx_12h_2} of {len(df_12h_2)}")

    # Report any unused records
    if idx_12h_1 < len(df_12h_1):
        print(f"Note: {len(df_12h_1) - idx_12h_1} records from the first 12h file were not processed")

    if idx_12h_2 < len(df_12h_2):
        print(f"Note: {len(df_12h_2) - idx_12h_2} records from the second 12h file were not processed")

    # Show information about ending records
    print("\n=== ENDING RECORD INFORMATION ===")
    print(f"First dataset ending record processed (index {idx_12h_1 - 1}):")
    if idx_12h_1 > 0 and idx_12h_1 <= len(df_12h_1):
        end_record_1 = df_12h_1.iloc[idx_12h_1 - 1]
        print(f"  - Timestamp: {end_record_1['timestamp']} ({end_record_1['unix_timestamp']})")
        # Display a few key columns if they exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in end_record_1:
                print(f"  - {col}: {end_record_1[col]}")

    print(f"Second dataset ending record processed (index {idx_12h_2 - 1}):")
    if idx_12h_2 > 0 and idx_12h_2 <= len(df_12h_2):
        end_record_2 = df_12h_2.iloc[idx_12h_2 - 1]
        # Handle different timestamp column names and formats
        if timestamp_col_2 == 'timestamp':
            print(f"  - Timestamp: {end_record_2[timestamp_col_2]} ({end_record_2['unix_timestamp']})")
        else:  # timestamp_col_2 is 'time'
            end_time_readable = datetime.fromtimestamp(end_record_2[timestamp_col_2]).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - Time: {end_time_readable} ({end_record_2['unix_timestamp']})")
        # Display a few key columns if they exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in end_record_2:
                print(f"  - {col}: {end_record_2[col]}")

    # Load and analyze the output file to provide summary information
    try:
        df_output = pd.read_csv(output_file)

        print("\n=== OUTPUT DATASET SUMMARY ===")
        print(f"Shape: {df_output.shape} (rows, columns)")
        print(f"Columns: {len(df_output.columns)}")
        print("Time range:")
        if 'timestamp' in df_output.columns:
            timestamp_col = 'timestamp'
        elif 'dataset1_unix_timestamp' in df_output.columns:
            timestamp_col = 'dataset1_unix_timestamp'
            # Convert unix timestamp to readable format for display
            first_ts = datetime.fromtimestamp(df_output[timestamp_col].iloc[0])
            last_ts = datetime.fromtimestamp(df_output[timestamp_col].iloc[-1])
            print(f"  - Start: {first_ts} ({df_output[timestamp_col].iloc[0]})")
            print(f"  - End: {last_ts} ({df_output[timestamp_col].iloc[-1]})")
            print(
                f"  - Duration: {(df_output[timestamp_col].iloc[-1] - df_output[timestamp_col].iloc[0]) / 3600:.1f} hours")

        print("Column categories:")
        first_dataset_cols = [col for col in df_output.columns if not col.startswith(
            'dataset2_') and col != 'dataset1_unix_timestamp' and col != 'dataset2_unix_timestamp']
        second_dataset_cols = [col for col in df_output.columns if col.startswith('dataset2_')]
        reference_cols = [col for col in df_output.columns if 'unix_timestamp' in col]

        print(f"  - First dataset columns: {len(first_dataset_cols)}")
        print(f"  - Second dataset columns: {len(second_dataset_cols)}")
        print(f"  - Reference timestamp columns: {len(reference_cols)}")

        # Sample data
        print("\nSample data (first 3 rows):")
        # Format sample data to show first few columns and timestamps
        sample_cols = first_dataset_cols[:5] + ['dataset1_unix_timestamp', 'dataset2_unix_timestamp']
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