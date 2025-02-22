from src.BinanceHistoricalDataFetcher import BinanceHistoricalDataFetcher
import pandas as pd
from datetime import datetime, timedelta
import time


def fetch_and_validate_data(symbol: str, interval: str, min_days: int = 365):
    """
    Fetch and validate historical data with proper error handling

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe interval (e.g., '1h')
        min_days: Minimum number of days of historical data required
    """

    fetcher = BinanceHistoricalDataFetcher(
        symbol=symbol,
        interval=interval,
        data_dir="binance_futures_historical_data"
    )

    # First try to load existing data
    existing_data = fetcher.load_data()

    if existing_data is not None:
        print(f"Found existing data with {len(existing_data)} records")

        # Check if data is recent enough
        latest_timestamp = existing_data.index.max()
        if latest_timestamp > pd.Timestamp.now() - timedelta(hours=2):
            print("Data is up to date")
            return existing_data

        print("Data needs updating...")

    # Fetch new data
    try:
        df = fetcher.fetch_from_end_time_working_backwards()

        if df.empty:
            raise ValueError(f"No data retrieved for {symbol}")

        # Validate data completeness
        days_of_data = (df.index.max() - df.index.min()).days
        if days_of_data < min_days:
            print(f"Warning: Only retrieved {days_of_data} days of data")

        # Validate data quality
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("Warning: Found missing values:")
            print(missing_values[missing_values > 0])

        # Basic statistics
        print(f"\nSuccessfully collected {len(df)} historical records")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Number of days: {days_of_data}")

        print("\nSample of data:")
        print(df.head())

        print("\nData statistics:")
        print(df.describe())

        return df

    except Exception as e:
        print(f"Error during data collection: {str(e)}")
        raise


if __name__ == "__main__":
    # Configuration
    SYMBOL = "BTCUSDT"
    INTERVAL = "1h"
    MIN_DAYS = 365  # Minimum one year of data

    try:
        # Fetch data with validation
        df = fetch_and_validate_data(SYMBOL, INTERVAL, MIN_DAYS)

        # Additional analysis if needed
        if df is not None:
            # Example: Check for gaps in hourly data
            expected_periods = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1H')
            missing_periods = expected_periods.difference(df.index)

            if len(missing_periods) > 0:
                print(f"\nWarning: Found {len(missing_periods)} missing periods")
                print("Sample of missing periods:")
                print(missing_periods[:5])

    except Exception as e:
        print(f"Script execution failed: {str(e)}")