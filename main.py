from src.BinanceHistoricalDataFetcher import BinanceHistoricalDataFetcher
from datetime import datetime
import time


def test_detailed_timestamp_fetch():
    base_dt = "2025-02-16 22:00:00"
    dt_obj = datetime.strptime(base_dt, "%Y-%m-%d %H:%M:%S")
    base_ts = int(dt_obj.timestamp() * 1000)

    # Try several smaller windows
    for hours_offset in range(6):
        start_ts = base_ts + (3600000 * hours_offset)  # Add hours one by one
        end_ts = start_ts + 3600000  # One hour window

        print(f"\nTrying fetch for hour {hours_offset}:")
        print(f"Start: {datetime.fromtimestamp(start_ts / 1000)}")
        print(f"End: {datetime.fromtimestamp(end_ts / 1000)}")

        chunk = fetcher.fetch_klines(
            start_time=start_ts,
            end_time=end_ts,
            limit=1000
        )

        if not chunk.empty:
            print(f"Got data! First record: {chunk.index[0]}")
            print(f"Last record: {chunk.index[-1]}")
            print(f"Number of records: {len(chunk)}")
        else:
            print("No data returned for this window")

        time.sleep(1)  # Respect rate limits

if __name__ == "__main__":
    fetcher = BinanceHistoricalDataFetcher(
        symbol="BTCUSDT",
        interval="4h",
        exchange="binance_us"
    )

    # Fetch complete history
    try:
        df = fetcher.fetch_from_end_time_working_backwards()
        if not df.empty:
            print(f"\nData Summary:")
            print(f"Total records: {len(df)}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            print("\nSample of data:")
            print(df.head())
        else:
            print("No data collected!")

    except Exception as e:
        print(f"Error during data collection: {str(e)}")
