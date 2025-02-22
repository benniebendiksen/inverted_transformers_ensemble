from src.BinanceHistoricalDataFetcher import BinanceHistoricalDataFetcher

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
