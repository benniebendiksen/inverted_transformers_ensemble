from src.Config import Config
from src.BinanceHistoricalDataFetcher import BinanceHistoricalDataFetcher
from src.indicators.MACDProcessor import MACDProcessor
from src.indicators.RSIProcessor import RSIProcessor
from pathlib import Path


def get_historical_data(symbol, interval, exchange):
    fetcher = BinanceHistoricalDataFetcher(
        symbol=symbol,
        interval=interval,
        exchange=exchange
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


def calculate_macd_values(directory_name, symbol, interval):
    data_dir = Path(directory_name)
    processor = MACDProcessor(
        data_dir=data_dir,
        ma_fast=Config.MA_FAST,
        ma_slow=Config.MA_SLOW,
        signal_length=Config.SIGNAL_LENGTH
    )

    processor.process_csv(symbol, interval)


    processor = MACDProcessor(
        data_dir=data_dir,
        ma_fast=8,
        ma_slow=17,
        signal_length=9
    )
    # Process specific symbol and interval
    processor.process_csv(symbol, interval)

def calculate_rsi_values(directory_name, symbol, interval):
    data_dir = Path(directory_name)
    processor = RSIProcessor(
        data_dir=data_dir,
        length=Config.RSI_LOOKBACK,
        oversold=Config.RSI_OVERSOLD,
        overbought=Config.RSI_OVERBOUGHT
    )

    processor.process_csv(symbol, interval)


if __name__ == "__main__":
    ###############################
    ####### FETCH DATA ############
    ###############################
    symbol = "BTCUSDC"
    interval = "15m"
    exchange = "binance_us"
    # get_historical_data(symbol="BTCUSDC", interval="15m", exchange=exchange)

    ###############################
    ####### CALCULATE MACD VALUES #
    ###############################
    # calculate_macd_values(directory_name="binance_us_historical_data", symbol=symbol, interval=interval)

    ###############################
    ####### CALCULATE RSI VALUES #
    ###############################
    calculate_rsi_values(directory_name="binance_us_historical_data", symbol=symbol, interval=interval)

    ##################################
    ####### CALCULATE LOG and LOGGRO #
    ##################################
