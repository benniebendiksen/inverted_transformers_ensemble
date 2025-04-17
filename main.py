import sys

from src.Config import Config
from src.BinanceHistoricalDataFetcher import BinanceHistoricalDataFetcher
from src.indicators.MACDProcessor import MACDProcessor
from src.indicators.BollingerBandsProcessor import BollingerBandsProcessor
from src.indicators.RSIProcessor import RSIProcessor
from src.indicators.MarketFeaturesProcessor import MarketFeaturesProcessor
from src.indicators.HorizonAlignedIndicatorsProcessor import HorizonAlignedIndicatorsProcessor
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from datetime import datetime

def convert_timestamp(timestamp_str):
    """Convert human-readable timestamp to Unix timestamp"""
    try:
        return int(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp())
    except ValueError:
        print(f"Warning: Could not parse timestamp {timestamp_str}")
        return 0


def preprocess_dataset(df, data_dir, symbol, interval):
    """
    Preprocess the dataset before calculating indicators.

    Args:
        df: DataFrame with historical data

    Returns:
        DataFrame with processed timestamp and ordered columns
    """
    print(f"Starting preprocessing. Original columns: {df.columns.tolist()}")

    # Make a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()

    # Check if we need to rename time column
    if 'time' in processed_df.columns and 'timestamp' not in processed_df.columns:
        print(f"converting time column to timestamp")
        processed_df = processed_df.rename(columns={'time': 'timestamp'})

    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], unit='s')

    # Get the list of required columns that should come first
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Make sure all required columns exist
    for col in required_columns:
        if col not in processed_df.columns:
            print(col)
            raise ValueError(f"Required column not found in the dataset")

    # Get list of all other columns (that aren't in required_columns)
    other_columns = [col for col in processed_df.columns if col not in required_columns]

    # Reorder columns: first the required columns, then all other columns
    processed_df = processed_df[required_columns + other_columns]

    print(f"Preprocessing complete. First columns: {processed_df.columns[:10].tolist()}")
    print(f"Sample of preprocessed data:")
    print(processed_df.head())
    filename = data_dir / f"{symbol.lower()}_{interval}_seed_april_15.csv"
    processed_df.to_csv(filename)
    print(f"Processed and stored at {filename}")

    return processed_df


def add_price_directionality(df):
    """
    Add a price directionality indicator to the dataframe
    1 indicates price increase, 0 indicates non-increasing price

    Args:
        df: DataFrame with historical data

    Returns:
        DataFrame with added directionality column
    """
    print(f"Row count at start of add_price_directionality: {len(df)}")

    # Calculate price change
    df['price_change'] = df['close'].diff()

    # Create directionality indicator (1 for increase, 0 for non-increase)
    df['direction'] = (df['price_change'] > 0).astype(int)

    # Remove the temporary price_change column
    df = df.drop('price_change', axis=1)

    print(f"Direction distribution: {df['direction'].value_counts().to_dict()}")

    print(f"\nData Summary 2:")
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    print("\nSample of data:")
    print(df.tail(10))

    return df


def get_historical_data(str_data_dir, symbol, interval, exchange):
    """
    Fetch historical data from Binance

    Args:
        symbol: Trading pair symbol
        interval: Timeframe interval
        exchange: Exchange name

    Returns:
        DataFrame with historical data if successful, None otherwise
    """
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
            print(df.tail(10))

            data_directory = Path(str_data_dir)
            filename = data_directory / f"{symbol.lower()}_{interval}_historical.csv"
            print(f"get_historical_data: saving file to: {filename}")
            df.to_csv(filename)

            return df
        else:
            print("No data collected!")
            return None

    except Exception as e:
        print(f"Error during data collection: {str(e)}")
        sys.exit(1)


def get_data_working_forward(symbol, interval, exchange, string_datetime):
    """
    Fetch historical data from binance exchange working forwards

    Args:
        symbol: Trading pair symbol
        interval: Timeframe interval
        exchange: Exchange name
        string_datetime: String datetime to start fetching data from and proceed working forwards

    Returns:
        DataFrame with historical data if successful, None otherwise
    """
    fetcher = BinanceHistoricalDataFetcher(
        symbol=symbol,
        interval=interval,
        exchange=exchange
    )

    # Fetch complete history
    try:
        df = fetcher.fetch_from_start_time_working_forwards(string_datetime)
        if not df.empty:
            print(f"\nData Summary:")
            print(f"Total records: {len(df)}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            print("\nSample of data:")
            print(df.head())
            return df
        else:
            print("No data collected!")
            return None

    except Exception as e:
        print(f"Error during data collection: {str(e)}")
        return None


# Modify the calculate_indicators function to include the directionality calculation
def calculate_indicators(directory_name, symbols, intervals, filename, output_filename):
    """
    Calculate technical indicators for all specified symbols and intervals

    Args:
        directory_name: Directory containing historical data
        symbols: List of trading pair symbols
        intervals: List of timeframe intervals
    """
    data_dir = Path(directory_name)

    # Process each symbol and interval
    for symbol in symbols:
        for interval in intervals:
            print(f"\nProcessing {symbol} {interval} data with technical indicators...")
            df = pd.read_csv(filename)

            # Print column count
            print(f"Column count load: {len(df.columns)}")

            # Check if we need to convert timestamp
            if 'time' in df.columns and 'timestamp' not in df.columns:
                print(f"Converting time column to timestamp")
                # Check if time column contains Unix timestamps (integers) or datetime strings
                sample_time = df['time'].iloc[0]

                try:
                    # If it's already a Unix timestamp (integer)
                    if isinstance(sample_time, (int, float)) or str(sample_time).isdigit():
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    else:
                        # If it's a string datetime format
                        df['timestamp'] = pd.to_datetime(df['time'])

                    # Drop the original time column
                    df.drop('time', axis=1, inplace=True)

                    # Reorder columns to put timestamp first
                    cols = df.columns.tolist()
                    cols = ['timestamp'] + [col for col in cols if col != 'timestamp']
                    df = df[cols]

                    print(f"Successfully converted time to timestamp")
                except Exception as e:
                    print(f"Error converting time to timestamp: {e}")
                    # Fallback method using the custom convert_timestamp function
                    print("Attempting to convert using custom function...")
                    df['timestamp'] = df['time'].apply(convert_timestamp)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.drop('time', axis=1, inplace=True)

            # Apply preprocessing
            #df = preprocess_dataset(df, data_dir, symbol, interval)

            # Add price directionality indicator before other processing
            df = add_price_directionality(df)

            # Save the updated dataframe back to CSV
            # df.to_csv(filename)

            print(f"Column count add dir: {len(df.columns)}")

            # Primary MACD processor (12-26-9 standard configuration)
            macd_processor = MACDProcessor(
                data_dir=data_dir,
                ma_fast=Config.MA_FAST,
                ma_slow=Config.MA_SLOW,
                signal_length=Config.SIGNAL_LENGTH
            )
            df = macd_processor.process_csv(symbol, interval, df)

            print(f"Column count mac 1: {len(df.columns)}")

            # Secondary MACD processor (8-17-9 configuration)
            macd_processor_secondary = MACDProcessor(
                data_dir=data_dir,
                ma_fast=8,
                ma_slow=17,
                signal_length=9
            )
            df = macd_processor_secondary.process_csv(symbol, interval, df)

            print(f"Column count mac 2: {len(df.columns)}")

            # Primary Bollinger Bands processor (20-period standard configuration)
            bband_processor = BollingerBandsProcessor(
                data_dir=data_dir,
                length=Config.BOLL_LENGTH,
                multiplier=Config.BOLL_MULTIPLIER,
                slope_period=Config.SLOPE_PERIOD
            )
            df = bband_processor.process_csv(symbol, interval, df)

            print(f"Column count bband: {len(df.columns)}")

            # Secondary Bollinger Bands processor (50-period longer term configuration)
            bband_processor_secondary = BollingerBandsProcessor(
                data_dir=data_dir,
                length=50,
                multiplier=Config.BOLL_MULTIPLIER,
                slope_period=Config.SLOPE_PERIOD
            )
            df = bband_processor_secondary.process_csv(symbol, interval, df)

            print(f"Column count bband 2: {len(df.columns)}")

            # RSI processor
            rsi_processor = RSIProcessor(
                data_dir=data_dir,
                length=Config.RSI_LOOKBACK,
                oversold=Config.RSI_OVERSOLD,
                overbought=Config.RSI_OVERBOUGHT
            )
            df = rsi_processor.process_csv(symbol, interval, df)

            print(f"Column count rsi: {len(df.columns)}")

            # Fundamental Market Features processor
            market_processor = MarketFeaturesProcessor(
                data_dir=data_dir,
                lag_periods=[1],
                volatility_windows=[4, 8],
                volume_windows=[4, 8]
            )
            df = market_processor.process_csv(symbol, interval, df)

            print(f"Column count market features: {len(df.columns)}")

            # Horizon-Aligned Features processor
            horizon_processor = HorizonAlignedIndicatorsProcessor(
                data_dir=data_dir,
                forecast_steps=Config.FORECAST_STEPS,
                multiples=[1]
            )
            df = horizon_processor.process_csv(symbol, interval, df)

            print(f"Column count horizon aligned: {len(df.columns)}")
            # Save to CSV
            # filename = data_dir / f"{symbol.lower()}_{interval}_historical_reduced_python_processed_1_2_1_old.csv"
            # filename = data_dir / f"{symbol.lower()}_{interval}_historical_reduced_python_processed_1_2_1_old_reattempt.csv"
            df.to_csv(output_filename, index=False)
            print(f"Processed and stored at {output_filename}")


if __name__ == "__main__":
    # Configuration. symbols and intervals can be extended for multi-symbol and multi-interval processing
    symbols = ["BTCUSDT"]
    intervals = ["4h"]

    data_directory = "binance_futures_historical_data"
    # Create data directory if it doesn't exist
    os.makedirs(data_directory, exist_ok=True)
    data_dir = Path(data_directory)

    filename = data_dir / "btcusdt_4h_features_04_05.csv"
    output_filename=data_dir / f"{symbols[0].lower()}_{intervals[0]}_features_april_15_lance_4.csv"
    print(f"Loaded historical dataset from: {filename}")

    # 1. Fetch historical data for each symbol and interval
    # for symbol in symbols:
    #     for interval in intervals:
    #         get_historical_data(str_data_dir=data_directory, symbol=symbol, interval=interval, exchange="binance_futures")  # "binance_us"
    #         # get_data_working_forward(symbol=symbol, interval=interval, exchange="binance_us", string_datetime="2025-03-05 23:45:00")  # will update indicator values as well

    # 2. Calculate technical indicators for all symbols and intervals
    calculate_indicators(directory_name=data_directory, symbols=symbols, intervals=intervals, filename=filename, output_filename=output_filename)

    # # 3. Prepare datasets for machine learning with normalization
    # processed_data = prepare_ml_datasets(
    #     directory_name=data_directory,
    #     symbols=symbols,
    #     intervals=intervals,
    #     target_horizon=24,  # 24 periods ahead prediction
    #     train_ratio=0.7,
    #     val_ratio=0.15
    # )
    #
    # # 4. Create sequence datasets for transformer model
    # sequence_datasets = create_sequence_datasets(
    #     processed_data=processed_data,
    #     symbols=symbols,
    #     intervals=intervals,
    #     sequence_length=60  # 60 periods for input sequence
    # )
    #
    # # 5. Train transformer model (placeholder)
    # # This would be implemented with your preferred deep learning framework
    # train_transformer_model(sequence_datasets)
    #
    # # 6. Save experiment metadata
    # experiment_info = {
    #     "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     "symbols": symbols,
    #     "intervals": intervals,
    #     "target_horizon": 24,
    #     "sequence_length": 60,
    #     "train_samples": len(sequence_datasets['train']['y']),
    #     "val_samples": len(sequence_datasets['val']['y']),
    #     "test_samples": len(sequence_datasets['test']['y'])
    # }
    #
    # with open(Path(data_directory) / 'experiment_info.json', 'w') as f:
    #     json.dump(experiment_info, f, indent=4)
    #
    # print("\nExperiment information saved.")
    # print("Pipeline execution complete.")
