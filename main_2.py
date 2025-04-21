import sys

from src.Config import Config
from src.BinanceHistoricalDataFetcher import BinanceHistoricalDataFetcher
from src.indicators.MACDProcessor import MACDProcessor
from src.indicators.BollingerBandsProcessor import BollingerBandsProcessor
from src.indicators.RSIProcessor import RSIProcessor
from src.indicators.MarketFeaturesProcessor import MarketFeaturesProcessor
from src.indicators.HorizonAlignedIndicatorsProcessor import HorizonAlignedIndicatorsProcessor
from src.indicators.HorizonAlignedIndicatorsProcessor_2 import HorizonAlignedIndicatorsProcessor_2
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from datetime import datetime

def convert_timestamp(timestamp_str):
    """New datetime parsed from a string"""
    try:
        return int(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp())
    except ValueError:
        print(f"Warning: Could not parse timestamp {timestamp_str}")
        return 0


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
    print(df.head(10))

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
def calculate_indicators(directory_name, symbols, intervals):
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

            # Load the data file
            # filename = data_dir / f"{symbol.lower()}_{interval}_historical_reduced.csv"
            filename = data_dir / f"{symbol.lower()}_{interval}_ohlcv_reduced_march_17.csv"
            # filename = data_dir / f"{symbol.lower()}_{interval}_ohlcv_reduced_april_15.csv"
            #filename = data_dir / f"{symbol.lower()}_{interval}_ohlcv.csv"

            print(f"calculate_indicators: loading historical dataset from: {filename}")
            df = pd.read_csv(filename)

            # Print column count
            print(f"Column count load: {len(df.columns)}")
            print(f"Column names: {df.columns.tolist()}")

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
            # horizon_processor = HorizonAlignedIndicatorsProcessor(
            #     data_dir=data_dir,
            #     forecast_steps=Config.FORECAST_STEPS,
            #     multiples=[1]
            # )
            horizon_processor = HorizonAlignedIndicatorsProcessor_2(
                data_dir=data_dir,
                forecast_steps=Config.FORECAST_STEPS,
                multiples=[1]
            )
            df = horizon_processor.process_csv(symbol, interval, df)

            print(f"Column count horizon aligned: {len(df.columns)}")
            # Save to CSV
            filename = data_dir / f"{symbol.lower()}_{interval}_reduced_python_processed_1_2_1_march_17_tmp.csv"
            # filename = data_dir / f"{symbol.lower()}_{interval}_reduced_python_processed_1_2_1_april_15.csv"
            #filename = data_dir / f"{symbol.lower()}_{interval}_python_processed_1_2_1_april_15.csv"
            df.to_csv(filename, index=False)
            print(f"Processed file stored at {filename}")


def prepare_ml_datasets(directory_name, symbols, intervals, target_horizon=24, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare datasets for machine learning by creating train/val/test splits
    and applying normalization

    Args:
        directory_name: Directory containing historical data
        symbols: List of trading pair symbols
        intervals: List of timeframe intervals
        target_horizon: Prediction horizon in periods
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation

    Returns:
        Dictionary with processed datasets for training, validation and testing
    """
    data_dir = Path(directory_name)

    # Initialize processors
    macd_processor = MACDProcessor(
        data_dir=data_dir,
        ma_fast=Config.MA_FAST,
        ma_slow=Config.MA_SLOW,
        signal_length=Config.SIGNAL_LENGTH
    )

    macd_processor_secondary = MACDProcessor(
        data_dir=data_dir,
        ma_fast=8,
        ma_slow=17,
        signal_length=9
    )

    bband_processor = BollingerBandsProcessor(
        data_dir=data_dir,
        length=Config.BOLL_LENGTH,
        multiplier=Config.BOLL_MULTIPLIER,
        slope_period=Config.SLOPE_PERIOD
    )

    bband_processor_secondary = BollingerBandsProcessor(
        data_dir=data_dir,
        length=50,
        multiplier=2.0,
        slope_period=Config.SLOPE_PERIOD
    )

    rsi_processor = RSIProcessor(
        data_dir=data_dir,
        length=Config.RSI_LOOKBACK,
        oversold=Config.RSI_OVERSOLD,
        overbought=Config.RSI_OVERBOUGHT
    )

    market_processor = MarketFeaturesProcessor(
        data_dir=data_dir,
        lag_periods=[1],
        volatility_windows=[4, 8],
        volume_windows=[4, 8]
    )

    # Horizon-Aligned Features processor
    horizon_processor = HorizonAlignedIndicatorsProcessor(
        data_dir=data_dir,
        forecast_steps=Config.FORECAST_STEPS,
        multiples=[1]
    )

    # Initialize data structures for split datasets
    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    # Process each symbol and interval
    for symbol in symbols:
        for interval in intervals:
            # Load the processed data
            filename = data_dir / f"{symbol.lower()}_{interval}_historical.csv"
            if not os.path.exists(filename):
                print(f"Warning: Data file not found for {symbol} {interval}")
                continue

            # Read the CSV file
            df = pd.read_csv(filename, index_col=0)

            # Calculate future price change for target variable
            df['future_price_change'] = df['close'].shift(-target_horizon) / df['close'] - 1

            # Remove rows with NaN in target variable
            df = df.dropna(subset=['future_price_change'])

            # Split the data chronologically
            train_end = int(len(df) * train_ratio)
            val_end = train_end + int(len(df) * val_ratio)

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]

            # Store in split_data structure
            key = f"{symbol.lower()}_{interval}"
            split_data['train'][key] = train_df
            split_data['val'][key] = val_df
            split_data['test'][key] = test_df

            print(f"Split {symbol} {interval} data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Process data with normalization for each processor
    print("\nApplying normalization to prepare machine learning datasets...")

    # Apply normalization for each processor
    processed_macd = macd_processor.process_data(split_data, symbols, intervals)
    processed_macd_secondary = macd_processor_secondary.process_data(split_data, symbols, intervals)
    processed_bband = bband_processor.process_data(split_data, symbols, intervals)
    processed_bband_secondary = bband_processor_secondary.process_data(split_data, symbols, intervals)
    processed_rsi = rsi_processor.process_data(split_data, symbols, intervals)
    processed_fundamentals = market_processor.process_data(split_data, symbols, intervals)
    processed_horizon_aligned_features = horizon_processor.process_data(split_data, symbols, intervals)

    # Get all the normalized features for reference
    all_features = []
    all_features.extend(macd_processor.get_feature_names(include_normalized=True))
    all_features.extend(macd_processor_secondary.get_feature_names(include_normalized=True))
    all_features.extend(bband_processor.get_feature_names(include_normalized=True))
    all_features.extend(bband_processor_secondary.get_feature_names(include_normalized=True))
    all_features.extend(rsi_processor.get_feature_names(include_normalized=True))
    all_features.extend(market_processor.get_feature_names(include_normalized=True))
    all_features.extend(horizon_processor.get_feature_names(include_normalized=True))

    # Save feature list for model training reference
    with open(data_dir / 'feature_list.json', 'w') as f:
        json.dump(all_features, f, indent=4)

    print(f"Generated {len(all_features)} normalized features for model training")
    print("ML datasets preparation complete")

    return {
        'macd': processed_macd,
        'macd_secondary': processed_macd_secondary,
        'bband': processed_bband,
        'bband_secondary': processed_bband_secondary,
        'rsi': processed_rsi,
        'fundamentals': processed_fundamentals,
        'features': all_features
    }


def create_sequence_datasets(processed_data, symbols, intervals, sequence_length=60):
    """
    Create sequence datasets for transformer-based models

    Args:
        processed_data: Dictionary with processed datasets
        symbols: List of trading pair symbols
        intervals: List of timeframe intervals
        sequence_length: Length of input sequences

    Returns:
        Dictionary with X and y for each split
    """
    # Initialize sequence datasets
    sequence_datasets = {
        'train': {'X': [], 'y': []},
        'val': {'X': [], 'y': []},
        'test': {'X': [], 'y': []}
    }

    # Get feature list
    all_features = processed_data['features']

    # Process each split
    for split in ['train', 'val', 'test']:
        for symbol in symbols:
            for interval in intervals:
                key = f"{symbol.lower()}_{interval}"

                # Collect data from all processors for this symbol/interval
                df_macd = processed_data['macd'][split].get(key)
                df_macd_secondary = processed_data['macd_secondary'][split].get(key)
                df_bband = processed_data['bband'][split].get(key)
                df_bband_secondary = processed_data['bband_secondary'][split].get(key)
                df_rsi = processed_data['rsi'][split].get(key)

                if df_macd is None or df_macd_secondary is None or df_bband is None or df_bband_secondary is None or df_rsi is None:
                    print(f"Warning: Missing data for {key} in {split} split")
                    continue

                # Ensure all dataframes have the same index
                df_macd = df_macd.copy()
                common_idx = df_macd.index.intersection(df_macd_secondary.index)
                common_idx = common_idx.intersection(df_bband.index)
                common_idx = common_idx.intersection(df_bband_secondary.index)
                common_idx = common_idx.intersection(df_rsi.index)

                # Filter all dataframes to common index
                df_macd = df_macd.loc[common_idx]
                df_macd_secondary = df_macd_secondary.loc[common_idx]
                df_bband = df_bband.loc[common_idx]
                df_bband_secondary = df_bband_secondary.loc[common_idx]
                df_rsi = df_rsi.loc[common_idx]

                # Get target variable
                y = df_macd['future_price_change'].values

                # Create feature matrix by selecting common normalized features
                X = pd.DataFrame(index=common_idx)

                # Add normalized features from each processor
                for feature in all_features:
                    if feature in df_macd.columns:
                        X[feature] = df_macd[feature]
                    elif feature in df_macd_secondary.columns:
                        X[feature] = df_macd_secondary[feature]
                    elif feature in df_bband.columns:
                        X[feature] = df_bband[feature]
                    elif feature in df_bband_secondary.columns:
                        X[feature] = df_bband_secondary[feature]
                    elif feature in df_rsi.columns:
                        X[feature] = df_rsi[feature]

                # Replace any remaining NaN with 0
                X = X.fillna(0)

                # Create sequences
                n_samples = len(X) - sequence_length

                for i in range(n_samples):
                    sequence_datasets[split]['X'].append(X.iloc[i:i + sequence_length].values)
                    sequence_datasets[split]['y'].append(y[i + sequence_length - 1])

                print(f"Added {n_samples} sequences from {key} to {split} dataset")

    # Convert to numpy arrays
    for split in sequence_datasets:
        sequence_datasets[split]['X'] = np.array(sequence_datasets[split]['X'])
        sequence_datasets[split]['y'] = np.array(sequence_datasets[split]['y'])

        print(
            f"{split} dataset shape: X={sequence_datasets[split]['X'].shape}, y={sequence_datasets[split]['y'].shape}")

    return sequence_datasets


def train_transformer_model(sequence_datasets):
    """
    Placeholder for transformer model training function

    This would be implemented with your preferred deep learning framework
    (PyTorch, TensorFlow, etc.)

    Args:
        sequence_datasets: Dictionary with sequence datasets
    """
    print("\nTransformer model training placeholder")
    print("Training dataset size:", sequence_datasets['train']['X'].shape)
    print("Validation dataset size:", sequence_datasets['val']['X'].shape)
    print("Test dataset size:", sequence_datasets['test']['X'].shape)


if __name__ == "__main__":
    # Configuration. symbols and intervals can be extended for multi-symbol and multi-interval processing
    symbols = ["BTCUSDT"]
    intervals = ["12h"]
    data_directory = "binance_futures_historical_data"
    # data_directory = "binance_us_historical_data"

    # Create data directory if it doesn't exist
    os.makedirs(data_directory, exist_ok=True)

    # 1. Fetch historical data for each symbol and interval
    # for symbol in symbols:
    #     for interval in intervals:
    #         get_historical_data(str_data_dir=data_directory, symbol=symbol, interval=interval, exchange="binance_futures")  # "binance_us"
    #         # get_data_working_forward(symbol=symbol, interval=interval, exchange="binance_us", string_datetime="2025-03-05 23:45:00")  # will update indicator values as well

    # 2. Calculate technical indicators for all symbols and intervals
    calculate_indicators(directory_name=data_directory, symbols=symbols, intervals=intervals)

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