import sys

from src.Config import Config
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional


class MACDProcessor:
    def __init__(self, data_dir: Path, ma_fast: int = Config.MA_FAST, ma_slow: int = Config.MA_SLOW,
                 signal_length: int = Config.SIGNAL_LENGTH):
        """
        Initialize MACD processor with parameters

        Args:
            data_dir: Directory containing historical data CSV files
            ma_fast: Fast EMA period
            ma_slow: Slow EMA period
            signal_length: Signal line EMA period
        """
        self.data_dir = data_dir
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.signal_length = signal_length

        # Determine MACD variant based on parameters
        if self.ma_fast == 12:
            self.macd_variant = "long"
        else:
            self.macd_variant = "short"

        # Define field names for base calculations
        self.macd_line_field_name = f"MACD_macd_line_{self.macd_variant}"
        self.signal_line_field_name = f"MACD_signal_line_{self.macd_variant}"
        self.histogram_field_name = f"MACD_macd_histogram_{self.macd_variant}"
        self.trend_direction_field_name = f"MACD_trend_direction_{self.macd_variant}"

        # Define field names for enhanced features
        self.macd_distance_field_name = f"MACD_macd_distance_{self.macd_variant}"
        self.macd_distance_norm_field_name = f"MACD_macd_distance_norm_{self.macd_variant}"
        self.macd_slope_field_name = f"MACD_macd_slope_{self.macd_variant}"
        self.macd_slope_norm_field_name = f"MACD_macd_slope_norm_{self.macd_variant}"
        self.signal_slope_field_name = f"MACD_signal_slope_{self.macd_variant}"
        self.signal_slope_norm_field_name = f"MACD_signal_slope_norm_{self.macd_variant}"
        self.histogram_slope_field_name = f"MACD_histogram_slope_{self.macd_variant}"
        self.histogram_slope_norm_field_name = f"MACD_histogram_slope_norm_{self.macd_variant}"

        # Dictionary to store normalization parameters
        self.normalization_params = {}

        # Default slope period for calculating rates of change
        self.slope_period = 4

    def process_csv(self, symbol: str, interval: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single CSV file to add MACD indicators without normalization.
        This method focuses on feature calculation for the growing dataset.

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '1h', '4h')

        Returns:
            Processed DataFrame with MACD features
            :param filename:
        """
        # filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        # if not os.path.exists(filename):
        #     raise FileNotFoundError(f"Historical data file not found: {filename}")
        #
        # # Read the CSV file
        # df = pd.read_csv(filename, index_col=0)

        initial_row_count = len(df)

        # Calculate base MACD values
        df = self.calculate_macd_values(df)

        # Calculate enhanced features with first-level normalization
        df = self.calculate_enhanced_features(df)

        if len(df) != initial_row_count:
            raise ValueError(f"Row count changed during processing: {initial_row_count} -> {len(df)}")


        # # Save back to CSV
        # df.to_csv(filename)
        #
        # print(f"Processed and stored MACD features for {filename}")
        print(f"Sample of processed data:")
        print(df[[
            'close',
            self.macd_line_field_name,
            self.signal_line_field_name,
            self.histogram_field_name,
            self.trend_direction_field_name
        ]].tail())

        return df

    def process_data(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], symbols: List[str],
                     intervals: List[str], save_params: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process pre-split data for machine learning - calculating normalization parameters
        from training data and applying them to all splits.

        Args:
            data_dict: Dictionary with structure {split_name: {symbol_interval: dataframe}}
                       where split_name is 'train', 'val', or 'test'
            symbols: List of trading pair symbols to process
            intervals: List of timeframe intervals to process
            save_params: Whether to save normalization parameters to disk

        Returns:
            Dictionary with the same structure containing processed dataframes
        """
        processed_data = {}

        # First, process training data to calculate normalization parameters
        if 'train' not in data_dict:
            raise ValueError("Training data is required to calculate normalization parameters")

        processed_data['train'] = {}
        for symbol in symbols:
            for interval in intervals:
                key = f"{symbol.lower()}_{interval}"
                if key in data_dict['train']:
                    # Process training data and calculate normalization parameters
                    df = data_dict['train'][key].copy()

                    # Check if MACD features already exist
                    if self.macd_line_field_name not in df.columns:
                        df = self.calculate_macd_values(df)
                        df = self.calculate_enhanced_features(df)

                    # Calculate normalization parameters from training data
                    self.fit_normalization_params(df, symbol, interval)

                    # Apply normalization
                    df = self.apply_normalization(df, symbol, interval)
                    processed_data['train'][key] = df

                    # Save normalization parameters if requested
                    if save_params:
                        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_macd_params.json"
                        with open(norm_params_filename, 'w') as f:
                            json.dump(self.normalization_params, f, indent=4)
                        print(f"Saved normalization parameters for {key}")

        # Process validation and test data using the parameters from training data
        for split in ['val', 'test']:
            if split in data_dict:
                processed_data[split] = {}
                for symbol in symbols:
                    for interval in intervals:
                        key = f"{symbol.lower()}_{interval}"
                        if key in data_dict[split]:
                            # Process data
                            df = data_dict[split][key].copy()

                            # Check if MACD features already exist
                            if self.macd_line_field_name not in df.columns:
                                df = self.calculate_macd_values(df)
                                df = self.calculate_enhanced_features(df)

                            # Apply normalization
                            df = self.apply_normalization(df, symbol, interval)
                            processed_data[split][key] = df

        return processed_data

    def process_inference_data(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """
        Process new data for inference using stored normalization parameters

        Args:
            df: DataFrame containing historical data
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '1h', '4h')

        Returns:
            Processed DataFrame with normalized features ready for prediction
        """
        # Check if we have normalization parameters
        symbol_key = f"{symbol}_{interval}"
        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_macd_params.json"

        if symbol_key not in self.normalization_params:
            if os.path.exists(norm_params_filename):
                with open(norm_params_filename, 'r') as f:
                    self.normalization_params = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Normalization parameters not found for {symbol_key}. Process training data first.")

        # Calculate features if they don't exist
        if self.macd_line_field_name not in df.columns:
            df = self.calculate_macd_values(df)
            df = self.calculate_enhanced_features(df)

        # Apply normalization using stored parameters
        df = self.apply_normalization(df, symbol, interval)

        return df

    def calculate_macd_values(self, df):
        """
        Calculate base MACD indicators
        """
        try:
            # Calculate MACD indicators
            signals = self._generate_signals(df['close'])

            # Store the components
            df[self.macd_line_field_name] = signals['macd_line']
            df[self.signal_line_field_name] = signals['signal_line']
            df[self.histogram_field_name] = signals['MACD-Signal']

            # Generate trend direction
            df[self.trend_direction_field_name] = -99
            df.loc[signals['macd_line'] > signals['signal_line'], self.trend_direction_field_name] = 1
            df.loc[signals['macd_line'] < signals['signal_line'], self.trend_direction_field_name] = -1
            df.loc[signals['macd_line'] == signals['signal_line'], self.trend_direction_field_name] = 0

            return df
        except Exception as e:
            print(f"Error MACDProcessor: during calculate_macd_values: {str(e)}")
            sys.exit(2)

    def calculate_enhanced_features(self, df):
        """
        Calculate enhanced MACD features with first-level normalization
        """
        # 1. MACD Distance: Relative distance between MACD and Signal lines
        # Normalize by average true range over the same period for scale-invariance
        # Using close prices for normalization as an approximation
        try:
            price_scale = df['close'].rolling(window=self.ma_slow).mean()
            df[self.macd_distance_field_name] = (df[self.macd_line_field_name] - df[
                self.signal_line_field_name]) / price_scale * 100

            # 2. Calculate slopes (percentage change over slope_period)
            # Initialize slope columns
            df[self.macd_slope_field_name] = 0.0
            df[self.signal_slope_field_name] = 0.0
            df[self.histogram_slope_field_name] = 0.0

            # Calculate slopes from row i-slope_period to row i
            for i in range(self.slope_period, len(df)):
                # Skip if we encounter zero in the denominator to avoid division by zero
                # MACD line slope
                if abs(df[self.macd_line_field_name].iloc[i - self.slope_period]) > 1e-9:
                    df.loc[df.index[i], self.macd_slope_field_name] = (
                            (df[self.macd_line_field_name].iloc[i] - df[self.macd_line_field_name].iloc[
                                i - self.slope_period]) /
                            abs(df[self.macd_line_field_name].iloc[i - self.slope_period]) * 100
                    )

                # Signal line slope
                if abs(df[self.signal_line_field_name].iloc[i - self.slope_period]) > 1e-9:
                    df.loc[df.index[i], self.signal_slope_field_name] = (
                            (df[self.signal_line_field_name].iloc[i] - df[self.signal_line_field_name].iloc[
                                i - self.slope_period]) /
                            abs(df[self.signal_line_field_name].iloc[i - self.slope_period]) * 100
                    )

                # Histogram slope
                if abs(df[self.histogram_field_name].iloc[i - self.slope_period]) > 1e-9:
                    df.loc[df.index[i], self.histogram_slope_field_name] = (
                            (df[self.histogram_field_name].iloc[i] - df[self.histogram_field_name].iloc[
                                i - self.slope_period]) /
                            abs(df[self.histogram_field_name].iloc[i - self.slope_period]) * 100
                    )

            return df
        except Exception as e:
            print(f"Error MACDProcessor: during calculate_enhanced_features: {str(e)}")
            sys.exit(2)

    def fit_normalization_params(self, df, symbol, interval):
        """
        Calculate normalization parameters from training data

        Args:
            df: DataFrame containing the features to normalize
            symbol: Trading pair symbol
            interval: Timeframe interval
        """
        # Features to normalize
        features = [
            self.macd_distance_field_name,
            self.macd_slope_field_name,
            self.signal_slope_field_name,
            self.histogram_slope_field_name
        ]

        # Initialize symbol-interval parameters if not exist
        symbol_key = f"{symbol}_{interval}"
        if symbol_key not in self.normalization_params:
            self.normalization_params[symbol_key] = {}

        # Calculate and store parameters for each feature
        for feature in features:
            # Skip initial values for more accurate statistics
            min_lookback = max(self.ma_slow, self.slope_period) + 5
            valid_data = df[feature].iloc[min_lookback:]

            # Remove infinite values and nulls
            valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna()

            if len(valid_data) > 0:
                # Calculate mean and standard deviation
                mean = valid_data.mean()
                std = valid_data.std()

                if std == 0 or pd.isna(std):
                    std = 1.0  # Avoid division by zero

                # Store parameters
                self.normalization_params[symbol_key][feature] = {
                    "mean": float(mean),
                    "std": float(std)
                }
            else:
                # If no valid data, use default parameters
                self.normalization_params[symbol_key][feature] = {
                    "mean": 0.0,
                    "std": 1.0
                }

    def apply_normalization(self, df, symbol, interval):
        """
        Apply z-score normalization using pre-calculated parameters

        Args:
            df: DataFrame containing the features to normalize
            symbol: Trading pair symbol
            interval: Timeframe interval

        Returns:
            DataFrame with normalized features added
        """
        # Features to normalize and their corresponding normalized field names
        feature_pairs = [
            (self.macd_distance_field_name, self.macd_distance_norm_field_name),
            (self.macd_slope_field_name, self.macd_slope_norm_field_name),
            (self.signal_slope_field_name, self.signal_slope_norm_field_name),
            (self.histogram_slope_field_name, self.histogram_slope_norm_field_name)
        ]

        symbol_key = f"{symbol}_{interval}"

        # Apply normalization to each feature
        for src_field, dest_field in feature_pairs:
            if symbol_key in self.normalization_params and src_field in self.normalization_params[symbol_key]:
                mean = self.normalization_params[symbol_key][src_field]["mean"]
                std = self.normalization_params[symbol_key][src_field]["std"]

                # Apply z-score normalization
                df[dest_field] = (df[src_field] - mean) / std
            else:
                raise ValueError(f"Normalization parameters not found for {symbol_key}/{src_field}")

        return df

    def denormalize_values(self, normalized_values, symbol, interval, feature_name):
        """
        Convert normalized values back to their original scale

        Args:
            normalized_values: Z-score normalized values
            symbol: Trading pair symbol
            interval: Timeframe interval
            feature_name: Name of the feature to denormalize

        Returns:
            Denormalized values
        """
        symbol_key = f"{symbol}_{interval}"

        if symbol_key not in self.normalization_params or feature_name not in self.normalization_params[symbol_key]:
            # Try to load parameters from saved file
            norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_macd_params.json"
            if os.path.exists(norm_params_filename):
                with open(norm_params_filename, 'r') as f:
                    self.normalization_params = json.load(f)
            else:
                raise ValueError(f"Normalization parameters not found for {symbol_key}/{feature_name}")

        # Get mean and std for this feature
        mean = self.normalization_params[symbol_key][feature_name]["mean"]
        std = self.normalization_params[symbol_key][feature_name]["std"]

        # Denormalize
        return normalized_values * std + mean

    def get_feature_names(self, include_normalized=True):
        """
        Get list of feature names generated by this processor

        Args:
            include_normalized: Whether to include normalized feature names

        Returns:
            List of feature names
        """
        base_features = [
            self.macd_line_field_name,
            self.signal_line_field_name,
            self.histogram_field_name,
            self.trend_direction_field_name,
            self.macd_distance_field_name,
            self.macd_slope_field_name,
            self.signal_slope_field_name,
            self.histogram_slope_field_name
        ]

        if include_normalized:
            normalized_features = [
                self.macd_distance_norm_field_name,
                self.macd_slope_norm_field_name,
                self.signal_slope_norm_field_name,
                self.histogram_slope_norm_field_name
            ]
            return base_features + normalized_features

        return base_features

    def _generate_signals(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Generate MACD signals from price series

        Args:
            price_series: Series of closing prices

        Returns:
            DataFrame with MACD indicators
        """
        # Calculate EMAs
        ma_fast = price_series.ewm(span=self.ma_fast, adjust=True).mean()
        ma_slow = price_series.ewm(span=self.ma_slow, adjust=True).mean()

        # Calculate MACD line
        macd_line = ma_fast - ma_slow

        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_length).mean()

        # Create signals DataFrame
        signals = pd.DataFrame(index=price_series.index)
        signals['macd_line'] = macd_line
        signals['signal_line'] = signal_line
        signals['MACD-Signal'] = signals['macd_line'] - signals['signal_line']

        return signals