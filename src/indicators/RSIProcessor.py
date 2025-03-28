import sys

from src.Config import Config
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional


class RSIProcessor:
    def __init__(self, data_dir: Path, length: int = Config.RSI_LOOKBACK,
                 oversold: float = Config.RSI_OVERSOLD, overbought: float = Config.RSI_OVERBOUGHT):
        """
        Initialize RSI processor with parameters

        Args:
            data_dir: Directory containing historical data CSV files
            length: RSI period length
            oversold: Oversold threshold for crossover signals
            overbought: Overbought threshold for crossover signals
        """
        self.data_dir = data_dir
        self.length = length
        self.oversold = oversold
        self.overbought = overbought

        # Define field names for base calculations
        self.rsi_field_name = f"RSI_{self.length}"
        self.rsi_signal_field_name = f"RSI_signal_{self.length}"

        # Define field names for enhanced features
        self.rsi_distance_field_name = f"RSI_distance_{self.length}"
        self.rsi_distance_norm_field_name = f"RSI_distance_norm_{self.length}"
        self.rsi_slope_field_name = f"RSI_slope_{self.length}"
        self.rsi_slope_norm_field_name = f"RSI_slope_norm_{self.length}"
        self.rsi_oversold_field_name = f"RSI_oversold_{self.length}"
        self.rsi_oversold_norm_field_name = f"RSI_oversold_norm_{self.length}"
        self.rsi_overbought_field_name = f"RSI_overbought_{self.length}"
        self.rsi_overbought_norm_field_name = f"RSI_overbought_norm_{self.length}"

        # Dictionary to store normalization parameters
        self.normalization_params = {}

        # Default slope period for calculating rates of change
        self.slope_period = 10

    def process_csv(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Process a single CSV file to add RSI indicators without normalization.
        This method focuses on feature calculation for the growing dataset.

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '1h', '4h')

        Returns:
            Processed DataFrame with RSI features
        """
        filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Historical data file not found: {filename}")

        # Read the CSV file
        df = pd.read_csv(filename, index_col=0)

        initial_row_count = len(df)

        # Calculate base RSI values
        df = self.calculate_rsi(df)

        # Calculate enhanced features with first-level normalization
        df = self.calculate_enhanced_features(df)

        if len(df) != initial_row_count:
            raise ValueError(f"Row count changed during processing: {initial_row_count} -> {len(df)}")

        # Save back to CSV
        df.to_csv(filename)

        print(f"Processed and stored RSI features for {filename}")
        print(f"Sample of processed data:")
        print(df[[
            'close',
            self.rsi_field_name,
            self.rsi_signal_field_name,
            self.rsi_distance_field_name,
            self.rsi_slope_field_name
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

                    # Check if RSI features already exist
                    if self.rsi_field_name not in df.columns:
                        df = self.calculate_rsi(df)
                        df = self.calculate_enhanced_features(df)

                    # Calculate normalization parameters from training data
                    self.fit_normalization_params(df, symbol, interval)

                    # Apply normalization
                    df = self.apply_normalization(df, symbol, interval)
                    processed_data['train'][key] = df

                    # Save normalization parameters if requested
                    if save_params:
                        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_rsi_params.json"
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

                            # Check if RSI features already exist
                            if self.rsi_field_name not in df.columns:
                                df = self.calculate_rsi(df)
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
        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_rsi_params.json"

        if symbol_key not in self.normalization_params:
            if os.path.exists(norm_params_filename):
                with open(norm_params_filename, 'r') as f:
                    self.normalization_params = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Normalization parameters not found for {symbol_key}. Process training data first.")

        # Calculate features if they don't exist
        if self.rsi_field_name not in df.columns:
            df = self.calculate_rsi(df)
            df = self.calculate_enhanced_features(df)

        # Apply normalization using stored parameters
        df = self.apply_normalization(df, symbol, interval)

        return df

    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and generate trading signals

        Args:
            df: DataFrame with historical price data

        Returns:
            DataFrame with added RSI columns
        """
        try:
            df = df.copy()
            close_prices = df['close'].values

            # Initialize arrays for gains and losses
            gains = np.zeros_like(close_prices)
            losses = np.zeros_like(close_prices)

            # Calculate initial price changes
            price_changes = np.diff(close_prices)
            gains[1:] = np.where(price_changes > 0, price_changes, 0)
            losses[1:] = np.where(price_changes < 0, -price_changes, 0)

            # Calculate initial averages
            avg_gain = np.sum(gains[1:self.length + 1]) / self.length
            avg_loss = np.sum(losses[1:self.length + 1]) / self.length

            # Initialize RSI array
            rsi_values = np.zeros_like(close_prices)
            rsi_values[:self.length] = 0  # First 'length' periods are 0

            # Calculate first RSI value
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[self.length] = 100 - (100 / (1 + rs))
            else:
                rsi_values[self.length] = 100

            # Calculate subsequent RSI values
            for i in range(self.length + 1, len(close_prices)):
                avg_gain = ((avg_gain * (self.length - 1)) + gains[i]) / self.length
                avg_loss = ((avg_loss * (self.length - 1)) + losses[i]) / self.length

                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi_values[i] = 100 - (100 / (1 + rs))
                else:
                    rsi_values[i] = 100

            # Add RSI values to DataFrame
            df[self.rsi_field_name] = rsi_values

            # Generate trading signals
            signals = np.full(len(df), 0, dtype=object)
            skip_next = 2  # Skip first two signals

            for i in range(2, len(df)):
                current_rsi = rsi_values[i]
                prev_rsi = rsi_values[i - 1]

                # Detect Long Entry (RSI crosses below oversold level)
                if current_rsi < self.oversold:
                    if skip_next <= 0:
                        signals[i] = -1

                # Detect Short Entry (RSI crosses above overbought level)
                elif current_rsi > self.overbought:
                    if skip_next <= 0:
                        signals[i] = 1

                if skip_next > 0:
                    skip_next -= 1

            df[self.rsi_signal_field_name] = signals

            return df
        except Exception as e:
            print(f"Error RSIProcessor: during calculate_rsi: {str(e)}")
            sys.exit(2)

    def calculate_enhanced_features(self, df):
        """
        Calculate enhanced RSI features with first-level normalization

        Args:
            df: DataFrame with RSI values

        Returns:
            DataFrame with enhanced RSI features
        """
        try:
            # 1. RSI Distance from mid-point (50)
            # This measures how far RSI is from equilibrium (i.e., an rsi value of 50)
            df[self.rsi_distance_field_name] = df[self.rsi_field_name] - 50

            # 2. RSI Slope (rate of change over slope_period)
            df[self.rsi_slope_field_name] = 0.0
            for i in range(self.slope_period, len(df)):
                df.loc[df.index[i], self.rsi_slope_field_name] = (
                        df[self.rsi_field_name].iloc[i] - df[self.rsi_field_name].iloc[i - self.slope_period]
                )

            # 3. RSI Oversold Distance - measures how far below oversold threshold
            # Positive when oversold (RSI < oversold threshold), 0 otherwise
            df[self.rsi_oversold_field_name] = np.maximum(self.oversold - df[self.rsi_field_name], 0)

            # 4. RSI Overbought Distance - measures how far above overbought threshold
            # Positive when overbought (RSI > overbought threshold), 0 otherwise
            df[self.rsi_overbought_field_name] = np.maximum(df[self.rsi_field_name] - self.overbought, 0)

            return df
        except Exception as e:
            print(f"Error RSIProcessor: during calculate_enhanced_features: {str(e)}")
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
            self.rsi_distance_field_name,
            self.rsi_slope_field_name,
            self.rsi_oversold_field_name,
            self.rsi_overbought_field_name
        ]

        # Initialize symbol-interval parameters if not exist
        symbol_key = f"{symbol}_{interval}"
        if symbol_key not in self.normalization_params:
            self.normalization_params[symbol_key] = {}

        # Calculate and store parameters for each feature
        for feature in features:
            # Skip initial NaN/zero values for more accurate statistics
            valid_data = df[feature].iloc[self.length + self.slope_period:]

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
            (self.rsi_distance_field_name, self.rsi_distance_norm_field_name),
            (self.rsi_slope_field_name, self.rsi_slope_norm_field_name),
            (self.rsi_oversold_field_name, self.rsi_oversold_norm_field_name),
            (self.rsi_overbought_field_name, self.rsi_overbought_norm_field_name)
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
            norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_rsi_params.json"
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
            self.rsi_field_name,
            self.rsi_signal_field_name,
            self.rsi_distance_field_name,
            self.rsi_slope_field_name,
            self.rsi_oversold_field_name,
            self.rsi_overbought_field_name
        ]

        if include_normalized:
            normalized_features = [
                self.rsi_distance_norm_field_name,
                self.rsi_slope_norm_field_name,
                self.rsi_oversold_norm_field_name,
                self.rsi_overbought_norm_field_name
            ]
            return base_features + normalized_features

        return base_features