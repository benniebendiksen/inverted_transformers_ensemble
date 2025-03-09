import sys

from src.Config import Config
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
from typing import Union, Dict, List, Tuple, Optional


class BollingerBandsProcessor:
    def __init__(self, data_dir: Path, length: int = Config.BOLL_LENGTH, multiplier: float = Config.BOLL_MULTIPLIER,
                 slope_period: int = Config.SLOPE_PERIOD):
        """
        Initialize Bollinger Bands processor with parameters

        Args:
            data_dir: Directory containing historical data CSV files
            length: Period for moving average calculation (default: 20)
            multiplier: Standard deviation multiplier (default: 2.0)
            slope_period: Period for calculating slopes
        """
        self.data_dir = data_dir
        self.length = length
        self.multiplier = multiplier
        self.slope_period = slope_period

        # Define field names for base calculations
        self.basis_field_name = f"BB_basis_{self.length}"
        self.upper_band_field_name = f"BB_upper_band_{self.length}"
        self.lower_band_field_name = f"BB_lower_band_{self.length}"
        self.signal_field_name = f"BB_bband_signal_{self.length}"

        # Define field names for enhanced features
        self.band_width_field_name = f"BB_band_width_{self.length}"
        self.band_width_norm_field_name = f"BB_band_width_norm_{self.length}"
        self.price_band_pos_field_name = f"BB_price_band_pos_{self.length}"
        self.price_band_pos_norm_field_name = f"BB_price_band_pos_norm_{self.length}"
        self.upper_dist_field_name = f"BB_upper_dist_{self.length}"
        self.upper_dist_norm_field_name = f"BB_upper_dist_norm_{self.length}"
        self.lower_dist_field_name = f"BB_lower_dist_{self.length}"
        self.lower_dist_norm_field_name = f"BB_lower_dist_norm_{self.length}"
        self.basis_slope_field_name = f"BB_basis_slope_{self.length}"
        self.basis_slope_norm_field_name = f"BB_basis_slope_norm_{self.length}"
        self.upper_slope_field_name = f"BB_upper_slope_{self.length}"
        self.upper_slope_norm_field_name = f"BB_upper_slope_norm_{self.length}"
        self.lower_slope_field_name = f"BB_lower_slope_{self.length}"
        self.lower_slope_norm_field_name = f"BB_lower_slope_norm_{self.length}"

        # Dictionary to store normalization parameters
        self.normalization_params = {}

    def process_csv(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Process a single CSV file to add Bollinger Bands indicators without normalization.
        This method focuses on feature calculation for the growing dataset.

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '1h', '4h')

        Returns:
            Processed DataFrame with Bollinger Bands features
        """
        filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Historical data file not found: {filename}")

        # Read the CSV file
        df = pd.read_csv(filename, index_col=0)

        # Calculate base Bollinger Bands values
        df = self.calculate_bollinger_values(df)

        # Calculate enhanced features with first-level normalization
        df = self.calculate_enhanced_features(df)

        # Save back to CSV
        df.to_csv(filename)

        print(f"Processed and stored Bollinger Bands features for {filename}")

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

                    # Check if Bollinger Bands features already exist
                    if self.basis_field_name not in df.columns:
                        df = self.calculate_bollinger_values(df)
                        df = self.calculate_enhanced_features(df)

                    # Calculate normalization parameters from training data ONLY
                    self.fit_normalization_params(df, symbol, interval)

                    # Apply normalization
                    df = self.apply_normalization(df, symbol, interval)
                    processed_data['train'][key] = df

                    # Save normalization parameters if requested
                    if save_params:
                        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_bband_params.json"
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

                            # Check if Bollinger Bands features already exist
                            if self.basis_field_name not in df.columns:
                                df = self.calculate_bollinger_values(df)
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
        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_bband_params.json"

        if symbol_key not in self.normalization_params:
            if os.path.exists(norm_params_filename):
                with open(norm_params_filename, 'r') as f:
                    self.normalization_params = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Normalization parameters not found for {symbol_key}. Process training data first.")

        # Calculate features if they don't exist
        if self.basis_field_name not in df.columns:
            df = self.calculate_bollinger_values(df)
            df = self.calculate_enhanced_features(df)

        # Apply normalization using stored parameters
        df = self.apply_normalization(df, symbol, interval)

        return df

    def calculate_bollinger_values(self, df):
        """
        Calculate base Bollinger Bands indicators
        """
        try:
            # Calculate Bollinger Bands indicators
            signals = self._generate_signals(df['close'])
            df[self.basis_field_name] = signals['basis']
            df[self.upper_band_field_name] = signals['upper_band']
            df[self.lower_band_field_name] = signals['lower_band']

            # Generate signal field for trend identification
            df[self.signal_field_name] = 0

            # Identify Bollinger Band Squeeze Exit (SE) and Lower Exit (LE) signals
            for i in range(self.length + 1, len(df)):
                current_close = df['close'].iloc[i]
                prev_close = df['close'].iloc[i - 1]
                prev_prev_close = df['close'].iloc[i - 2]

                current_upper = df[self.upper_band_field_name].iloc[i]
                prev_upper = df[self.upper_band_field_name].iloc[i - 1]
                prev_prev_upper = df[self.upper_band_field_name].iloc[i - 2]

                current_lower = df[self.lower_band_field_name].iloc[i]
                prev_lower = df[self.lower_band_field_name].iloc[i - 1]
                prev_prev_lower = df[self.lower_band_field_name].iloc[i - 2]

                # Check for Squeeze Exit (price crossing below upper band) "BBandSE"
                if prev_close < prev_upper and prev_prev_close >= prev_prev_upper:
                    prev_signal = df[self.signal_field_name].iloc[i - 1]
                    if prev_signal != 1:
                        df.loc[df.index[i], self.signal_field_name] = 1

                # Check for Lower Exit (price crossing above lower band) "BBandLE"
                elif prev_close > prev_lower and prev_prev_close <= prev_prev_lower:
                    prev_signal = df[self.signal_field_name].iloc[i - 1]
                    if prev_signal != -1:
                        df.loc[df.index[i], self.signal_field_name] = -1

            # Remove the first signal to match original function behavior
            first_signal_idx = df[df[self.signal_field_name] != 0].index.min()
            if first_signal_idx != 0:
                df.loc[first_signal_idx, self.signal_field_name] = 0

            return df
        except Exception as e:
            print(f"Error BollingerBandsProcessor: during calculate_bollinger_values: {str(e)}")
            sys.exit(2)

    def calculate_enhanced_features(self, df):
        """
        Calculate enhanced Bollinger Bands features with first-level normalization
        """
        try:
            # 1. Band Width: Width between bands as percentage of basis
            df[self.band_width_field_name] = (df[self.upper_band_field_name] - df[self.lower_band_field_name]) / df[
                self.basis_field_name].replace(0, np.nan) * 100

            # 2. Price Position: Where price is within the bands (0 = lower band, 1 = upper band)
            band_diff = df[self.upper_band_field_name] - df[self.lower_band_field_name]
            df[self.price_band_pos_field_name] = (df['close'] - df[self.lower_band_field_name]) / band_diff.replace(0,
                                                                                                                    np.nan)
            # Clip to handle cases where price is outside bands
            df[self.price_band_pos_field_name] = df[self.price_band_pos_field_name].clip(0, 1)

            # 3. Distance from Upper Band: Percentage distance from price to upper band
            df[self.upper_dist_field_name] = (df[self.upper_band_field_name] - df['close']) / df['close'].replace(0,
                                                                                                                  np.nan) * 100

            # 4. Distance from Lower Band: Percentage distance from price to lower band
            df[self.lower_dist_field_name] = (df['close'] - df[self.lower_band_field_name]) / df['close'].replace(0,
                                                                                                                  np.nan) * 100

            # 5. Calculate slopes for bands over specified period
            # Initialize slope columns with zeros
            df[self.basis_slope_field_name] = 0.0
            df[self.upper_slope_field_name] = 0.0
            df[self.lower_slope_field_name] = 0.0

            # Calculate slopes from row i-slope_period to row i
            for i in range(self.slope_period, len(df)):
                # Basis slope (percentage change over period)
                denom = df[self.basis_field_name].iloc[i - self.slope_period]
                if denom != 0 and not pd.isna(denom):
                    df.loc[df.index[i], self.basis_slope_field_name] = (
                            (df[self.basis_field_name].iloc[i] - denom) /
                            abs(denom) * 100
                    )

                # Upper band slope
                denom = df[self.upper_band_field_name].iloc[i - self.slope_period]
                if denom != 0 and not pd.isna(denom):
                    df.loc[df.index[i], self.upper_slope_field_name] = (
                            (df[self.upper_band_field_name].iloc[i] - denom) /
                            abs(denom) * 100
                    )

                # Lower band slope
                denom = df[self.lower_band_field_name].iloc[i - self.slope_period]
                if denom != 0 and not pd.isna(denom):
                    df.loc[df.index[i], self.lower_slope_field_name] = (
                            (df[self.lower_band_field_name].iloc[i] - denom) /
                            abs(denom) * 100
                    )

            # Replace any remaining NaN values with 0 for stability
            df = df.fillna(0)

            return df
        except Exception as e:
            print(f"Error BollingerBandsProcessor: during calculate_enhanced_features: {str(e)}")
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
            self.band_width_field_name,
            self.price_band_pos_field_name,
            self.upper_dist_field_name,
            self.lower_dist_field_name,
            self.basis_slope_field_name,
            self.upper_slope_field_name,
            self.lower_slope_field_name
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
            (self.band_width_field_name, self.band_width_norm_field_name),
            (self.price_band_pos_field_name, self.price_band_pos_norm_field_name),
            (self.upper_dist_field_name, self.upper_dist_norm_field_name),
            (self.lower_dist_field_name, self.lower_dist_norm_field_name),
            (self.basis_slope_field_name, self.basis_slope_norm_field_name),
            (self.upper_slope_field_name, self.upper_slope_norm_field_name),
            (self.lower_slope_field_name, self.lower_slope_norm_field_name)
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
            norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_bband_params.json"
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
            self.basis_field_name,
            self.upper_band_field_name,
            self.lower_band_field_name,
            self.band_width_field_name,
            self.price_band_pos_field_name,
            self.upper_dist_field_name,
            self.lower_dist_field_name,
            self.basis_slope_field_name,
            self.upper_slope_field_name,
            self.lower_slope_field_name,
            self.signal_field_name
        ]

        if include_normalized:
            normalized_features = [
                self.band_width_norm_field_name,
                self.price_band_pos_norm_field_name,
                self.upper_dist_norm_field_name,
                self.lower_dist_norm_field_name,
                self.basis_slope_norm_field_name,
                self.upper_slope_norm_field_name,
                self.lower_slope_norm_field_name
            ]
            return base_features + normalized_features

        return base_features

    def _generate_signals(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Generate Bollinger Bands signals from price series

        Args:
            price_series: Series of closing prices

        Returns:
            DataFrame with Bollinger Bands indicators
        """
        # Create signals DataFrame
        signals = pd.DataFrame(index=price_series.index)

        # Initialize with zeros for periods before we have enough data
        signals['basis'] = 0.0
        signals['upper_band'] = 0.0
        signals['lower_band'] = 0.0

        # Calculate for valid periods
        for i in range(self.length - 1, len(price_series)):
            # Get window of prices for calculation
            window = price_series.iloc[i - self.length + 1:i + 1]

            # Calculate simple moving average (basis)
            basis = window.mean()

            # Calculate standard deviation
            dev = window.std() * self.multiplier

            # Calculate upper and lower bands
            upper_band = basis + dev
            lower_band = basis - dev

            # Update signals at this position
            signals.loc[price_series.index[i], 'basis'] = basis
            signals.loc[price_series.index[i], 'upper_band'] = upper_band
            signals.loc[price_series.index[i], 'lower_band'] = lower_band

        return signals