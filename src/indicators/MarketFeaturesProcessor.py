import sys

from src.Config import Config
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional


class MarketFeaturesProcessor:
    def __init__(self, data_dir: Path, lag_periods: List[int] = None,
                 volatility_windows: List[int] = None, volume_windows: List[int] = None,
                 include_candlestick: bool = True, include_volume: bool = True):
        """
        Initialize Market Features processor with parameters

        Args:
            data_dir: Directory containing historical data CSV files
            lag_periods: List of periods for lagged features (e.g., [1, 5, 10, 20])
            volatility_windows: List of window sizes for volatility calculation
            volume_windows: List of window sizes for volume metrics
            include_candlestick: Whether to include candlestick relationship features
            include_volume: Whether to include volume-based features
        """
        self.data_dir = data_dir
        self.lag_periods = lag_periods if lag_periods is not None else [1, 5, 10, 20]
        self.volatility_windows = volatility_windows if volatility_windows is not None else [5, 10, 20]
        self.volume_windows = volume_windows if volume_windows is not None else [5, 10, 20, 50]
        self.include_candlestick = include_candlestick
        self.include_volume = include_volume

        # Dictionary to store normalization parameters
        self.normalization_params = {}

        # Define field names for base calculations
        # self.base_fields = {
        #     # Price returns
        #     'returns': 'price_return',
        #
        #     # Price velocity (first derivative)
        #     'velocity': 'price_velocity',
        #
        #     # Price acceleration (second derivative)
        #     'acceleration': 'price_acceleration'
        # }

        # Will be populated in respective calculation methods
        self.candlestick_fields = []
        self.volume_fields = []

        # Generate field names for all features
        self.field_names = self._generate_field_names()

    def _generate_field_names(self):
        """
        Generate all field names for the processor

        Returns:
            Dictionary of field names
        """
        field_names = {}

        # Lagged returns
        for lag in self.lag_periods:
            # Lagged simple returns
            field_names[f'MARKET_FEATURES_return_lag_{lag}'] = f'MARKET_FEATURES_return_lag_{lag}'
            field_names[f'MARKET_FEATURES_return_lag_{lag}_norm'] = f'MARKET_FEATURES_return_lag_{lag}_norm'

        # Velocity features
        for lag in self.lag_periods:
            # Lagged price velocity
            field_names[f'MARKET_FEATURES_velocity_lag_{lag}'] = f'MARKET_FEATURES_velocity_lag_{lag}'
            field_names[f'MARKET_FEATURES_velocity_lag_{lag}_norm'] = f'MARKET_FEATURES_velocity_lag_{lag}_norm'

        # Acceleration features
        for lag in self.lag_periods:
            # Lagged price acceleration
            field_names[f'MARKET_FEATURES_acceleration_lag_{lag}'] = f'MARKET_FEATURES_acceleration_lag_{lag}'
            field_names[f'MARKET_FEATURES_acceleration_lag_{lag}_norm'] = f'MARKET_FEATURES_acceleration_lag_{lag}_norm'

        # Volatility features
        for window in self.volatility_windows:
            field_names[f'MARKET_FEATURES_volatility_{window}'] = f'MARKET_FEATURES_volatility_{window}'
            field_names[f'MARKET_FEATURES_volatility_{window}_norm'] = f'MARKET_FEATURES_volatility_{window}_norm'

        return field_names

    def process_csv(self, symbol: str, interval: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single CSV file to add market features without normalization.
        This method focuses on feature calculation for the growing dataset.

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '1h', '4h')

        Returns:
            Processed DataFrame with market features
        """
        # filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        # if not os.path.exists(filename):
        #     raise FileNotFoundError(f"Historical data file not found: {filename}")
        #
        # # Read the CSV file
        # df = pd.read_csv(filename, index_col=0)

        initial_row_count = len(df)

        # Calculate base features
        df = self.calculate_base_features(df)

        # Calculate lagged features
        df = self.calculate_lagged_features(df)

        # Calculate volatility features
        df = self.calculate_volatility_features(df)

        # Calculate candlestick features if enabled
        if self.include_candlestick:
            df = self.calculate_candlestick_features(df)

        # Calculate volume features if enabled
        if self.include_volume and 'volume' in df.columns:
            df = self.calculate_volume_features(df)

        if len(df) != initial_row_count:
            raise ValueError(f"Row count changed during processing: {initial_row_count} -> {len(df)}")

        # Save back to CSV
        # df.to_csv(filename)
        #
        # print(f"Processed and stored market features for {filename}")

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

                    # Check if market features already exist, calculate if needed
                    if 'MARKET_FEATURES_price_return' not in df.columns:
                        df = self.calculate_base_features(df)
                        df = self.calculate_lagged_features(df)
                        df = self.calculate_volatility_features(df)

                        # Calculate candlestick features if enabled
                        if self.include_candlestick:
                            df = self.calculate_candlestick_features(df)

                        # Calculate volume features if enabled
                        if self.include_volume and 'volume' in df.columns:
                            df = self.calculate_volume_features(df)

                    # Calculate normalization parameters from training data
                    self.fit_normalization_params(df, symbol, interval)

                    # Apply normalization
                    df = self.apply_normalization(df, symbol, interval)
                    processed_data['train'][key] = df

                    # Save normalization parameters if requested
                    if save_params:
                        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_market_params.json"
                        with open(norm_params_filename, 'w') as f:
                            json.dump(self.normalization_params, f, indent=4)
                        print(f"Saved market features normalization parameters for {key}")

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

                            # Check if market features already exist, calculate if needed
                            if 'MARKET_FEATURES_price_return' not in df.columns:
                                df = self.calculate_base_features(df)
                                df = self.calculate_lagged_features(df)
                                df = self.calculate_volatility_features(df)

                                # Calculate candlestick features if enabled
                                if self.include_candlestick:
                                    df = self.calculate_candlestick_features(df)

                                # Calculate volume features if enabled
                                if self.include_volume and 'volume' in df.columns:
                                    df = self.calculate_volume_features(df)

                            # Apply normalization
                            df = self.apply_normalization(df, symbol, interval)
                            processed_data[split][key] = df

        return processed_data

    def calculate_base_features(self, df):
        """
        Calculate base price features: returns, velocity, and acceleration

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added base features
        """
        try:
            # Simple returns: (price_t / price_t-1) - 1
            df['MARKET_FEATURES_price_return'] = df['close'].pct_change()

            # First derivative of returns (velocity)
            df['MARKET_FEATURES_price_velocity'] = df['MARKET_FEATURES_price_return'] - df[
                'MARKET_FEATURES_price_return'].shift(1)

            # Second derivative of returns (acceleration)
            df['MARKET_FEATURES_price_acceleration'] = df['MARKET_FEATURES_price_velocity'] - df[
                'MARKET_FEATURES_price_velocity'].shift(1)

            return df
        except Exception as e:
            print(f"Error MarketFeaturesProcessor: during calculate_base_features: {str(e)}")
            sys.exit(2)

    def calculate_lagged_features(self, df):
        """
        Calculate lagged features for returns, velocity, and acceleration

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with added lagged features
        """
        try:
            for lag in self.lag_periods:
                # Lagged simple returns
                df[f'MARKET_FEATURES_return_lag_{lag}'] = df['MARKET_FEATURES_price_return'].shift(lag)

                # Lagged price velocity
                df[f'MARKET_FEATURES_velocity_lag_{lag}'] = df['MARKET_FEATURES_price_velocity'].shift(lag)

                # Lagged price acceleration
                df[f'MARKET_FEATURES_acceleration_lag_{lag}'] = df['MARKET_FEATURES_price_acceleration'].shift(lag)

            return df
        except Exception as e:
            print(f"Error MarketFeaturesProcessor: during calculate_lagged_features: {str(e)}")
            sys.exit(2)

    def calculate_volatility_features(self, df):
        """
        Calculate volatility features based on different window sizes

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added volatility features
        """
        try:
            for window in self.volatility_windows:
                # Rolling standard deviation of simple returns
                df[f'MARKET_FEATURES_volatility_{window}'] = df['MARKET_FEATURES_price_return'].rolling(window=window).std()

            return df
        except Exception as e:
            print(f"Error MarketFeaturesProcessor: during calculate_volatility_features: {str(e)}")
            sys.exit(2)

    def calculate_candlestick_features(self, df):
        """
        Calculate features based on candlestick relationships

        Args:
            df: DataFrame with OHLC price data

        Returns:
            DataFrame with added candlestick features
        """
        try:
            # Verify OHLC columns exist
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("DataFrame must contain 'open', 'high', 'low', and 'close' columns")

            # 1. Candle body size (close-open) relative to range (high-low)
            # This measures the strength of the directional move
            df['MARKET_FEATURES_candle_body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)

            # 2. Upper shadow relative to range
            # This measures upper price rejection
            df['MARKET_FEATURES_upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (
                        df['high'] - df['low'] + 1e-8)

            # 3. Lower shadow relative to range
            # This measures lower price rejection
            df['MARKET_FEATURES_lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (
                        df['high'] - df['low'] + 1e-8)

            # 4. Close position within range
            # This measures where in the day's range the close occurred (0=low, 1=high)
            df['MARKET_FEATURES_close_position_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

            # 5. Open position within range
            # This measures where in the day's range the open occurred (0=low, 1=high)
            df['MARKET_FEATURES_open_position_ratio'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)

            # 6. Candle direction (1 for bullish, -1 for bearish, 0 for doji)
            df['MARKET_FEATURES_candle_direction'] = np.sign(df['close'] - df['open'])

            # 7. High-Close to High-Low ratio (measures strength at close relative to the high)
            df['MARKET_FEATURES_high_close_strength'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)

            # 8. Close-Low to High-Low ratio (measures bounce from low)
            df['MARKET_FEATURES_close_low_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

            # Add field names to class tracking for normalization
            self.candlestick_fields = [
                'MARKET_FEATURES_candle_body_ratio',
                'MARKET_FEATURES_upper_shadow_ratio',
                'MARKET_FEATURES_lower_shadow_ratio',
                'MARKET_FEATURES_close_position_ratio',
                'MARKET_FEATURES_open_position_ratio',
                'MARKET_FEATURES_high_close_strength',
                'MARKET_FEATURES_close_low_strength'
            ]

            return df
        except Exception as e:
            print(f"Error MarketFeaturesProcessor: during calculate_candlestick_features: {str(e)}")
            sys.exit(2)

    def calculate_volume_features(self, df):
        """
        Calculate features based on volume data

        Args:
            df: DataFrame with OHLCV price data

        Returns:
            DataFrame with added volume features
        """
        try:
            # Verify volume column exists
            if 'volume' not in df.columns:
                raise ValueError("DataFrame must contain a 'volume' column")

            # 1. Relative Volume: Compare current volume to recent average
            for window in self.volume_windows:
                df[f'MARKET_FEATURES_rel_volume_{window}'] = df['volume'] / df['volume'].rolling(
                    window=window).mean().replace(0, np.nan)

            # 2. Volume Trend: Rate of change in volume
            for window in self.volume_windows:
                df[f'MARKET_FEATURES_volume_trend_{window}'] = df['volume'].pct_change(periods=window)

            # 3. Price-Volume Relationship: Correlation between price and volume
            for window in self.volume_windows:
                # Using rolling correlation between returns and volume
                df[f'MARKET_FEATURES_price_vol_corr_{window}'] = df['MARKET_FEATURES_price_return'].rolling(
                    window=window).corr(df['volume'].pct_change())

            # 4. Volume Force: Combines direction and volume (positive for up days, negative for down days)
            df['MARKET_FEATURES_volume_force'] = df['volume'] * np.sign(df['close'] - df['open'])

            # 5. Normalized Volume Force: Volume force relative to recent average volume
            for window in self.volume_windows:
                avg_volume = df['volume'].rolling(window=window).mean().replace(0, np.nan)
                df[f'MARKET_FEATURES_norm_volume_force_{window}'] = df['MARKET_FEATURES_volume_force'] / avg_volume

            # 6. Money Flow: Volume weighted by the position of close within high-low range
            price_position = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-8)
            df['MARKET_FEATURES_money_flow'] = price_position * df['volume']

            # 7. Money Flow Ratio: Positive money flow to negative money flow ratio
            for window in self.volume_windows:
                pos_money_flow = df['MARKET_FEATURES_money_flow'].rolling(window=window).apply(
                    lambda x: sum(i for i in x if i > 0) or 1e-8)
                neg_money_flow = df['MARKET_FEATURES_money_flow'].rolling(window=window).apply(
                    lambda x: abs(sum(i for i in x if i < 0)) or 1e-8)
                df[f'MARKET_FEATURES_money_flow_ratio_{window}'] = pos_money_flow / neg_money_flow

            # 8. On-Balance Volume (OBV): Running sum of volume signed by price direction
            df['MARKET_FEATURES_obv_change'] = df['volume'] * np.where(df['close'] > df['close'].shift(1), 1,
                                                                             np.where(df['close'] < df['close'].shift(1),
                                                                                      -1, 0))
            df['MARKET_FEATURES_obv'] = df['MARKET_FEATURES_obv_change'].cumsum()

            # 9. OBV Slope: Rate of change in OBV
            for window in self.volume_windows:
                df[f'MARKET_FEATURES_obv_slope_{window}'] = (df['MARKET_FEATURES_obv'] - df['MARKET_FEATURES_obv'].shift(
                    window)) / window

            # 10. Volume-Adjusted Returns: Returns weighted by relative volume
            for window in self.volume_windows:
                rel_volume = df['volume'] / df['volume'].rolling(window=window).mean().replace(0, np.nan)
                df[f'MARKET_FEATURES_vol_adj_return_{window}'] = df['MARKET_FEATURES_price_return'] * rel_volume

            # Track volume feature fields for normalization
            self.volume_fields = []

            # Add relative volume fields
            for window in self.volume_windows:
                self.volume_fields.append(f'MARKET_FEATURES_rel_volume_{window}')
                self.volume_fields.append(f'MARKET_FEATURES_volume_trend_{window}')
                self.volume_fields.append(f'MARKET_FEATURES_price_vol_corr_{window}')
                self.volume_fields.append(f'MARKET_FEATURES_norm_volume_force_{window}')
                self.volume_fields.append(f'MARKET_FEATURES_money_flow_ratio_{window}')
                self.volume_fields.append(f'MARKET_FEATURES_obv_slope_{window}')
                self.volume_fields.append(f'MARKET_FEATURES_vol_adj_return_{window}')

            # Add global volume fields
            self.volume_fields.extend(['MARKET_FEATURES_volume_force', 'MARKET_FEATURES_money_flow', 'MARKET_FEATURES_obv'])

            # Replace NaN values with 0 for stability
            df = df.fillna(0)

            return df
        except Exception as e:
            print(f"Error MarketFeaturesProcessor: during calculate_base_features: {str(e)}")
            sys.exit(2)

    def fit_normalization_params(self, df, symbol, interval):
        """
        Calculate normalization parameters from training data

        Args:
            df: DataFrame containing the features to normalize
            symbol: Trading pair symbol
            interval: Timeframe interval
        """
        # Get list of features to normalize
        features_to_normalize = []

        # Add lagged return features
        for lag in self.lag_periods:
            features_to_normalize.append(f'MARKET_FEATURES_return_lag_{lag}')
            features_to_normalize.append(f'MARKET_FEATURES_velocity_lag_{lag}')
            features_to_normalize.append(f'MARKET_FEATURES_acceleration_lag_{lag}')

        # Add volatility features
        for window in self.volatility_windows:
            features_to_normalize.append(f'MARKET_FEATURES_volatility_{window}')

        # Add candlestick features if enabled
        if self.include_candlestick:
            features_to_normalize.extend(self.candlestick_fields)

        # Add volume features if enabled
        if self.include_volume:
            features_to_normalize.extend(self.volume_fields)

        # Initialize symbol-interval parameters if not exist
        symbol_key = f"{symbol}_{interval}"
        if symbol_key not in self.normalization_params:
            self.normalization_params[symbol_key] = {}

        # Calculate and store parameters for each feature
        for feature in features_to_normalize:
            # Skip if feature doesn't exist
            if feature not in df.columns:
                continue

            # Skip initial NaN/zero values for more accurate statistics
            valid_data = df[feature].dropna()

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
        # Generate pairs of features to normalize and their normalized field names
        feature_pairs = []

        # Add lagged return features
        for lag in self.lag_periods:
            feature_pairs.append((f'MARKET_FEATURES_return_lag_{lag}', f'MARKET_FEATURES_return_lag_{lag}_norm'))
            feature_pairs.append((f'MARKET_FEATURES_velocity_lag_{lag}', f'MARKET_FEATURES_velocity_lag_{lag}_norm'))
            feature_pairs.append(
                (f'MARKET_FEATURES_acceleration_lag_{lag}', f'MARKET_FEATURES_acceleration_lag_{lag}_norm'))

        # Add volatility features
        for window in self.volatility_windows:
            feature_pairs.append((f'MARKET_FEATURES_volatility_{window}', f'MARKET_FEATURES_volatility_{window}_norm'))

        # Add candlestick features if enabled
        if self.include_candlestick:
            for field in self.candlestick_fields:
                feature_pairs.append((field, f'{field}_norm'))

        # Add volume features if enabled
        if self.include_volume:
            for field in self.volume_fields:
                feature_pairs.append((field, f'{field}_norm'))

        symbol_key = f"{symbol}_{interval}"

        # Apply normalization to each feature
        for src_field, dest_field in feature_pairs:
            # Skip if source field doesn't exist
            if src_field not in df.columns:
                continue

            if symbol_key in self.normalization_params and src_field in self.normalization_params[symbol_key]:
                mean = self.normalization_params[symbol_key][src_field]["mean"]
                std = self.normalization_params[symbol_key][src_field]["std"]

                # Apply z-score normalization
                df[dest_field] = (df[src_field] - mean) / std
            else:
                # If parameters don't exist, try to load from file
                norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_market_params.json"
                if os.path.exists(norm_params_filename):
                    with open(norm_params_filename, 'r') as f:
                        self.normalization_params = json.load(f)

                    if symbol_key in self.normalization_params and src_field in self.normalization_params[symbol_key]:
                        mean = self.normalization_params[symbol_key][src_field]["mean"]
                        std = self.normalization_params[symbol_key][src_field]["std"]

                        # Apply z-score normalization
                        df[dest_field] = (df[src_field] - mean) / std
                    else:
                        # If still not found, use standard normalization
                        df[dest_field] = (df[src_field] - df[src_field].mean()) / df[src_field].std().clip(lower=1e-8)
                else:
                    # If file doesn't exist, use standard normalization
                    df[dest_field] = (df[src_field] - df[src_field].mean()) / df[src_field].std().clip(lower=1e-8)

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
            norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_market_params.json"
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
        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_market_params.json"

        if symbol_key not in self.normalization_params:
            if os.path.exists(norm_params_filename):
                with open(norm_params_filename, 'r') as f:
                    self.normalization_params = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Normalization parameters not found for {symbol_key}. Process training data first.")

        # Calculate features if they don't exist
        if 'MARKET_FEATURES_price_return' not in df.columns:
            df = self.calculate_base_features(df)
            df = self.calculate_lagged_features(df)
            df = self.calculate_volatility_features(df)

            # Calculate candlestick features if enabled
            if self.include_candlestick:
                df = self.calculate_candlestick_features(df)

            # Calculate volume features if enabled
            if self.include_volume and 'volume' in df.columns:
                df = self.calculate_volume_features(df)

        # Apply normalization using stored parameters
        df = self.apply_normalization(df, symbol, interval)

        return df

    def get_feature_names(self, include_normalized=True):
        """
        Get list of feature names generated by this processor

        Args:
            include_normalized: Whether to include normalized feature names

        Returns:
            List of feature names
        """
        base_features = []

        # Add base and lagged return features
        base_features.extend(
            ['MARKET_FEATURES_price_return', 'MARKET_FEATURES_price_velocity', 'MARKET_FEATURES_price_acceleration'])
        for lag in self.lag_periods:
            base_features.append(f'MARKET_FEATURES_return_lag_{lag}')
            base_features.append(f'MARKET_FEATURES_velocity_lag_{lag}')
            base_features.append(f'MARKET_FEATURES_acceleration_lag_{lag}')

        # Add volatility features
        for window in self.volatility_windows:
            base_features.append(f'MARKET_FEATURES_volatility_{window}')

        # Add candlestick features if enabled
        if self.include_candlestick:
            base_features.extend(self.candlestick_fields)

        # Add volume features if enabled
        if self.include_volume:
            base_features.extend(self.volume_fields)

        if include_normalized:
            normalized_features = []
            # Add normalized lagged return features
            for lag in self.lag_periods:
                normalized_features.append(f'MARKET_FEATURES_return_lag_{lag}_norm')
                normalized_features.append(f'MARKET_FEATURES_velocity_lag_{lag}_norm')
                normalized_features.append(f'MARKET_FEATURES_acceleration_lag_{lag}_norm')

            # Add normalized volatility features
            for window in self.volatility_windows:
                normalized_features.append(f'MARKET_FEATURES_volatility_{window}_norm')

            # Add normalized candlestick features if enabled
            if self.include_candlestick:
                for field in self.candlestick_fields:
                    normalized_features.append(f'{field}_norm')

            # Add normalized volume features if enabled
            if self.include_volume:
                for field in self.volume_fields:
                    normalized_features.append(f'{field}_norm')

            return base_features + normalized_features

        return base_features
