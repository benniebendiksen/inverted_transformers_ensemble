from src.Config import Config
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List


class HorizonAlignedIndicatorsProcessor:
    def __init__(self, data_dir: Path,
                 forecast_steps: int = Config.FORECAST_STEPS,
                 multiples: List[int] = None,
                 include_moving_averages: bool = True,
                 include_bands: bool = True,
                 include_momentum: bool = True,
                 include_virtual_candles: bool = True):
        """
        Initialize Horizon-Aligned Indicators processor with parameters

        Args:
            data_dir: Directory containing historical data CSV files
            forecast_steps: Number of steps ahead to forecast (e.g., 4 for 1-hour forecast with 15-min data)
            multiples: List of multiples of forecast_steps to use (e.g., [1, 2, 4, 8])
            include_moving_averages: Whether to include various moving averages
            include_bands: Whether to include support/resistance bands
            include_momentum: Whether to include momentum indicators
            include_virtual_candles: Whether to include virtualized candlestick features
        """
        self.data_dir = data_dir
        self.forecast_steps = forecast_steps

        # Set default multiples if not provided
        self.multiples = multiples if multiples is not None else [1, 2, 4, 8]

        # Generate periods based on forecast_steps and multiples
        self.periods = [self.forecast_steps * multiple for multiple in self.multiples]

        # Feature inclusion flags
        self.include_moving_averages = include_moving_averages
        self.include_bands = include_bands
        self.include_momentum = include_momentum
        self.include_virtual_candles = include_virtual_candles

        # Lists to store field names for each indicator type
        self.moving_average_fields = []
        self.band_fields = []
        self.momentum_fields = []
        self.virtual_candle_fields = []

        # Dictionary to store normalization parameters
        self.normalization_params = {}

    def process_csv(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Process a single CSV file to add horizon-aligned indicators without normalization.

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '15m', '1h')

        Returns:
            Processed DataFrame with horizon-aligned indicators
        """
        filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Historical data file not found: {filename}")

        # Read the CSV file
        df = pd.read_csv(filename, index_col=0)

        # Calculate virtual candle features first (other indicators may use them)
        if self.include_virtual_candles:
            df = self.calculate_virtual_candles(df)

        # Calculate indicators based on inclusion flags
        if self.include_moving_averages:
            df = self.calculate_moving_averages_and_regression(df)

        if self.include_bands:
            df = self.calculate_bands(df)

        if self.include_momentum:
            df = self.calculate_momentum(df)

        # Save back to CSV
        df.to_csv(filename)

        print(f"Processed and stored horizon-aligned indicators for {filename}")

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

                    # Calculate indicators if they don't exist
                    df = self._ensure_indicators_exist(df)

                    # Calculate normalization parameters from training data
                    self.fit_normalization_params(df, symbol, interval)

                    # Apply normalization
                    df = self.apply_normalization(df, symbol, interval)
                    processed_data['train'][key] = df

                    # Save normalization parameters if requested
                    if save_params:
                        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_horizon_params.json"
                        with open(norm_params_filename, 'w') as f:
                            json.dump(self.normalization_params, f, indent=4)
                        print(f"Saved horizon indicator normalization parameters for {key}")

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

                            # Calculate indicators if they don't exist
                            df = self._ensure_indicators_exist(df)

                            # Apply normalization
                            df = self.apply_normalization(df, symbol, interval)
                            processed_data[split][key] = df

        return processed_data

    def _ensure_indicators_exist(self, df):
        """
        Ensure all selected indicators exist in the DataFrame, calculate if needed

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all required indicators
        """
        # Calculate virtual candle features first (other indicators may depend on them)
        if self.include_virtual_candles and not any(col.startswith('HORIZON_VC_') for col in df.columns):
            df = self.calculate_virtual_candles(df)

        # Calculate indicators if they don't exist
        if self.include_moving_averages and not any(col.startswith('HORIZON_MA_') for col in df.columns):
            df = self.calculate_moving_averages_and_regression(df)

        if self.include_bands and not any(col.startswith('HORIZON_BAND_') for col in df.columns):
            df = self.calculate_bands(df)

        if self.include_momentum and not any(col.startswith('HORIZON_MOM_') for col in df.columns):
            df = self.calculate_momentum(df)

        return df

    def calculate_moving_averages_and_regression(self, df):
        """
        Calculate various moving averages and linear regression estimates aligned with forecast horizon

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with moving average and regression indicators added
        """
        # Reset moving average fields list
        self.moving_average_fields = []

        # Calculate moving averages for each period
        for period in self.periods:
            prefix = f"HORIZON_MA_{period}"

            # Simple Moving Average (SMA)
            df[f'{prefix}_SMA'] = df['close'].rolling(window=period).mean()

            # Exponential Moving Average (EMA)
            df[f'{prefix}_EMA'] = df['close'].ewm(span=period, adjust=False).mean()

            # Weighted Moving Average (WMA)
            weights = np.arange(1, period + 1)
            df[f'{prefix}_WMA'] = df['close'].rolling(window=period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )

            # Hull Moving Average (HMA)
            wma_half_period = df['close'].rolling(window=period // 2).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True
            )
            wma_full_period = df['close'].rolling(window=period).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True
            )
            df[f'{prefix}_HMA'] = (2 * wma_half_period - wma_full_period).rolling(window=int(np.sqrt(period))).mean()

            # Distance from price to moving averages (percentage)
            df[f'{prefix}_SMA_DIST'] = (df['close'] - df[f'{prefix}_SMA']) / df['close'] * 100
            df[f'{prefix}_EMA_DIST'] = (df['close'] - df[f'{prefix}_EMA']) / df['close'] * 100

            # SMA-EMA spread (percentage)
            df[f'{prefix}_SPREAD'] = (df[f'{prefix}_SMA'] - df[f'{prefix}_EMA']) / df[f'{prefix}_SMA'] * 100

            # --- Linear Regression Features ---

            # Calculate slope directly (returns scalar)
            def calc_slope(x):
                if len(x) < 2:
                    return 0
                # Handle both Series and ndarray input
                y = x if isinstance(x, np.ndarray) else x.values
                x_vals = np.arange(len(y))
                slope, _ = np.polyfit(x_vals, y, 1)
                return slope

            # Calculate intercept directly (returns scalar)
            def calc_intercept(x):
                if len(x) < 2:
                    # Handle both Series and ndarray input for the initial value
                    return x[0] if isinstance(x, np.ndarray) else (x.iloc[0] if len(x) > 0 else 0)
                y = x if isinstance(x, np.ndarray) else x.values
                x_vals = np.arange(len(y))
                _, intercept = np.polyfit(x_vals, y, 1)
                return intercept

            # Apply rolling linear regression and extract parameters directly
            df[f'{prefix}_LR_SLOPE'] = df['close'].rolling(window=period).apply(calc_slope, raw=False)
            df[f'{prefix}_LR_INTERCEPT'] = df['close'].rolling(window=period).apply(calc_intercept, raw=False)

            # Calculate the linear regression value at the current point
            df[f'{prefix}_LR_VALUE'] = df[f'{prefix}_LR_INTERCEPT'] + df[f'{prefix}_LR_SLOPE'] * (period - 1)

            # Calculate the forecast value for the next forecast_steps
            df[f'{prefix}_LR_FORECAST'] = df[f'{prefix}_LR_INTERCEPT'] + df[f'{prefix}_LR_SLOPE'] * (
                    period - 1 + self.forecast_steps)

            # Calculate expected percentage change based on linear regression
            df[f'{prefix}_LR_CHANGE_PCT'] = (df[f'{prefix}_LR_FORECAST'] - df['close']) / df['close'] * 100

            # Calculate R-squared to measure trend strength
            def calc_rsquared(x):
                if len(x) < 2:
                    return 0
                # When raw=True, x is already a numpy array
                # When raw=False, x is a pandas Series and needs .values
                y = x if isinstance(x, np.ndarray) else x.values
                x_vals = np.arange(len(y))
                slope, intercept = np.polyfit(x_vals, y, 1)
                y_pred = slope * x_vals + intercept
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                return 1 - (ss_residual / ss_total) if ss_total != 0 else 0

            df[f'{prefix}_LR_R2'] = df['close'].rolling(window=period).apply(
                calc_rsquared, raw=True
            )

            # Direction of the trend (1 for up, -1 for down, 0 for flat)
            df[f'{prefix}_LR_DIRECTION'] = np.sign(df[f'{prefix}_LR_SLOPE'])

            # Trend acceleration (change in slope)
            df[f'{prefix}_LR_ACCEL'] = df[f'{prefix}_LR_SLOPE'].diff()

            # Distance from current price to regression line (percentage)
            df[f'{prefix}_LR_DIST'] = (df['close'] - df[f'{prefix}_LR_VALUE']) / df['close'] * 100

            # Add regression fields to list
            self.moving_average_fields.extend([
                f'{prefix}_LR_SLOPE', f'{prefix}_LR_INTERCEPT', f'{prefix}_LR_VALUE',
                f'{prefix}_LR_FORECAST', f'{prefix}_LR_CHANGE_PCT', f'{prefix}_LR_R2',
                f'{prefix}_LR_DIRECTION', f'{prefix}_LR_ACCEL', f'{prefix}_LR_DIST'
            ])

            # Add original fields to list
            self.moving_average_fields.extend([
                f'{prefix}_SMA', f'{prefix}_EMA', f'{prefix}_WMA', f'{prefix}_HMA',
                f'{prefix}_SMA_DIST', f'{prefix}_EMA_DIST', f'{prefix}_SPREAD'
            ])

            # Calculate crossovers
            df[f'{prefix}_CROSS'] = 0
            # Price crossing SMA
            df.loc[(df['close'] > df[f'{prefix}_SMA']) & (df['close'].shift(1) <= df[f'{prefix}_SMA'].shift(1)),
            f'{prefix}_CROSS'] = 1  # Bullish
            df.loc[(df['close'] < df[f'{prefix}_SMA']) & (df['close'].shift(1) >= df[f'{prefix}_SMA'].shift(1)),
            f'{prefix}_CROSS'] = -1  # Bearish

            self.moving_average_fields.append(f'{prefix}_CROSS')

        return df

    def calculate_bands(self, df):
        """
        Calculate support and resistance bands aligned with forecast horizon

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with band indicators added
        """
        # Reset band fields list
        self.band_fields = []

        # Calculate bands for each period
        for period in self.periods:
            prefix = f"HORIZON_BAND_{period}"

            # Bollinger Bands
            middle = df['close'].rolling(window=period).mean()
            std_dev = df['close'].rolling(window=period).std()
            df[f'{prefix}_BB_UPPER'] = middle + (2 * std_dev)
            df[f'{prefix}_BB_MIDDLE'] = middle
            df[f'{prefix}_BB_LOWER'] = middle - (2 * std_dev)

            # Bollinger Band Width (volatility indicator)
            df[f'{prefix}_BB_WIDTH'] = (df[f'{prefix}_BB_UPPER'] - df[f'{prefix}_BB_LOWER']) / df[
                f'{prefix}_BB_MIDDLE'] * 100

            # Channel features based on period high/low
            df[f'{prefix}_HIGH'] = df['high'].rolling(window=period).max()
            df[f'{prefix}_LOW'] = df['low'].rolling(window=period).min()
            df[f'{prefix}_CHAN_WIDTH'] = (df[f'{prefix}_HIGH'] - df[f'{prefix}_LOW']) / df['close'] * 100

            # Distance to upper and lower bands (percentage)
            df[f'{prefix}_BB_UPPER_DIST'] = (df[f'{prefix}_BB_UPPER'] - df['close']) / df['close'] * 100
            df[f'{prefix}_BB_LOWER_DIST'] = (df['close'] - df[f'{prefix}_BB_LOWER']) / df['close'] * 100

            # Distance to period high/low (percentage)
            df[f'{prefix}_HIGH_DIST'] = (df[f'{prefix}_HIGH'] - df['close']) / df['close'] * 100
            df[f'{prefix}_LOW_DIST'] = (df['close'] - df[f'{prefix}_LOW']) / df['close'] * 100

            # Relative position within the band (0 = at lower band, 1 = at upper band)
            bb_range = df[f'{prefix}_BB_UPPER'] - df[f'{prefix}_BB_LOWER']
            df[f'{prefix}_BB_POS'] = (df['close'] - df[f'{prefix}_BB_LOWER']) / bb_range.replace(0, np.nan)

            # Channel position
            chan_range = df[f'{prefix}_HIGH'] - df[f'{prefix}_LOW']
            df[f'{prefix}_CHAN_POS'] = (df['close'] - df[f'{prefix}_LOW']) / chan_range.replace(0, np.nan)

            # Add fields to list
            self.band_fields.extend([
                f'{prefix}_BB_UPPER', f'{prefix}_BB_MIDDLE', f'{prefix}_BB_LOWER',
                f'{prefix}_BB_WIDTH', f'{prefix}_HIGH', f'{prefix}_LOW',
                f'{prefix}_CHAN_WIDTH', f'{prefix}_BB_UPPER_DIST', f'{prefix}_BB_LOWER_DIST',
                f'{prefix}_HIGH_DIST', f'{prefix}_LOW_DIST', f'{prefix}_BB_POS',
                f'{prefix}_CHAN_POS'
            ])

        return df

    def calculate_momentum(self, df):
        """
        Calculate momentum indicators aligned with forecast horizon

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with momentum indicators added
        """
        # Reset momentum fields list
        self.momentum_fields = []

        # Calculate momentum indicators for each period
        for period in self.periods:
            prefix = f"HORIZON_MOM_{period}"

            # Price Rate of Change (ROC)
            df[f'{prefix}_ROC'] = ((df['close'] / df['close'].shift(period)) - 1) * 100

            # Momentum: Current price - price 'period' periods ago
            df[f'{prefix}_RAW'] = df['close'] - df['close'].shift(period)

            # Acceleration: Change in momentum
            df[f'{prefix}_ACC'] = df[f'{prefix}_ROC'] - df[f'{prefix}_ROC'].shift(1)

            # Moving Average Convergence Divergence (MACD) adjusted to forecast horizon
            # Use period as slow EMA and period/2 as fast EMA
            fast_ema = df['close'].ewm(span=period // 2, adjust=False).mean()
            slow_ema = df['close'].ewm(span=period, adjust=False).mean()
            df[f'{prefix}_MACD'] = fast_ema - slow_ema
            df[f'{prefix}_SIGNAL'] = df[f'{prefix}_MACD'].ewm(span=period // 4, adjust=False).mean()
            df[f'{prefix}_HIST'] = df[f'{prefix}_MACD'] - df[f'{prefix}_SIGNAL']

            # Relative Strength Index (RSI) for the specific period
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'{prefix}_RSI'] = 100 - (100 / (1 + rs))

            # Add fields to list
            self.momentum_fields.extend([
                f'{prefix}_ROC', f'{prefix}_RAW', f'{prefix}_ACC',
                f'{prefix}_MACD', f'{prefix}_SIGNAL', f'{prefix}_HIST',
                f'{prefix}_RSI'
            ])

            # RSI state
            df[f'{prefix}_RSI_STATE'] = 0
            df.loc[df[f'{prefix}_RSI'] > 70, f'{prefix}_RSI_STATE'] = 2  # Overbought
            df.loc[df[f'{prefix}_RSI'] < 30, f'{prefix}_RSI_STATE'] = -2  # Oversold
            df.loc[(df[f'{prefix}_RSI'] >= 50) & (df[f'{prefix}_RSI'] <= 70), f'{prefix}_RSI_STATE'] = 1  # Bullish
            df.loc[(df[f'{prefix}_RSI'] >= 30) & (df[f'{prefix}_RSI'] < 50), f'{prefix}_RSI_STATE'] = -1  # Bearish

            self.momentum_fields.append(f'{prefix}_RSI_STATE')

        return df

    def calculate_virtual_candles(self, df):
        """
        Calculate features derived from virtualized horizon-aligned candlesticks

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with virtual candle-derived features added
        """
        # Reset virtual candle fields list
        self.virtual_candle_fields = []

        # Calculate virtual candle features for each period
        for period in self.periods:
            prefix = f"HORIZON_VC_{period}"

            # Virtual candle OHLC values
            df[f'{prefix}_OPEN'] = df['open'].shift(period - 1)  # Open from period-steps ago
            df[f'{prefix}_HIGH'] = df['high'].rolling(window=period).max()  # Highest high in the period
            df[f'{prefix}_LOW'] = df['low'].rolling(window=period).min()  # Lowest low in the period
            df[f'{prefix}_CLOSE'] = df['close']  # Current close

            # Virtual candle range metrics
            df[f'{prefix}_RANGE'] = df[f'{prefix}_HIGH'] - df[f'{prefix}_LOW']
            df[f'{prefix}_RANGE_PCT'] = df[f'{prefix}_RANGE'] / df[f'{prefix}_OPEN'] * 100

            # Body size and direction
            df[f'{prefix}_BODY'] = abs(df[f'{prefix}_CLOSE'] - df[f'{prefix}_OPEN'])
            df[f'{prefix}_BODY_PCT'] = df[f'{prefix}_BODY'] / df[f'{prefix}_OPEN'] * 100
            df[f'{prefix}_DIRECTION'] = np.sign(df[f'{prefix}_CLOSE'] - df[f'{prefix}_OPEN'])

            # Upper and lower shadows
            upper_price = df[['close', 'open']].max(axis=1)
            lower_price = df[['close', 'open']].min(axis=1)
            df[f'{prefix}_UPPER_SHADOW'] = df[f'{prefix}_HIGH'] - upper_price
            df[f'{prefix}_LOWER_SHADOW'] = lower_price - df[f'{prefix}_LOW']
            df[f'{prefix}_UPPER_SHADOW_PCT'] = df[f'{prefix}_UPPER_SHADOW'] / df[f'{prefix}_OPEN'] * 100
            df[f'{prefix}_LOWER_SHADOW_PCT'] = df[f'{prefix}_LOWER_SHADOW'] / df[f'{prefix}_OPEN'] * 100

            # Price position within virtual candle range
            df[f'{prefix}_POS'] = (df['close'] - df[f'{prefix}_LOW']) / df[f'{prefix}_RANGE'].replace(0, np.nan)

            # Virtual candle pattern indicators
            # Engulfing patterns
            df[f'{prefix}_ENGULFING'] = 0
            # Bullish engulfing
            bullish_cond = (df[f'{prefix}_DIRECTION'] > 0) & \
                           (df[f'{prefix}_OPEN'] < df[f'{prefix}_CLOSE'].shift(1)) & \
                           (df[f'{prefix}_CLOSE'] > df[f'{prefix}_OPEN'].shift(1))
            df.loc[bullish_cond, f'{prefix}_ENGULFING'] = 1

            # Bearish engulfing
            bearish_cond = (df[f'{prefix}_DIRECTION'] < 0) & \
                           (df[f'{prefix}_OPEN'] > df[f'{prefix}_CLOSE'].shift(1)) & \
                           (df[f'{prefix}_CLOSE'] < df[f'{prefix}_OPEN'].shift(1))
            df.loc[bearish_cond, f'{prefix}_ENGULFING'] = -1

            # Doji detection (body less than 10% of range) - Convert to integer
            doji_threshold = 0.1
            df[f'{prefix}_DOJI'] = ((df[f'{prefix}_BODY'] / df[f'{prefix}_RANGE']) < doji_threshold).astype(int)

            # Hammer/Hanging Man (small body, long lower shadow, small upper shadow) - Convert to integer
            hammer_cond = (df[f'{prefix}_BODY'] / df[f'{prefix}_RANGE'] < 0.3) & \
                          (df[f'{prefix}_LOWER_SHADOW'] / df[f'{prefix}_RANGE'] > 0.6) & \
                          (df[f'{prefix}_UPPER_SHADOW'] / df[f'{prefix}_RANGE'] < 0.1)
            df[f'{prefix}_HAMMER'] = hammer_cond.astype(int)

            # Shooting Star/Inverted Hammer (small body, long upper shadow, small lower shadow) - Convert to integer
            star_cond = (df[f'{prefix}_BODY'] / df[f'{prefix}_RANGE'] < 0.3) & \
                        (df[f'{prefix}_UPPER_SHADOW'] / df[f'{prefix}_RANGE'] > 0.6) & \
                        (df[f'{prefix}_LOWER_SHADOW'] / df[f'{prefix}_RANGE'] < 0.1)
            df[f'{prefix}_SHOOTING_STAR'] = star_cond.astype(int)

            # Add fields to list first (before conditional volume section)
            self.virtual_candle_fields.extend([
                f'{prefix}_OPEN', f'{prefix}_HIGH', f'{prefix}_LOW', f'{prefix}_CLOSE',
                f'{prefix}_RANGE', f'{prefix}_RANGE_PCT', f'{prefix}_BODY', f'{prefix}_BODY_PCT',
                f'{prefix}_DIRECTION', f'{prefix}_UPPER_SHADOW', f'{prefix}_LOWER_SHADOW',
                f'{prefix}_UPPER_SHADOW_PCT', f'{prefix}_LOWER_SHADOW_PCT', f'{prefix}_POS',
                f'{prefix}_ENGULFING', f'{prefix}_DOJI', f'{prefix}_HAMMER', f'{prefix}_SHOOTING_STAR'
            ])

            # Volume concentration - only if volume exists in the dataframe
            if 'quote_volume' in df.columns:
                # Calculate relative volume compared to the average over the period
                df[f'{prefix}_REL_VOLUME'] = df['quote_volume'] / df['quote_volume'].rolling(window=period).mean()

                # Calculate volume-weighted average price (VWAP) for the virtual candle period
                # Modified approach to avoid the error
                def calculate_vwap(prices, volumes):
                    if sum(volumes) > 0:
                        return sum(prices * volumes) / sum(volumes)
                    return sum(prices) / len(prices) if len(prices) > 0 else 0

                vwap_values = []
                for i in range(len(df)):
                    if i < period - 1:
                        vwap_values.append(df['close'].iloc[i])  # Use close price for initial values
                    else:
                        price_window = df['close'].iloc[i - period + 1:i + 1].values
                        volume_window = df['quote_volume'].iloc[i - period + 1:i + 1].values
                        vwap_values.append(calculate_vwap(price_window, volume_window))

                df[f'{prefix}_VWAP'] = vwap_values

                # Distance from current price to VWAP (percentage)
                df[f'{prefix}_VWAP_DIST'] = (df['close'] - df[f'{prefix}_VWAP']) / df['close'] * 100

                # Add volume fields to list
                self.virtual_candle_fields.extend([
                    f'{prefix}_REL_VOLUME', f'{prefix}_VWAP', f'{prefix}_VWAP_DIST'
                ])

        return df

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

        # Add all continuous indicator fields (exclude categorical/signal fields)
        for field in self.moving_average_fields:
            if '_CROSS' not in field:  # Skip crossover signals
                features_to_normalize.append(field)

        for field in self.band_fields:
            features_to_normalize.append(field)

        for field in self.momentum_fields:
            if '_STATE' not in field:  # Skip state signals
                features_to_normalize.append(field)

        # Add virtual candle fields (excluding pattern indicators which are categorical)
        for field in self.virtual_candle_fields:
            if not any(x in field for x in ['_ENGULFING', '_DOJI', '_HAMMER', '_SHOOTING_STAR', '_DIRECTION']):
                features_to_normalize.append(field)

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
        symbol_key = f"{symbol}_{interval}"

        # Apply normalization to each feature that has parameters
        if symbol_key in self.normalization_params:
            for src_field, params in self.normalization_params[symbol_key].items():
                # Skip if source field doesn't exist
                if src_field not in df.columns:
                    continue

                # Define destination field name
                dest_field = f"{src_field}_norm"

                # Get parameters
                mean = params["mean"]
                std = params["std"]

                # Apply z-score normalization
                df[dest_field] = (df[src_field] - mean) / std
        else:
            # If parameters don't exist, try to load from file
            norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_horizon_params.json"
            if os.path.exists(norm_params_filename):
                with open(norm_params_filename, 'r') as f:
                    self.normalization_params = json.load(f)

                # Retry normalization
                return self.apply_normalization(df, symbol, interval)
            else:
                print(f"Warning: Normalization parameters not found for {symbol_key}.")

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
            norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_horizon_params.json"
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
            interval: Timeframe interval (e.g., '15m', '1h')

        Returns:
            Processed DataFrame with normalized features ready for prediction
        """
        # Check if we have normalization parameters
        symbol_key = f"{symbol}_{interval}"
        norm_params_filename = self.data_dir / f"{symbol.lower()}_{interval}_horizon_params.json"

        if symbol_key not in self.normalization_params:
            if os.path.exists(norm_params_filename):
                with open(norm_params_filename, 'r') as f:
                    self.normalization_params = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Normalization parameters not found for {symbol_key}. Process training data first.")

        # Calculate indicators if they don't exist
        df = self._ensure_indicators_exist(df)

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

        # Add all indicator fields
        base_features.extend(self.moving_average_fields)
        base_features.extend(self.band_fields)
        base_features.extend(self.momentum_fields)
        base_features.extend(self.virtual_candle_fields)

        if include_normalized:
            normalized_features = []

            # Add normalized versions of continuous fields
            for field in base_features:
                if not any(x in field for x in
                           ['_CROSS', '_STATE', '_ENGULFING', '_DOJI', '_HAMMER', '_SHOOTING_STAR', '_DIRECTION']):
                    normalized_features.append(f"{field}_norm")

            return base_features + normalized_features

        return base_features
