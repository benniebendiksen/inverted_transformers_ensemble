import sys
from src.Config import Config
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List


class HorizonAlignedIndicatorsProcessor:
    def __init__(self, data_dir: Path,
                 forecast_steps: int = 1,  # Default to 1 for single-step forecasting
                 multiples: List[int] = None,
                 include_moving_averages: bool = True,
                 include_bands: bool = True,
                 include_momentum: bool = True,
                 include_virtual_candles: bool = True):
        """
        Initialize Horizon-Aligned Indicators processor with parameters
        Optimized for single-step forecasting (FORECAST_STEPS=1)

        Args:
            data_dir: Directory containing historical data CSV files
            forecast_steps: Number of steps ahead to forecast (default: 1 for single-step)
            multiples: List of multiples of forecast_steps to use
            include_moving_averages: Whether to include various moving averages
            include_bands: Whether to include support/resistance bands
            include_momentum: Whether to include momentum indicators
            include_virtual_candles: Whether to include virtualized candlestick features
        """
        self.data_dir = data_dir
        self.forecast_steps = forecast_steps

        # Ensure forecast_steps is at least 1
        if self.forecast_steps < 1:
            self.forecast_steps = 1
            print("Warning: FORECAST_STEPS must be at least 1. Setting to 1.")

        # Set default multiples if not provided - adjusted for single-step forecasting
        self.multiples = multiples if multiples is not None else [1, 2, 4, 8, 16]

        # Generate lookback periods based on multiples
        # These will be used for various indicator calculations
        self.periods = [self.forecast_steps * multiple for multiple in self.multiples]

        # Ensure we don't have periods of 0
        self.periods = [max(p, 1) for p in self.periods]

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

    def process_csv(self, symbol: str, interval: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single CSV file to add horizon-aligned indicators without normalization.

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '15m', '1h')

        Returns:
            Processed DataFrame with horizon-aligned indicators
        """
        # filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        # if not os.path.exists(filename):
        #     raise FileNotFoundError(f"Historical data file not found: {filename}")
        #
        # # Read the CSV file
        # df = pd.read_csv(filename, index_col=0)

        initial_row_count = len(df)

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

        if len(df) != initial_row_count:
            raise ValueError(f"Row count changed during processing: {initial_row_count} -> {len(df)}")

        # # Save back to CSV
        # df.to_csv(filename)
        #
        # print(f"Processed and stored horizon-aligned indicators for {filename}")

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
        Calculate various moving averages and linear regression estimates aligned with forecast horizon.
        Modified for single-step forecasting with reduced redundancy.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with moving average and regression indicators added
        """
        # Reset moving average fields list
        self.moving_average_fields = []

        try:
            # Calculate moving averages for each period
            for period in self.periods:
                prefix = f"HORIZON_ALIGNED_MA_{period}"
                # For periods less than 3, we need special handling
                if period < 3:
                    period = 3  # Minimum period for reliable calculations

                # Calculate the MA types that aren't redundant with other processors

                # ===== Future-oriented change metrics =====
                # Calculate percent change of close to future_close if forecasting is purpose
                df[f'{prefix}_FUTURE_BIAS'] = (df['close'].shift(-self.forecast_steps) / df['close'] - 1) * 100

                # ===== Triangular Moving Average (not in other processors) =====
                # TMA is smoother than SMA and not in other processors
                sma = df['close'].rolling(window=period).mean()
                df[f'{prefix}_TMA'] = sma.rolling(window=period).mean()

                # ===== Price Position Relative to Moving Averages =====
                # Calculate price position within bands formed by fast and slow moving averages
                fast_period = max(3, period // 2)  # Ensure minimum of 3
                slow_period = period

                fast_ma = df['close'].rolling(window=fast_period).mean()
                slow_ma = df['close'].rolling(window=slow_period).mean()

                # Position of price within the band formed by fast and slow MAs
                ma_diff = fast_ma - slow_ma
                df[f'{prefix}_MA_BAND_POS'] = (df['close'] - slow_ma) / ma_diff.replace(0, np.nan)

                # ===== Moving Average Convergence/Divergence based on adaptive periods =====
                # Use larger periods for MACD to avoid redundancy with standard MACD processor
                # and address the original error by ensuring minimum periods
                macd_fast = max(5, period // 2)
                macd_slow = max(10, period)
                macd_signal = max(3, period // 3)

                fast_ema = df['close'].ewm(span=macd_fast, adjust=False).mean()
                slow_ema = df['close'].ewm(span=macd_slow, adjust=False).mean()
                df[f'{prefix}_MACD'] = fast_ema - slow_ema
                df[f'{prefix}_SIGNAL'] = df[f'{prefix}_MACD'].ewm(span=macd_signal).mean()
                df[f'{prefix}_HIST'] = df[f'{prefix}_MACD'] - df[f'{prefix}_SIGNAL']

                # ===== Rate of Change of Volume (not in other processors) =====
                if 'quote_volume' in df.columns:
                    df[f'{prefix}_VOLUME_ROC'] = df['quote_volume'].pct_change(periods=period) * 100

                # ===== Adapted Linear Regression Features for Short-term Forecasting =====
                # For single-step forecasting, focus on short-term regression metrics
                if period <= 16:  # Only for reasonably short lookbacks to avoid redundancy
                    # Calculate slope directly
                    def calc_slope(x):
                        if len(x) < 2:
                            return 0
                        y = x if isinstance(x, np.ndarray) else x.values
                        x_vals = np.arange(len(y))
                        slope, _ = np.polyfit(x_vals, y, 1)
                        return slope

                    # Calculate R-squared to measure trend strength
                    def calc_rsquared(x):
                        if len(x) < 2:
                            return 0
                        y = x if isinstance(x, np.ndarray) else x.values
                        x_vals = np.arange(len(y))
                        slope, intercept = np.polyfit(x_vals, y, 1)
                        y_pred = slope * x_vals + intercept
                        ss_total = np.sum((y - np.mean(y)) ** 2)
                        ss_residual = np.sum((y - y_pred) ** 2)
                        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0

                    # Calculate the linear regression slope over this period
                    df[f'{prefix}_LR_SLOPE'] = df['close'].rolling(window=period).apply(
                        calc_slope, raw=False
                    )

                    # Calculate the R-squared value to measure trend strength
                    df[f'{prefix}_LR_R2'] = df['close'].rolling(window=period).apply(
                        calc_rsquared, raw=True
                    )

                    # Convert slope to percentage for easier interpretation
                    mean_price = df['close'].rolling(window=period).mean()
                    df[f'{prefix}_LR_SLOPE_PCT'] = df[f'{prefix}_LR_SLOPE'] / mean_price * 100

                    # Add regression fields to list
                    self.moving_average_fields.extend([
                        f'{prefix}_LR_SLOPE', f'{prefix}_LR_R2', f'{prefix}_LR_SLOPE_PCT'
                    ])

                # Add all calculated fields to the features list
                self.moving_average_fields.extend([
                    f'{prefix}_FUTURE_BIAS',
                    f'{prefix}_TMA',
                    f'{prefix}_MA_BAND_POS',
                    f'{prefix}_MACD',
                    f'{prefix}_SIGNAL',
                    f'{prefix}_HIST'
                ])

                if 'quote_volume' in df.columns:
                    self.moving_average_fields.append(f'{prefix}_VOLUME_ROC')

            return df
        except Exception as e:
            print(f"Error in calculate_moving_averages_and_regression: {str(e)}")
            return df

    def calculate_bands(self, df):
        """
        Calculate support and resistance bands aligned with forecast horizon
        Modified to reduce redundancy with Bollinger Bands processor

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with band indicators added
        """
        # Reset band fields list
        self.band_fields = []
        try:
            # Calculate bands for each period
            for period in self.periods:
                prefix = f"HORIZON_ALIGNED_BAND_{period}"

                # Use minimum period of 3 for calculations
                actual_period = max(3, period)

                # ===== Donchian Channels (not in other processors) =====
                # Donchian channels use high/low values and differ from Bollinger Bands
                df[f'{prefix}_DONCH_UPPER'] = df['high'].rolling(window=actual_period).max()
                df[f'{prefix}_DONCH_LOWER'] = df['low'].rolling(window=actual_period).min()
                df[f'{prefix}_DONCH_MID'] = (df[f'{prefix}_DONCH_UPPER'] + df[f'{prefix}_DONCH_LOWER']) / 2

                # ===== Keltner Channels (not in other processors) =====
                # Use Average True Range for Keltner Channel width
                tr1 = abs(df['high'] - df['low'])
                tr2 = abs(df['high'] - df['close'].shift(1))
                tr3 = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=actual_period).mean()

                ema = df['close'].ewm(span=actual_period, adjust=False).mean()
                df[f'{prefix}_KELT_UPPER'] = ema + (2 * atr)
                df[f'{prefix}_KELT_LOWER'] = ema - (2 * atr)
                df[f'{prefix}_KELT_MID'] = ema

                # ===== Relative Position within Channels =====
                # Position within Donchian channel (0=lower, 1=upper)
                donch_range = df[f'{prefix}_DONCH_UPPER'] - df[f'{prefix}_DONCH_LOWER']
                df[f'{prefix}_DONCH_POS'] = (df['close'] - df[f'{prefix}_DONCH_LOWER']) / donch_range.replace(0, np.nan)

                # Position within Keltner channel (0=lower, 1=upper)
                kelt_range = df[f'{prefix}_KELT_UPPER'] - df[f'{prefix}_KELT_LOWER']
                df[f'{prefix}_KELT_POS'] = (df['close'] - df[f'{prefix}_KELT_LOWER']) / kelt_range.replace(0, np.nan)

                # ===== Channel Comparisons (unique to this processor) =====
                # Relative width comparison between Donchian and Keltner (high values indicate volatility)
                df[f'{prefix}_CHAN_WIDTH_RATIO'] = donch_range / kelt_range.replace(0, np.nan)

                # Add fields to list
                self.band_fields.extend([
                    f'{prefix}_DONCH_UPPER', f'{prefix}_DONCH_LOWER', f'{prefix}_DONCH_MID',
                    f'{prefix}_KELT_UPPER', f'{prefix}_KELT_LOWER', f'{prefix}_KELT_MID',
                    f'{prefix}_DONCH_POS', f'{prefix}_KELT_POS', f'{prefix}_CHAN_WIDTH_RATIO'
                ])

            return df
        except Exception as e:
            print(f"Error HorizonAlignedIndicatorsProcessor: during calculate_bands: {str(e)}")
            sys.exit(2)

    def calculate_momentum(self, df):
        """
        Calculate momentum indicators aligned with forecast horizon
        Modified to reduce redundancy with RSI and MACD processors

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with momentum indicators added
        """
        # Reset momentum fields list
        self.momentum_fields = []
        try:
            # Calculate momentum indicators for each period
            for period in self.periods:
                prefix = f"HORIZON_ALIGNED_MOM_{period}"

                # Use minimum period of 3 for calculations
                actual_period = max(3, period)

                # ===== Unique momentum indicators not in other processors =====

                # 1. Ultimate Oscillator (uses multiple timeframes combined)
                if actual_period >= 7:  # Only for periods of reasonable length
                    # Calculate buying pressure and true range
                    bp = df['close'] - df[['low', 'close']].shift(1).min(axis=1)
                    tr = pd.concat([
                        df['high'] - df['low'],
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    ], axis=1).max(axis=1)

                    # Calculate average values for different periods
                    short_period = max(3, actual_period // 4)
                    mid_period = max(5, actual_period // 2)
                    long_period = actual_period

                    avg7 = bp.rolling(window=short_period).sum() / tr.rolling(window=short_period).sum()
                    avg14 = bp.rolling(window=mid_period).sum() / tr.rolling(window=mid_period).sum()
                    avg28 = bp.rolling(window=long_period).sum() / tr.rolling(window=long_period).sum()

                    # Calculate Ultimate Oscillator
                    df[f'{prefix}_ULT_OSC'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

                # 2. Chande Momentum Oscillator (sum of ups/downs rather than average like RSI)
                # This differs from RSI by focusing on magnitude of price changes
                price_change = df['close'].diff(actual_period)
                pos_sum = price_change.rolling(window=actual_period).apply(lambda x: sum([i for i in x if i > 0]) or 0)
                neg_sum = price_change.rolling(window=actual_period).apply(lambda x: sum([-i for i in x if i < 0]) or 0)

                df[f'{prefix}_CMO'] = 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum).replace(0, 1)

                # 3. Center of Gravity Oscillator (by John Ehlers)
                # Measures cycles and turning points
                def calc_cog(window):
                    if len(window) < actual_period:
                        return 0
                    numerator = sum((i + 1) * window.iloc[i] for i in range(len(window)))
                    denominator = sum(window)
                    if denominator == 0:
                        return 0
                    return numerator / denominator

                df[f'{prefix}_COG'] = df['close'].rolling(window=actual_period).apply(
                    calc_cog, raw=False
                )

                # 4. Money Flow Index - combines price and volume
                if 'quote_volume' in df.columns:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    money_flow = typical_price * df['quote_volume']

                    pos_flow = money_flow * (typical_price > typical_price.shift(1))
                    neg_flow = money_flow * (typical_price < typical_price.shift(1))

                    pos_sum = pos_flow.rolling(window=actual_period).sum()
                    neg_sum = neg_flow.rolling(window=actual_period).sum()

                    money_ratio = pos_sum / neg_sum.replace(0, 1e-10)
                    df[f'{prefix}_MFI'] = 100 - (100 / (1 + money_ratio))

                # 5. Price-based Momentum Strength
                # This uses close vs open over multiple periods to gauge directional strength
                momentum_sum = sum(np.sign(df['close'].shift(i) - df['open'].shift(i))
                                   for i in range(actual_period))
                df[f'{prefix}_MOM_STRENGTH'] = 0.0  # Initialize first
                df[f'{prefix}_MOM_STRENGTH'] = momentum_sum / actual_period

                # Add fields to list
                self.momentum_fields.append(f'{prefix}_MOM_STRENGTH')
                self.momentum_fields.append(f'{prefix}_CMO')
                self.momentum_fields.append(f'{prefix}_COG')

                if actual_period >= 7:
                    self.momentum_fields.append(f'{prefix}_ULT_OSC')

                if 'quote_volume' in df.columns:
                    self.momentum_fields.append(f'{prefix}_MFI')

            return df
        except Exception as e:
            print(f"Error HorizonAlignedIndicatorsProcessor: during calculate_momentum: {str(e)}")
            sys.exit(2)

    def calculate_virtual_candles(self, df):
        """
        Calculate features derived from virtualized horizon-aligned candlesticks
        Modified for single-step forecasting with reduced redundancy

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with virtual candle-derived features added
        """
        # Reset virtual candle fields list
        self.virtual_candle_fields = []

        try:
            # Calculate virtual candle features for each period
            for period in self.periods:
                prefix = f"HORIZON_ALIGNED_VC_{period}"

                # Use minimum period of 3 for calculations
                actual_period = max(3, period)

                # Virtual candle OHLC values - aggregated over the period
                df[f'{prefix}_OPEN'] = df['open'].shift(actual_period - 1)  # Open from period-steps ago
                df[f'{prefix}_HIGH'] = df['high'].rolling(window=actual_period).max()  # Highest high in the period
                df[f'{prefix}_LOW'] = df['low'].rolling(window=actual_period).min()  # Lowest low in the period
                df[f'{prefix}_CLOSE'] = df['close']  # Current close

                # Unique aspects of virtual candles - direction and magnitude
                df[f'{prefix}_RANGE'] = df[f'{prefix}_HIGH'] - df[f'{prefix}_LOW']
                df[f'{prefix}_BODY'] = abs(df[f'{prefix}_CLOSE'] - df[f'{prefix}_OPEN'])
                df[f'{prefix}_DIRECTION'] = np.sign(df[f'{prefix}_CLOSE'] - df[f'{prefix}_OPEN'])

                # Candle efficiency - how much of the range was converted into directional movement
                df[f'{prefix}_EFFICIENCY'] = df[f'{prefix}_BODY'] / df[f'{prefix}_RANGE'].replace(0, np.nan)

                # Unique pattern detection (not in other processors)

                # 1. Three-period pattern detection
                if actual_period >= 3:
                    # Three Inside Up/Down (Harami)
                    three_inside_up = ((df[f'{prefix}_DIRECTION'].shift(2) < 0) &
                                       (abs(df[f'{prefix}_BODY'].shift(1)) < abs(df[f'{prefix}_BODY'].shift(2))) &
                                       (df[f'{prefix}_DIRECTION'] > 0))

                    three_inside_down = ((df[f'{prefix}_DIRECTION'].shift(2) > 0) &
                                         (abs(df[f'{prefix}_BODY'].shift(1)) < abs(df[f'{prefix}_BODY'].shift(2))) &
                                         (df[f'{prefix}_DIRECTION'] < 0))

                    df[f'{prefix}_THREE_INSIDE'] = 0
                    df.loc[three_inside_up, f'{prefix}_THREE_INSIDE'] = 1
                    df.loc[three_inside_down, f'{prefix}_THREE_INSIDE'] = -1

                # 2. Outside Pattern (Engulfing)
                outside_up = ((df[f'{prefix}_DIRECTION'].shift(1) < 0) &
                              (df[f'{prefix}_OPEN'] < df[f'{prefix}_CLOSE'].shift(1)) &
                              (df[f'{prefix}_CLOSE'] > df[f'{prefix}_OPEN'].shift(1)))

                outside_down = ((df[f'{prefix}_DIRECTION'].shift(1) > 0) &
                                (df[f'{prefix}_OPEN'] > df[f'{prefix}_CLOSE'].shift(1)) &
                                (df[f'{prefix}_CLOSE'] < df[f'{prefix}_OPEN'].shift(1)))

                df[f'{prefix}_OUTSIDE'] = 0
                df.loc[outside_up, f'{prefix}_OUTSIDE'] = 1
                df.loc[outside_down, f'{prefix}_OUTSIDE'] = -1

                # 3. Gap detection
                gap_up = df[f'{prefix}_LOW'] > df[f'{prefix}_HIGH'].shift(1)
                gap_down = df[f'{prefix}_HIGH'] < df[f'{prefix}_LOW'].shift(1)

                df[f'{prefix}_GAP'] = 0
                df.loc[gap_up, f'{prefix}_GAP'] = 1
                df.loc[gap_down, f'{prefix}_GAP'] = -1

                # 4. Thrust bars (strong momentum)
                thrust_up = ((df[f'{prefix}_DIRECTION'] > 0) &
                             (df[f'{prefix}_BODY'] / df[f'{prefix}_RANGE'].replace(0, np.nan) > 0.7) &
                             (df[f'{prefix}_BODY'] > df[f'{prefix}_BODY'].shift(1) * 1.5))

                thrust_down = ((df[f'{prefix}_DIRECTION'] < 0) &
                               (df[f'{prefix}_BODY'] / df[f'{prefix}_RANGE'].replace(0, np.nan) > 0.7) &
                               (df[f'{prefix}_BODY'] > df[f'{prefix}_BODY'].shift(1) * 1.5))

                df[f'{prefix}_THRUST'] = 0
                df.loc[thrust_up, f'{prefix}_THRUST'] = 1
                df.loc[thrust_down, f'{prefix}_THRUST'] = -1

                # Add fields to list
                self.virtual_candle_fields.extend([
                    f'{prefix}_OPEN', f'{prefix}_HIGH', f'{prefix}_LOW', f'{prefix}_CLOSE',
                    f'{prefix}_RANGE', f'{prefix}_BODY', f'{prefix}_DIRECTION',
                    f'{prefix}_EFFICIENCY', f'{prefix}_OUTSIDE', f'{prefix}_GAP', f'{prefix}_THRUST'
                ])

                if actual_period >= 3:
                    self.virtual_candle_fields.append(f'{prefix}_THREE_INSIDE')

            return df
        except Exception as e:
            print(f"Error HorizonAlignedIndicatorsProcessor: during calculate_virtual_candles: {str(e)}")
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

        # Add all continuous indicator fields (exclude categorical/signal fields)
        for field in self.moving_average_fields:
            if not any(x in field for x in ['_DIRECTION', '_CROSS']):  # Skip directional signals
                features_to_normalize.append(field)

        for field in self.band_fields:
            features_to_normalize.append(field)

        for field in self.momentum_fields:
            if not any(x in field for x in ['_STATE', '_SIGNAL']):  # Skip state signals
                features_to_normalize.append(field)

        # Add virtual candle fields (excluding pattern indicators which are categorical)
        for field in self.virtual_candle_fields:
            if not any(x in field for x in ['_DIRECTION', '_THREE_INSIDE', '_OUTSIDE', '_GAP', '_THRUST']):
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