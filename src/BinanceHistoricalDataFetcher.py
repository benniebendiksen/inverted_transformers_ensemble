import os
import sys

from src.Config import Config
from src.indicators.MACDProcessor import MACDProcessor
from src.indicators.BollingerBandsProcessor import BollingerBandsProcessor
from src.indicators.RSIProcessor import RSIProcessor
from src.indicators.MarketFeaturesProcessor import MarketFeaturesProcessor
from src.indicators.HorizonAlignedIndicatorsProcessor import HorizonAlignedIndicatorsProcessor
from dataset_descriptives import print_df_stats
from unicorn_binance_rest_api import BinanceRestApiManager
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class BinanceHistoricalDataFetcher:
    """
    Fetches and manages historical price data from Binance Futures
    with rate limiting and data persistence capabilities.
    """

    def __init__(
            self,
            symbol: str,
            interval: str,
            exchange: str,
            max_requests_per_minute: int = 1200,
            request_weight: int = 1
    ):
        print(f"Creating Fetcher Instance for Exchange: {exchange}")
        self.config = Config()
        self.have_loaded_index = False
        self.socks5_proxy = self.config.SOCKS5_IP_PORT
        self.logger = None
        self.stats = None
        self.symbol = symbol.upper()
        self.interval = interval
        self.exchange = exchange
        self.counter = 0
        if self.exchange == "binance_us":
            self.client = BinanceRestApiManager(
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
                exchange="binance.us"
            )
        else:
            self.client = BinanceRestApiManager(
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
                exchange="binance.com-futures",
                # socks5_proxy_server=self.socks5_proxy,
                # socks5_proxy_user=None,
                # socks5_proxy_pass=None,
                # socks5_proxy_ssl_verification=True
            )
        str_data_dir = exchange + "_" + "historical_data"
        self.data_dir = Path(str_data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.max_requests_per_minute = max_requests_per_minute
        self.request_weight = request_weight
        self.request_timestamps: List[float] = []

        self.setup_logging()

        self.interval_ms = {
            '1m': 60000,
            '3m': 180000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '2h': 7200000,
            '4h': 14400000,
            '6h': 21600000,
            '8h': 28800000,
            '12h': 43200000,
            '1d': 86400000,
        }

    def setup_logging(self):
        """Configure logging for the data fetcher"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.symbol.lower()}_data_fetcher.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Track data collection statistics
        self.stats = {
            'total_requests': 0,
            'empty_responses': 0,
            'total_records': 0,
            'earliest_timestamp': None,
            'latest_timestamp': None
        }

    def check_rate_limit(self) -> None:
        """
        Implements rate limiting by tracking request timestamps and waiting if necessary.
        """
        weight_info = self.client.get_used_weight()
        print(f"Current API weight usage: {weight_info}")

        current_time = time.time()
        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if current_time - ts < 60]

        if len(self.request_timestamps) * self.request_weight >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit approaching, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

        self.request_timestamps.append(current_time)

    def fetch_klines(
            self,
            start_time: int,
            end_time: Optional[int] = None,
            limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch kline data for a specific time range with rate limiting.
        start_time: int: timestamp
        end_time: int: timestamp
        """
        self.check_rate_limit()
        self.stats['total_requests'] += 1

        try:
            if self.exchange == "binance_us":
                klines = self.client.get_historical_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    start_str=start_time,
                    end_str=end_time,
                    limit=limit
                )
            else:
                klines = self.client.futures_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    startTime=start_time,
                    endTime=end_time,
                    limit=limit
                )

            time.sleep(0.5)

            if not klines:
                self.stats['empty_responses'] += 1
                self.logger.info(
                    f"No data available for period starting at {datetime.fromtimestamp(start_time / 1000)}")
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_volume',
                'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Update statistics
            self.stats['total_records'] += len(df)
            if self.stats['earliest_timestamp'] is None or df.index.min() < self.stats['earliest_timestamp']:
                self.stats['earliest_timestamp'] = df.index.min()
            if self.stats['latest_timestamp'] is None or df.index.max() > self.stats['latest_timestamp']:
                self.stats['latest_timestamp'] = df.index.max()

            return df

        except Exception as e:
            self.logger.error(f"Error fetching klines: {str(e)}")
            raise

    def fetch_from_end_time_working_backwards(self, end_datetime=None) -> pd.DataFrame:
        """
        Fetch all available historical data for the symbol and interval.
        """
        print(f"Starting historical data collection for {self.symbol}")

        chunk_size = 1000
        all_data = []
        consecutive_empty_responses = 0
        max_empty_responses = 3  # Stop after 3 consecutive empty responses

        if end_datetime:
            # Must be valid the datetime string, e.g., "2019-09-08 17:45:00"
            # Convert to a datetime object
            dt_obj = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)  # Explicitly set UTC
            # Convert to Unix timestamp (milliseconds)
            timestamp_ms = int(dt_obj.timestamp() * 1000)
            end_time = timestamp_ms
        else:
            end_time = int(time.time() * 1000)
        start_time = end_time - (chunk_size * self.interval_ms[self.interval])

        while True:
            print(f"Fetching data from {datetime.fromtimestamp(start_time / 1000)}")

            chunk = self.fetch_klines(
                start_time=start_time,
                end_time=end_time,
                limit=chunk_size
            )

            if chunk.empty:
                consecutive_empty_responses += 1
                if consecutive_empty_responses >= max_empty_responses:
                    self.logger.info(
                        "Reached maximum consecutive empty responses, assuming no more historical data available")
                    break
            else:
                consecutive_empty_responses = 0
                all_data.append(chunk)

            end_time = start_time
            start_time = end_time - (chunk_size * self.interval_ms[self.interval])

            if len(all_data) % 10 == 0:
                self.save_data_backwards(pd.concat(all_data))

        if not all_data:
            self.logger.error("No data collected!")
            return pd.DataFrame()

        final_df = pd.concat(all_data).sort_index()
        self.save_data_backwards(final_df)

        # Log collection statistics
        self.logger.info(f"Data collection completed:")
        self.logger.info(f"Total records: {self.stats['total_records']}")
        self.logger.info(f"Date range: {self.stats['earliest_timestamp']} to {self.stats['latest_timestamp']}")
        self.logger.info(f"Total requests: {self.stats['total_requests']}")
        self.logger.info(f"Empty responses: {self.stats['empty_responses']}")

        return final_df

    def fetch_from_start_time_working_forwards(self, start_datetime) -> pd.DataFrame:
        """
        Fetch all available historical data from a given start time working forwards.
        All timestamps are handled in UTC to ensure consistency with Binance's API.

        Args:
            start_datetime: string in format "YYYY-MM-DD HH:MM:SS"
                           Should be in UTC time
        """
        print(f"Starting forward historical data collection for {self.symbol}")

        chunk_size = 1000
        all_data = []
        consecutive_empty_responses = 0
        max_empty_responses = 3

        if start_datetime:
            # Convert start_datetime string to UTC timestamp
            dt_obj = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)  # Explicitly set UTC
            start_time = int(dt_obj.timestamp() * 1000)
        else:
            raise ValueError("start_datetime must be provided")

        # Get current time in UTC
        current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        original_start_time = start_time

        # Ensure start_time isn't in the future
        if start_time > current_time:
            self.logger.error("Start time cannot be in the future")
            return pd.DataFrame()

        while start_time < current_time:
            end_time = start_time + (chunk_size * self.interval_ms[self.interval])

            # Adjust end_time if it would exceed current time
            if end_time > current_time:
                end_time = current_time

            print(f"Fetching data from {datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)} to "
                  f"{datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)} UTC")

            chunk = self.fetch_klines(
                start_time=start_time,
                end_time=end_time,
                limit=chunk_size
            )

            if chunk.empty:
                consecutive_empty_responses += 1
                if consecutive_empty_responses >= max_empty_responses:
                    self.logger.info("Reached maximum consecutive empty responses, assuming no more data available")
                    break
            else:
                consecutive_empty_responses = 0
                all_data.append(chunk)

            # Save periodically
            if len(all_data) % 10 == 0:
                self.save_data_forwards(pd.concat(all_data))

            # If we've reached current_time, we're done
            if end_time >= current_time:
                break

            # Set up next iteration
            start_time = end_time

        if not all_data:
            self.logger.error("No data collected!")
            return pd.DataFrame()

        final_df = pd.concat(all_data).sort_index()
        self.save_data_forwards_with_indicators(final_df)

        # Log collection statistics
        self.logger.info(f"Data collection completed:")
        self.logger.info(f"Total records: {self.stats['total_records']}")
        self.logger.info(f"Date range: {self.stats['earliest_timestamp']} to {self.stats['latest_timestamp']}")
        self.logger.info(f"Total requests: {self.stats['total_requests']}")
        self.logger.info(f"Empty responses: {self.stats['empty_responses']}")

        return final_df

    # def save_data_forwards_with_indicators(self, df: pd.DataFrame) -> None:
    #     """
    #     Save the DataFrame to a CSV file while maintaining index consistency and indicator columns.
    #     Recalculates indicators only for new data points.
    #     """
    #     filename = self.data_dir / f"{self.symbol.lower()}_{self.interval}_historical.csv"
    #     df.sort_index(inplace=True)
    #     self.counter += 1
    #
    #     try:
    #         if os.path.exists(filename):
    #             # Load existing CSV with all columns - ensure index is parsed as datetime
    #             existing_df = pd.read_csv(filename, index_col=0, parse_dates=True)
    #
    #             # Identify columns that need to be numeric for calculations
    #             numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
    #                                'taker_buy_base', 'taker_buy_quote']
    #
    #             # Ensure critical columns are numeric in the existing dataframe
    #             for column in numeric_columns:
    #                 if column in existing_df.columns:
    #                     existing_df[column] = pd.to_numeric(existing_df[column], errors='coerce')
    #
    #             # Get newest stored timestamp
    #             newest_timestamp = existing_df.index[-1]
    #             print(f"existing oldest ts: {existing_df.index[0]}")
    #             print(f"existing newest ts: {newest_timestamp}")
    #             print(f"new chunk oldest ts: {df.index[0]}")
    #             print(f"new chunk newest ts: {df.index[-1]}")
    #
    #             # Only append data newer than our newest stored timestamp
    #             new_data = df[df.index > newest_timestamp].copy()  # Create an explicit copy
    #
    #             if not new_data.empty:
    #                 # Before combining, ensure new_data columns have matching types with existing_df
    #                 for column in numeric_columns:
    #                     if column in new_data.columns and column in existing_df.columns:
    #                         # Ensure the column has the same dtype in both dataframes
    #                         if new_data[column].dtype != existing_df[column].dtype:
    #                             # If existing is numeric and new is not, convert new to match existing
    #                             if pd.api.types.is_numeric_dtype(existing_df[column].dtype):
    #                                 new_data[column] = pd.to_numeric(new_data[column], errors='coerce')
    #
    #                 # Get lookback data for indicator calculations - using minimum required lookback
    #                 # Get the maximum lookback required by any indicator
    #                 lookback_size = 50  # BB secondary length
    #                 macd_lookback = max(Config.MA_SLOW, 17) + Config.SIGNAL_LENGTH  # For both MACD processors
    #                 rsi_lookback = Config.RSI_LOOKBACK
    #                 market_lookback = 8  # Maximum window size in market features
    #                 horizon_lookback = Config.FORECAST_STEPS + 1
    #
    #                 max_lookback = max(lookback_size, macd_lookback, rsi_lookback, market_lookback, horizon_lookback)
    #
    #                 # Get only necessary lookback data
    #                 lookback_data = existing_df.tail(max_lookback).copy()  # Create explicit copy
    #
    #                 print(
    #                     f"Processing with lookback size: {len(lookback_data)} records for {len(new_data)} new records")
    #
    #                 # Combine lookback data with new data
    #                 calculation_df = pd.concat([lookback_data, new_data])
    #
    #                 # Final check: ensure all critical columns are numeric in the combined dataframe
    #                 for column in numeric_columns:
    #                     if column in calculation_df.columns:
    #                         calculation_df[column] = pd.to_numeric(calculation_df[column], errors='coerce')
    #
    #                 # Debug - print datatypes
    #                 print("Data types after combination:")
    #                 for column in numeric_columns:
    #                     if column in calculation_df.columns:
    #                         print(f"Column {column}: {calculation_df[column].dtype}")
    #
    #                 # Process indicators with external processors
    #                 try:
    #                     macd_processor = MACDProcessor(
    #                         data_dir=Path("dummy_directory"),
    #                         ma_fast=Config.MA_FAST,
    #                         ma_slow=Config.MA_SLOW,
    #                         signal_length=Config.SIGNAL_LENGTH
    #                     )
    #                     macd_processor_short = MACDProcessor(
    #                         data_dir=Path("dummy_directory"),
    #                         ma_fast=8,
    #                         ma_slow=17,
    #                         signal_length=9
    #                     )
    #                     bband_processor = BollingerBandsProcessor(
    #                         data_dir=Path("dummy_directory"),
    #                         length=Config.BOLL_LENGTH,
    #                         multiplier=Config.BOLL_MULTIPLIER,
    #                         slope_period=Config.SLOPE_PERIOD
    #                     )
    #                     bband_processor_secondary = BollingerBandsProcessor(
    #                         data_dir=Path("dummy_directory"),
    #                         length=50,
    #                         multiplier=Config.BOLL_MULTIPLIER,
    #                         slope_period=Config.SLOPE_PERIOD
    #                     )
    #                     rsi_processor = RSIProcessor(
    #                         data_dir=Path("dummy_directory"),
    #                         length=Config.RSI_LOOKBACK,
    #                         oversold=Config.RSI_OVERSOLD,
    #                         overbought=Config.RSI_OVERBOUGHT
    #                     )
    #
    #                     market_processor = MarketFeaturesProcessor(
    #                         data_dir=Path("dummy_directory"),
    #                         lag_periods=[1],
    #                         volatility_windows=[4, 8],
    #                         volume_windows=[4, 8]
    #                     )
    #
    #                     horizon_processor = HorizonAlignedIndicatorsProcessor(
    #                         data_dir=Path("dummy_directory"),
    #                         forecast_steps=Config.FORECAST_STEPS,
    #                         multiples=[1]
    #                     )
    #
    #                     # Calculate indicators
    #                     temp_df = calculation_df.copy()
    #
    #                     # Track processing time
    #                     start_time = time.time()
    #
    #                     # Perform calculations that might need to be wrapped in try/except blocks
    #                     try:
    #                         temp_df = macd_processor.calculate_macd_values(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in MACD long calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = macd_processor_short.calculate_macd_values(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in MACD short calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = bband_processor.calculate_bollinger_values(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in Bollinger Bands calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = bband_processor_secondary.calculate_bollinger_values(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in secondary Bollinger Bands calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = rsi_processor.calculate_rsi(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in RSI calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = market_processor.calculate_base_features(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in market base features calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = market_processor.calculate_lagged_features(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in lagged features calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = market_processor.calculate_volatility_features(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in volatility features calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = market_processor.calculate_candlestick_features(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in candlestick features calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = market_processor.calculate_volume_features(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in volume features calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = horizon_processor.calculate_virtual_candles(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in virtual candles calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = horizon_processor.calculate_moving_averages_and_regression(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in moving averages calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = horizon_processor.calculate_bands(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in bands calculation: {str(e)}")
    #
    #                     try:
    #                         temp_df = horizon_processor.calculate_momentum(temp_df)
    #                     except Exception as e:
    #                         print(f"Error in momentum calculation: {str(e)}")
    #
    #                     end_time = time.time()
    #                     print(f"Total indicator calculation time: {end_time - start_time:.2f} seconds")
    #
    #                 except Exception as e:
    #                     print(f"Error setting up processors: {str(e)}")
    #                     sys.exit(2)
    #
    #                 # Extract only the new records with their indicators
    #                 new_data_with_indicators = temp_df[temp_df.index > newest_timestamp]
    #
    #                 # Combine with existing data
    #                 combined_df = pd.concat([existing_df, new_data_with_indicators])
    #             else:
    #                 raise Exception("No new data to append")
    #         else:
    #             # Raise exception as this functionality is not supported for new files
    #             raise Exception(
    #                 "Cannot append indicators to a new file; use fetch_from_end_time() instead to grab all historic data")
    #
    #         # Save back to CSV
    #         combined_df.to_csv(filename, mode="w", header=True, index=True)
    #         print(f"Appended {len(new_data) if 'new_data' in locals() else len(df)} new records to {filename}")
    #
    #     except Exception as e:
    #         print(f"Error during data saving: {str(e)}")
    #         sys.exit(2)

    def save_data_forwards_with_indicators(self, df: pd.DataFrame) -> None:
        """
        Save the DataFrame to a CSV file while maintaining index consistency and indicator columns.
        Recalculates indicators only for new data points.
        """
        filename = self.data_dir / f"{self.symbol.lower()}_{self.interval}_historical.csv"
        df.sort_index(inplace=True)
        self.counter += 1

        try:
            if os.path.exists(filename):
                # Load existing CSV with all columns - ensure index is parsed as datetime
                existing_df = pd.read_csv(filename, index_col=0, parse_dates=True)

                # Identify columns that need to be numeric for calculations
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                                   'taker_buy_base', 'taker_buy_quote']

                # Ensure critical columns are numeric in the existing dataframe
                for column in numeric_columns:
                    if column in existing_df.columns:
                        existing_df[column] = pd.to_numeric(existing_df[column], errors='coerce')

                # Get newest stored timestamp
                newest_timestamp = existing_df.index[-1]
                print(f"existing oldest ts: {existing_df.index[0]}")
                print(f"existing newest ts: {newest_timestamp}")
                print(f"new chunk oldest ts: {df.index[0]}")
                print(f"new chunk newest ts: {df.index[-1]}")

                # Only append data newer than our newest stored timestamp
                new_data = df[df.index > newest_timestamp].copy()  # Create an explicit copy to avoid SettingWithCopyWarning

                if not new_data.empty:
                    # Get lookback data for indicator calculations
                    lookback_size = 50  # Maximum lookback needed for indicators; currently determined by BB secondary length
                    lookback_data = existing_df.tail(lookback_size)

                    # Before combining, ensure new_data columns have matching types with existing_df
                    for column in numeric_columns:
                        if column in new_data.columns and column in existing_df.columns:
                            # Ensure the column has the same dtype in both dataframes
                            if new_data[column].dtype != existing_df[column].dtype:
                                # If existing is numeric and new is not, convert new to match existing
                                if pd.api.types.is_numeric_dtype(existing_df[column].dtype):
                                    new_data[column] = pd.to_numeric(new_data[column], errors='coerce')

                    # Combine lookback data with new data
                    calculation_df = pd.concat([lookback_data, new_data])

                    # Final check: ensure all critical columns are numeric in the combined dataframe
                    for column in numeric_columns:
                        if column in calculation_df.columns:
                            calculation_df[column] = pd.to_numeric(calculation_df[column], errors='coerce')

                    # Debug - print datatypes
                    print("Data types after combination:")
                    for column in numeric_columns:
                        if column in calculation_df.columns:
                            print(f"Column {column}: {calculation_df[column].dtype}")

                    # Process indicators with external processors
                    macd_processor = MACDProcessor(
                        data_dir=Path("dummy_directory"),# Don't need data_dir for direct processing
                        ma_fast=Config.MA_FAST,
                        ma_slow=Config.MA_SLOW,
                        signal_length=Config.SIGNAL_LENGTH
                    )
                    macd_processor_short = MACDProcessor(data_dir=Path("dummy_directory"), ma_fast=8, ma_slow=17, signal_length=9)
                    bband_processor = BollingerBandsProcessor(
                        data_dir=Path("dummy_directory"),
                        length=Config.BOLL_LENGTH,
                        multiplier=Config.BOLL_MULTIPLIER,
                        slope_period=Config.SLOPE_PERIOD
                    )
                    bband_processor_secondary = BollingerBandsProcessor(
                        data_dir=Path("dummy_directory"),
                        length=50,
                        multiplier=Config.BOLL_MULTIPLIER,
                        slope_period=Config.SLOPE_PERIOD
                    )
                    rsi_processor = RSIProcessor(
                        data_dir=Path("dummy_directory"),
                        length=Config.RSI_LOOKBACK,
                        oversold=Config.RSI_OVERSOLD,
                        overbought=Config.RSI_OVERBOUGHT
                    )

                    market_processor = MarketFeaturesProcessor(
                        data_dir=Path("dummy_directory"),
                        lag_periods=[1],
                        volatility_windows=[4, 8],
                        volume_windows=[4, 8]
                    )

                    horizon_processor = HorizonAlignedIndicatorsProcessor(
                        data_dir=Path("dummy_directory"),
                        forecast_steps=Config.FORECAST_STEPS,
                        multiples=[1]
                    )

                    # Calculate indicators
                    temp_df = calculation_df.copy()
                    temp_df = macd_processor.calculate_macd_values(temp_df)
                    temp_df = macd_processor.calculate_enhanced_features(temp_df)
                    temp_df = macd_processor_short.calculate_macd_values(temp_df)
                    temp_df = macd_processor_short.calculate_enhanced_features(temp_df)

                    temp_df = bband_processor.calculate_bollinger_values(temp_df)
                    temp_df = bband_processor.calculate_enhanced_features(temp_df)
                    temp_df = bband_processor_secondary.calculate_bollinger_values(temp_df)
                    temp_df = bband_processor_secondary.calculate_enhanced_features(temp_df)

                    temp_df = rsi_processor.calculate_rsi(temp_df)
                    temp_df = rsi_processor.calculate_enhanced_features(temp_df)

                    temp_df = market_processor.calculate_base_features(temp_df)
                    temp_df = market_processor.calculate_lagged_features(temp_df)
                    temp_df = market_processor.calculate_volatility_features(temp_df)
                    temp_df = market_processor.calculate_candlestick_features(temp_df)
                    temp_df = market_processor.calculate_volume_features(temp_df)

                    temp_df = horizon_processor.calculate_virtual_candles(temp_df)
                    temp_df = horizon_processor.calculate_moving_averages_and_regression(temp_df)
                    temp_df = horizon_processor.calculate_bands(temp_df)
                    temp_df = horizon_processor.calculate_momentum(temp_df)

                    # Extract only the new records with their indicators
                    new_data_with_indicators = temp_df[temp_df.index > newest_timestamp]

                    # Combine with existing data
                    combined_df = pd.concat([existing_df, new_data_with_indicators])
                else:
                    raise Exception("No new data to append")
            else:
                # Raise exception as this functionality is not supported for new files
                raise Exception("Cannot append indicators to a new file; use fetch_from_end_time() instead to grab all historic data")

            # Save back to CSV
            combined_df.to_csv(filename, mode="w", header=True, index=True)
            print(f"Appended {len(new_data) if 'new_data' in locals() else len(df)} new records to {filename}")

        except Exception as e:
            print(f"Error during data saving: {str(e)}")
            sys.exit(2)

    def save_data_backwards(self, df: pd.DataFrame) -> None:
        """Save the DataFrame to a CSV file while maintaining index consistency"""
        filename = self.data_dir / f"{self.symbol.lower()}_{self.interval}_historical.csv"
        df.sort_index(inplace=True)
        self.counter += 1

        try:
            if os.path.exists(filename):
                # Load existing CSV and ensure index is datetime
                existing_df = pd.read_csv(filename, index_col=0)

                # Get oldest stored timestamp
                last_timestamp = existing_df.index[0]
                print(f"existing oldest ts: {last_timestamp}")
                print(f"existing newest ts: {existing_df.index[-1]}")

                print(f"oldest ts: {df.index[0]}")
                print(f"newest ts: {df.index[-1]}")

                df = df[df.index < last_timestamp]
                combined_df = pd.concat([df, existing_df])
            else:
                combined_df = df

            # Save back to CSV (overwrite with new order)
            combined_df.to_csv(filename, mode="w", header=True, index=True)
            print(f"Prepended {len(df)} new records to {filename}")
        except Exception as e:
            print(f"Error during data saving: {str(e)}")
            sys.exit(2)

    def save_data_forwards(self, df: pd.DataFrame) -> None:
        """Save the DataFrame to a CSV file while maintaining index consistency for forward fetching"""
        filename = self.data_dir / f"{self.symbol.lower()}_{self.interval}_historical.csv"
        df.sort_index(inplace=True)
        self.counter += 1

        try:
            if os.path.exists(filename):
                # Load existing CSV and ensure index is datetime
                existing_df = pd.read_csv(filename, index_col=0)

                # Get newest stored timestamp
                newest_timestamp = existing_df.index[-1]
                print(f"existing oldest ts: {existing_df.index[0]}")
                print(f"existing newest ts: {newest_timestamp}")
                print(f"new chunk oldest ts: {df.index[0]}")
                print(f"new chunk newest ts: {df.index[-1]}")

                # Only append data newer than our newest stored timestamp
                df = df[df.index > newest_timestamp]
                combined_df = pd.concat([existing_df, df])
            else:
                combined_df = df

            # Save back to CSV (overwrite with new order)
            combined_df.to_csv(filename, mode="w", header=True, index=True)
            print(f"Appended {len(df)} new records to {filename}")
        except Exception as e:
            print(f"Error during data saving: {str(e)}")
            sys.exit(2)

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load historical data from CSV file if it exists"""
        filename = self.data_dir / f"{self.symbol.lower()}_{self.interval}_historical.csv"
        if filename.exists():
            return pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        return None
