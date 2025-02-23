import pandas as pd
from datetime import datetime, timezone
import os
from pathlib import Path


class MACDProcessor:
    def __init__(self, data_dir: Path, ma_fast: int = 12, ma_slow: int = 26, signal_length: int = 9):
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
        if self.ma_fast == 12:
            self.macd_variant = "long"
        else:
            self.macd_variant = "short"

    def process_csv(self, symbol: str, interval: str) -> None:
        """
        Process a single CSV file to add MACD indicators

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '1h', '4h')
        """
        filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Historical data file not found: {filename}")

        # Read the CSV file
        df = pd.read_csv(filename, index_col=0)

        # Calculate MACD indicators
        signals = self._generate_signals(df['close'])
        histogram_field_name = f"macd_histogram_{self.macd_variant}"
        df[histogram_field_name] = signals['MACD-Signal']
        # Generate trend direction
        trend_direction_field_name = f"trend_direction_{self.macd_variant}"
        df[trend_direction_field_name] = -99
        df.loc[signals['macd_line'] > signals['signal_line'], trend_direction_field_name] = 1
        df.loc[signals['macd_line'] < signals['signal_line'], trend_direction_field_name] = -1
        df.loc[signals['macd_line'] == signals['signal_line'], trend_direction_field_name] = 0

        # Save back to CSV
        df.to_csv(filename)

        print(f"Processed {filename}")
        print(f"Sample of processed data:")
        print(df[['close', histogram_field_name, trend_direction_field_name]].tail())

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
