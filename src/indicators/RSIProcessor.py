from src.Config import Config
import pandas as pd
import numpy as np
from pathlib import Path


class RSIProcessor:
    def __init__(self, data_dir: Path, length: int = Config.RSI_LOOKBACK, oversold: float = Config.RSI_OVERSOLD, overbought: float = Config.RSI_OVERBOUGHT):
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

    def process_csv(self, symbol: str, interval: str) -> None:
        """
        Process a single CSV file to add RSI indicators

        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            interval: Timeframe interval (e.g., '1h', '4h')
        """
        filename = self.data_dir / f"{symbol.lower()}_{interval}_historical.csv"

        if not filename.exists():
            raise FileNotFoundError(f"Historical data file not found: {filename}")

        # Read the CSV file
        df = pd.read_csv(filename, index_col=0)

        # Calculate RSI and signals
        df_processed = self.calculate_rsi(df)

        # Save back to CSV
        df_processed.to_csv(filename)

        print(f"Processed {filename}")
        print(f"Sample of processed data:")
        print(df_processed[['close', 'RSI', 'RSI_signal']].tail())

    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and generate trading signals

        Args:
            df: DataFrame with historical price data

        Returns:
            DataFrame with added RSI columns
        """
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
        df['RSI'] = rsi_values

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

        df['RSI_signal'] = signals

        return df