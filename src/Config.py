from src.BaseClass import BaseClass
import os


class Config(BaseClass):
    """
    Configuration class
    """

    def __init__(self):
        self.stdout(f"Loading trend-activated-bot configuration ...")

    API_KEY = os.environ.get('API_KEY')
    API_SECRET = os.environ.get('API_SECRET')
    SOCKS5_IP_PORT = "13.231.151.193:1080"
    REMOTE_KEYS_PATH = "/Users/bendiksen/Desktop/trend-activated-trailing-stop-loss-bot/MY_AWS_KEYS.pem"
    AGGTRADE_FILENAME = "df_aggTrade_returned.json"
    MARKETS = ["btc_usdt", "eth_usdt"]
    CANDLESTICK_TIME_INTERVAL = '1m'
    # MACD 1 PARAMS
    MA_FAST = 12
    MA_SLOW = 26
    SIGNAL_LENGTH = 9
    # RSI PARAMS
    RSI_LOOKBACK = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    # Bollinger Bands 1 PARAMS
    BOLL_LENGTH = 20
    BOLL_MULTIPLIER = 2.0
    SLOPE_PERIOD = 4
    # Future Horizon-Aligned PARAMS
    FORECAST_STEPS = 4

