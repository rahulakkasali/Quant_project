import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, tickers):
        """
        Initialize DataLoader with a list of tickers.
        Supports stocks ('AAPL'), Crypto ('BTC-USD'), Forex ('EURUSD=X').
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers

    def fetch_historical_data(self, start_date, end_date, interval='1d'):
        """
        Fetch historical data for the initialized tickers.
        Intervals can be: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        """
        logger.info(f"Fetching historical data for {self.tickers} from {start_date} to {end_date} (interval: {interval})")
        data = yf.download(self.tickers, start=start_date, end=end_date, interval=interval, progress=False)
        
        # Determine if multi-level columns exist (happens when >1 ticker)
        if len(self.tickers) > 1:
            # We can optionally stack or format it. Default yfinance behavior 
            # returns a MultiIndex column DataFrame (e.g. data['Close']['AAPL'])
            pass
            
        return data

    def fetch_real_time_mock(self, interval='1m'):
        """
        Mock real-time data fetching. 
        In production, this would connect to a WebSocket (e.g., Binance, Alpaca).
        """
        logger.info(f"Fetching real-time mock data for {self.tickers}")
        # Fetch the very last available period
        data = yf.download(self.tickers, period='1d', interval=interval, progress=False)
        if not data.empty:
            # Return the last row as the "real-time" tick
            return data.iloc[[-1]]
        return pd.DataFrame()

    def fetch_latest_window(self, lookback_window: int = 10, interval: str = '1m') -> pd.DataFrame:
        """
        Fetches the exact recent N periods of real-time data explicitly for RL state formulation.
        Provides a complete streaming lookback window identical to what is used during backtest training.
        """
        logger.info(f"Streaming latest {lookback_window} periods of {interval} data for {self.tickers}...")
        
        # '5d' period ensures enough data points are grabbed even across low volume or weekend hours
        try:
            data = yf.download(self.tickers, period='5d', interval=interval, progress=False)
            
            # YFinance heavily utilizes multi-index blocks when >1 ticker is provided
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    data = data['Close']
            else:
                if 'Close' in data.columns:
                    data = data[['Close']]
                    data.columns = self.tickers
                    
            # Return specifically the final tail end rows
            latest_rows = data.dropna(axis=0, how='any')
            if len(latest_rows) >= lookback_window:
                return latest_rows.iloc[-lookback_window:]
            else:
                logger.warning(f"Live yfinance window too sparse. Found {len(latest_rows)} rows cleanly.")
                return latest_rows
        except Exception as e:
            logger.error(f"Live API Fetch Failure: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    loader = DataLoader(['AAPL', 'BTC-USD'])
    df = loader.fetch_historical_data(start_date='2025-01-01', end_date='2025-12-31')
    print("Historical Data Sample:")
    print(df.head())
    
    real_time_df = loader.fetch_real_time_mock()
    print("\nReal-time Mock Data:")
    print(real_time_df)
