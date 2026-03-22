import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def add_technical_indicators(self, data: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
        """
        Add Returns, RSI, MACD, EMAs, and Bollinger Bands.
        If the DataFrame has a MultiIndex (from multiple tickers), specify the ticker.
        """
        df = data.copy()
        
        # Handle multi-index data (yfinance multiple tickers format)
        if isinstance(df.columns, pd.MultiIndex):
            if not ticker:
                raise ValueError("DataFrame has MultiIndex columns. Please provide a ticker.")
            df = df.xs(ticker, axis=1, level=1).copy()
        else:
            # If standard Index, ensure 'Close' exists
            if 'Close' not in df.columns:
                raise KeyError("'Close' column is missing from data.")

        logger.info(f"Computing technical indicators for {ticker or 'asset'}...")

        # Pandas-TA Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, sign=9, append=True)
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        # Returns and Custom Volatility
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Return'].rolling(window=20).std()

        # Drop NaNs caused by rolling windows
        df.dropna(inplace=True)
        return df

    def scale_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Scale selected features using StandardScaler.
        """
        logger.info("Scaling features...")
        scaled_df = df.copy()
        scaled_df[feature_cols] = self.scaler.fit_transform(scaled_df[feature_cols])
        return scaled_df

    def create_sequences(self, df: pd.DataFrame, feature_cols: list, target_col: str, seq_length: int = 60):
        """
        Create 3D sequence arrays for LSTM/GRU models.
        Returns: X (sequences), y (targets)
        """
        logger.info(f"Creating sequences of length {seq_length}...")
        X, y = [], []
        data_features = df[feature_cols].values
        data_target = df[target_col].values

        for i in range(len(df) - seq_length):
            X.append(data_features[i:(i + seq_length)])
            y.append(data_target[i + seq_length])

        return np.array(X), np.array(y)

if __name__ == "__main__":
    # Test with random data
    dates = pd.date_range("2025-01-01", periods=100)
    mock_data = pd.DataFrame({
        "Open": np.random.rand(100) * 100,
        "High": np.random.rand(100) * 100,
        "Low": np.random.rand(100) * 100,
        "Close": np.random.rand(100) * 100,
        "Volume": np.random.rand(100) * 1000
    }, index=dates)

    engineer = FeatureEngineer()
    processed_df = engineer.add_technical_indicators(mock_data)
    print("Processed Features Head:")
    print(processed_df.head())
    
    # Scale test
    feature_cols = ['Close', 'RSI_14', 'MACD_12_26_9']
    if all(c in processed_df.columns for c in feature_cols):
        scaled_df = engineer.scale_features(processed_df, feature_cols)
        print("\nScaled Features Head:")
        print(scaled_df[feature_cols].head())
        
        # Sequence test
        X, y = engineer.create_sequences(scaled_df, feature_cols, target_col='Return', seq_length=10)
        print(f"\nSequences created: X shape: {X.shape}, y shape: {y.shape}")
