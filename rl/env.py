import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioEnv(gym.Env):
    """
    Custom OpenAI Gymnasium Environment for Portfolio Optimization.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, price_data: pd.DataFrame, signal_data: pd.DataFrame = None, initial_balance=10000, lookback_window=10):
        super(PortfolioEnv, self).__init__()
        self.df = price_data
        self.n_assets = len(price_data.columns)
        
        if signal_data is None:
            self.signal_data = pd.DataFrame(np.zeros_like(price_data.values), columns=price_data.columns, index=price_data.index)
        else:
            self.signal_data = signal_data
            
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window

        # Action: weights for each asset. Continuous [0, 1].
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Observation: flattened lookback window of prices + current AI signals
        obs_shape = (self.lookback_window * self.n_assets) + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.current_step = self.lookback_window
        self.balance = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.df.iloc[self.current_step - self.lookback_window : self.current_step].values
        signals = self.signal_data.iloc[self.current_step].values
        
        obs = np.concatenate((window.flatten(), signals))
        return obs.astype(np.float32)

    def step(self, action):
        # Normalize weights to sum to 1
        action_sum = np.sum(action)
        weights = action / action_sum if action_sum > 0 else np.ones(self.n_assets) / self.n_assets
        
        current_prices = self.df.iloc[self.current_step].values
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            return self._get_obs(), 0, True, False, {}

        next_prices = self.df.iloc[self.current_step].values
        price_returns = (next_prices - current_prices) / current_prices
        
        portfolio_return = np.sum(weights * price_returns)
        
        # Calculate recent volatility to penalize risk
        recent_returns = self.df.iloc[self.current_step - self.lookback_window : self.current_step].pct_change().dropna().values
        if len(recent_returns) > 1:
            portfolio_daily_returns = np.sum(recent_returns * weights, axis=1)
            portfolio_volatility = np.std(portfolio_daily_returns)
        else:
            portfolio_volatility = 0.0
            
        # Reward function: Incentivize return, explicitly penalize risk (volatility)
        risk_penalty_factor = 2.0  # Hyperparameter
        reward = portfolio_return - (risk_penalty_factor * portfolio_volatility)
        
        self.balance *= (1 + portfolio_return)
        
        done = self.current_step >= len(self.df) - 1
        info = {
            "balance": self.balance, 
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_volatility,
            "reward": reward
        }
        
        return self._get_obs(), reward, done, False, info

    def render(self):
        logger.info(f"Step: {self.current_step}, Balance: ${self.balance:.2f}")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from execution.signal_generator import UnifiedSignalGenerator
    
    # Test env
    generator = UnifiedSignalGenerator()
    mock_prices = pd.DataFrame(np.random.rand(100, 3) * 100 + 50, columns=['AAPL', 'MSFT', 'BTC'])
    
    # Generate mock AI signals for the timeframe using the generator format
    signals_list = []
    for _ in range(100):
        aapl_sig = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        msft_sig = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        btc_sig  = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        signals_list.append([aapl_sig, msft_sig, btc_sig])
        
    mock_signals = pd.DataFrame(signals_list, columns=['AAPL', 'MSFT', 'BTC'])
    
    env = PortfolioEnv(mock_prices, mock_signals)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    print(f"Sample step info: {info}")
