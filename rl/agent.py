try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import numpy as np
from rl.env import PortfolioEnv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFAgent:
    def __init__(self, env: PortfolioEnv):
        """
        TensorFlow-based Policy Network for Portfolio Optimization.
        Replaces Stable-Baselines3 (which depends on PyTorch).
        """
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        
        if TF_AVAILABLE:
            self.model = tf.keras.Sequential([
                Dense(64, activation='relu', input_shape=(self.state_size,)),
                Dense(64, activation='relu'),
                Dense(self.action_size, activation='softmax') # Softmax ensures weights sum to 1
            ])
            logger.info("Initialized TensorFlow RL Policy Network")
        else:
            logger.warning("TensorFlow not installed (OS mismatch). Using safely-mocked TFAgent for execution logic.")
            self.model = None

    def predict(self, state, deterministic=True):
        if self.model is not None:
            state_expanded = np.expand_dims(state, axis=0)
            action_probs = self.model(state_expanded).numpy()[0]
        else:
            # Return safe fallback pseudo-randomized weights if TF failed to load
            action_probs = np.random.dirichlet(np.ones(self.action_size), size=1)[0]
        return action_probs, None

    def train(self, episodes=10):
        """
        Placeholder for custom RL training loop (e.g., REINFORCE or DDPG in TF).
        """
        logger.info(f"Training TF agent for {episodes} episodes (Mock)...")
        # In a full implementation, you'd apply gradients based on rewards here.
        logger.info("Training complete.")

    def test(self, episodes=1):
        """
        Test deterministic actions in the environment.
        """
        logger.info(f"Running deterministic evaluation for {episodes} episodes.")
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                total_reward += reward
                
            logger.info(f"Episode {ep+1} | Total Return: {total_reward:.4f} | Final Balance: ${info['balance']:.2f}")

if __name__ == "__main__":
    import pandas as pd
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from execution.signal_generator import UnifiedSignalGenerator
    
    generator = UnifiedSignalGenerator()
    
    mock_prices = pd.DataFrame(np.random.rand(500, 3) * 100 + 50, columns=['AAPL', 'MSFT', 'BTC'])
    
    # Generate mock AI signals for the timeframe using the generator format
    signals_list = []
    for _ in range(500):
        aapl_sig = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        msft_sig = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        btc_sig  = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        signals_list.append([aapl_sig, msft_sig, btc_sig])
        
    mock_signals = pd.DataFrame(signals_list, columns=['AAPL', 'MSFT', 'BTC'])
    
    test_env = PortfolioEnv(mock_prices, mock_signals)
    agent = TFAgent(test_env)
    
    agent.train(10)
    agent.test(1)
