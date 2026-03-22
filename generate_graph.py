import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from execution.signal_generator import UnifiedSignalGenerator
from rl.env import PortfolioEnv

def run_mock_episode():
    generator = UnifiedSignalGenerator()
    
    # Mock data
    np.random.seed(42)
    mock_prices = pd.DataFrame(np.random.rand(100, 3) * 100 + 50, columns=['AAPL', 'MSFT', 'BTC'])
    
    # Generate mock AI signals
    signals_list = []
    for _ in range(100):
        aapl_sig = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        msft_sig = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        btc_sig  = generator.generate_signal(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))['signal']
        signals_list.append([aapl_sig, msft_sig, btc_sig])
        
    mock_signals = pd.DataFrame(signals_list, columns=['AAPL', 'MSFT', 'BTC'])
    
    # Env
    env = PortfolioEnv(mock_prices, mock_signals, initial_balance=10000)
    obs, info = env.reset()
    
    balances = [10000]
    done = False
    
    # Run episode (equally weighted mock action since TF is not installed)
    while not done:
        action = np.array([0.33, 0.33, 0.34])
        obs, reward, done, trunc, info = env.step(action)
        balances.append(info['balance'])
        
    # Plotting
    balances = np.array(balances)
    pct_change = ((balances - balances[0]) / balances[0]) * 100
    
    plt.figure(figsize=(10, 5))
    plt.plot(pct_change, color='#00ffcc', linewidth=2)
    plt.title("Portfolio Percentage Change (Mock Episode)", color='white')
    plt.xlabel("Step", color='white')
    plt.ylabel("% Change from Initial Balance", color='white')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Dark Mode aesthetic styling
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    plt.gcf().patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        
    # Save directly to artifact folder
    plt.tight_layout()
    plt.savefig("/Users/apple/.gemini/antigravity/brain/8c06494f-3ce9-4067-810c-bd4b18c74be5/portfolio_pct_change.png", facecolor='#1e1e1e', edgecolor='none')
    print("Graph saved successfully!")

if __name__ == "__main__":
    run_mock_episode()
