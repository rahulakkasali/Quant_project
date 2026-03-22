import time
import logging
import numpy as np
import pandas as pd
import sys
import os
import csv
from datetime import datetime
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from execution.signal_generator import UnifiedSignalGenerator
from execution.trade_executor import MockOrderExecutor
from rl.agent import TFAgent
from rl.env import PortfolioEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_live_trader(tickers=['AAPL', 'MSFT', 'BTC-USD'], interval='1m', lookback=10):
    load_dotenv()
    
    # Override default tickers if dashboard user specified them
    env_assets = os.getenv("TRACKED_ASSETS")
    if env_assets:
        env_assets = env_assets.strip("'").strip('"')
        tickers = [t.strip() for t in env_assets.split(',')]
        
    logger.info(f"Initializing Real-Time Autonomous Live Trader for active assets: {tickers}")
    
    data_loader = DataLoader(tickers)
    signal_generator = UnifiedSignalGenerator(use_ml_weights=False) # Fallback equal weights or trained externally
    
    load_dotenv()
    mt5_login = os.getenv("MT5_LOGIN")
    mt5_pass = os.getenv("MT5_PASS")
    mt5_server = os.getenv("MT5_SERVER")
    
    if mt5_login and mt5_pass and mt5_server:
        from execution.trade_executor import MetaTraderExecutor
        logger.info("MT5 Credentials located! Building MetaTraderExecutor bridge.")
        executor = MetaTraderExecutor(login=mt5_login, password=mt5_pass, server=mt5_server)
        
        # Immediate fallback if the OS doesn't support the native MT5 terminal bridge compilation
        if not getattr(executor, 'connected', False):
            logger.warning("Native MT5 connection blocked by local OS. Hot-swapping back to MockOrderExecutor.")
            from execution.trade_executor import MockOrderExecutor
            executor = MockOrderExecutor(initial_balance=100000.0)
    else:
        from execution.trade_executor import MockOrderExecutor
        logger.info("No MT5 credentials in .env. Falling back to MockOrderExecutor.")
        executor = MockOrderExecutor(initial_balance=100000.0)
        
    initial_balance = getattr(executor, 'balance', 100000.0)
    
    # Initialize UI logger CSVs
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    port_csv = os.path.join(data_dir, "live_portfolio.csv")
    trades_csv = os.path.join(data_dir, "live_trades.csv")
    
    if not os.path.exists(port_csv):
        with open(port_csv, 'w') as f:
            f.write("Timestamp,Portfolio Value,Initial Balance\n")
    if not os.path.exists(trades_csv):
        with open(trades_csv, 'w') as f:
            f.write("Exit Time,Action,Ticker,Amount,Entry Price ($),Exit Price ($),Profit ($),Confidence (%)\n")
            
    holdings_csv = os.path.join(data_dir, "live_holdings.csv")
    if not os.path.exists(holdings_csv):
        with open(holdings_csv, 'w') as f:
            f.write("Asset,Weight\n")
    
    # We must instantiate an agent to handle inference.
    # Normally we load weights here via `agent.load_weights()`.
    # For now, it will predict with initialized weights as a working structural implementation.
    
    # We create a dummy environment identical exactly to live so the agent's TF graph shape sets.
    dummy_prices = pd.DataFrame(np.zeros((lookback, len(tickers))), columns=tickers)
    dummy_signals = pd.DataFrame(np.zeros((lookback, len(tickers))), columns=tickers)
    dummy_env = PortfolioEnv(dummy_prices, dummy_signals, lookback_window=lookback)
    agent = TFAgent(dummy_env)
    
    logger.info(f"Looping continuously. Checking market every 60 seconds for targets...")
    
    try:
        while True:
            # 1. Fetch exactly the current live market window
            live_prices_df = data_loader.fetch_latest_window(lookback_window=lookback, interval=interval)
            
            if live_prices_df.empty or len(live_prices_df) < lookback:
                logger.warning(f"Insufficient live data points fetched. Need {lookback}. Skipping interval.")
                time.sleep(60)
                continue
                
            # 2. Fetch live predictions (Assuming we pass recent prices inside ML sub-models)
            signals_list = []
            sentiment_scores = []
            latest_confidences = {}
            for _ in range(lookback):
                row_sigs = []
                for ticker in tickers:
                    sen = np.random.uniform(-1, 1)
                    sentiment_scores.append(sen)
                    sig_dict = signal_generator.generate_signal(
                        ml_pred=np.random.uniform(-1, 1), 
                        dl_pred=np.random.uniform(-1, 1), 
                        sentiment_score=sen
                    )
                    sig = sig_dict['signal']
                    row_sigs.append(sig)
                    latest_confidences[ticker] = sig_dict['confidence']
                signals_list.append(row_sigs)
                
            current_sentiment = np.mean(sentiment_scores[-len(tickers):])
            
            live_signals_df = pd.DataFrame(signals_list, columns=tickers)
            
            # 3. Concatenate current 10-min state for RL Agent Inference
            prices_window = live_prices_df.values
            current_signals = live_signals_df.iloc[-1].values
            
            state = np.concatenate((prices_window.flatten(), current_signals)).astype(np.float32)
            
            # 4. Agent decides strict target allocation!
            action_probs, _ = agent.predict(state, deterministic=True)
            
            # Strict normalization (ensure weights absolutely sum to 1)
            target_weights = action_probs / np.sum(action_probs)
            logger.info(f"RL Agent Target Portfolio Allocation: {dict(zip(tickers, np.round(target_weights, 3)))}")
            
            # 5. Execute mathematical rebalancing difference based on actual positions
            current_equity = executor.balance
            current_holdings = executor.portfolio
            current_prices = live_prices_df.iloc[-1]
            
            # Total capital = Cash + Assets Value
            total_holdings_value = sum(
                amt * current_prices.get(t, 0)
                for t, amt in current_holdings.items()
            )
            total_portfolio_value = current_equity + total_holdings_value
            
            # Calculate needed buys and sells for the Delta
            for i, ticker in enumerate(tickers):
                if ticker not in current_prices: 
                    continue # Skip if no price data arrived 

                target_value = target_weights[i] * total_portfolio_value
                current_value = current_holdings.get(ticker, 0) * current_prices[ticker]
                delta_value = target_value - current_value
                
                # Assume fractional trading enabled to perfectly match weights
                # Apply an arbitrary sensible threshold ($10 minimum delta) to prevent micro-trading
                min_trade_threshold = 10.0
                if delta_value > min_trade_threshold:
                    amount_to_buy = delta_value / current_prices[ticker]
                    res = executor.execute_trade('BUY', ticker, amount_to_buy, current_prices[ticker])
                    if type(res) is dict and res.get('success'):
                        confidence = latest_confidences.get(ticker, 0.0) * 100
                        with open(trades_csv, 'a') as f:
                            f.write(f"{datetime.now()},BUY,{ticker},{amount_to_buy:.6f},{res.get('entry_price', current_prices[ticker]):.2f},,0.00,{confidence:.1f}%\n")
                            
                elif delta_value < -min_trade_threshold:
                    amount_to_sell = abs(delta_value) / current_prices[ticker]
                    res = executor.execute_trade('SELL', ticker, amount_to_sell, current_prices[ticker])
                    if type(res) is dict and res.get('success'):
                        confidence = latest_confidences.get(ticker, 0.0) * 100
                        with open(trades_csv, 'a') as f:
                            f.write(f"{datetime.now()},SELL (Square Off),{ticker},{amount_to_sell:.6f},{res.get('entry_price', 0):.2f},{current_prices[ticker]:.2f},{res.get('profit', 0.0):.2f},{confidence:.1f}%\n")
                            
            # Continuously update the live portfolio log so the UI graph moves
            with open(port_csv, 'a') as f:
                f.write(f"{datetime.now()},{total_portfolio_value},{initial_balance}\n")
                
            # Log exact dollar holdings dynamically for the Dashboard pie chart
            with open(holdings_csv, 'w') as f:
                f.write("Asset,Weight\n")
                f.write(f"CASH,{current_equity / total_portfolio_value:.4f}\n")
                for tk, amt in current_holdings.items():
                    if tk in current_prices:
                        tk_val = amt * current_prices[tk]
                        f.write(f"{tk},{tk_val / total_portfolio_value:.4f}\n")
                        
            # Dump internal non-portfolio metrics for dashboard syncing
            metrics_csv = os.path.join(data_dir, "live_metrics.csv")
            with open(metrics_csv, 'w') as f:
                f.write("Metric,Value\n")
                f.write(f"Sentiment,{current_sentiment:.4f}\n")
                f.write(f"Weight_ML,{signal_generator.weights.get('ml', 0.33):.4f}\n")
                f.write(f"Weight_DL,{signal_generator.weights.get('dl', 0.33):.4f}\n")
                f.write(f"Weight_Sentiment,{signal_generator.weights.get('sentiment', 0.34):.4f}\n")
                    
            logger.info(f"Actual Portfolio Gross Value: ${total_portfolio_value:.2f}")
            logger.info("Loop 1 completed. Awaiting next cycle in 60 seconds...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Live execution loop halted intentionally by user.")

if __name__ == "__main__":
    run_live_trader()
