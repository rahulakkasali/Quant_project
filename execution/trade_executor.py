import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockOrderExecutor:
    """
    Simulated order execution engine that logs trades and updates portfolio balances locally.
    """
    def __init__(self, initial_balance=100000.0):
        self.balance = initial_balance
        self.portfolio: Dict[str, float] = {}
        self.avg_cost: Dict[str, float] = {} # Tracks the rolling average entry price
        logger.info(f"Mock Order Executor initialized. Balance: ${self.balance:.2f}")

    def execute_trade(self, action: str, ticker: str, amount: float, price: float) -> dict:
        cost = amount * price
        
        if action == 'BUY':
            if cost <= self.balance:
                self.balance -= cost
                
                # Calculate rolling average entry cost basis
                current_amount = self.portfolio.get(ticker, 0)
                current_avg = self.avg_cost.get(ticker, 0)
                new_total = current_amount + amount
                if new_total > 0:
                    self.avg_cost[ticker] = ((current_avg * current_amount) + (price * amount)) / new_total
                    
                self.portfolio[ticker] = new_total
                logger.info(f"EXECUTED BUY: {amount:.2f} {ticker} @ ${price:.2f} | Remaining Balance: ${self.balance:.2f}")
                return {'success': True, 'profit': 0.0, 'entry_price': price}
            else:
                logger.warning(f"FAILED BUY: Insufficient funds for {ticker}.")
                return {'success': False}
                
        elif action == 'SELL':
            if self.portfolio.get(ticker, 0) >= amount:
                self.balance += cost
                self.portfolio[ticker] -= amount
                
                # Calculate True Realized Profit based on average entry price
                entry_price = self.avg_cost.get(ticker, price)
                profit = (price - entry_price) * amount
                
                # Cleanup if squared off completely
                if self.portfolio[ticker] < 1e-6:
                    self.portfolio[ticker] = 0.0
                    self.avg_cost[ticker] = 0.0
                    
                logger.info(f"EXECUTED SELL: {amount:.2f} {ticker} @ ${price:.2f} | Profit: ${profit:.2f} | Remaining Balance: ${self.balance:.2f}")
                return {'success': True, 'profit': profit, 'entry_price': entry_price}
            else:
                logger.warning(f"FAILED SELL: Insufficient {ticker} holding {self.portfolio.get(ticker, 0)}.")
                return {'success': False}
        
        return {'success': False}

class MetaTraderExecutor:
    """
    Real execution bridge connecting to MetaTrader 5 terminal.
    NOTE: MetaTrader5 native Python library is Windows-only. 
    It will fail gracefully on MacOS returning False for trades.
    """
    def __init__(self, login, password, server):
        self.login = int(login)
        self.password = password
        self.server = server
        self.connected = False
        self.balance = 100000.0  # Fallback for local tracking
        self.portfolio: Dict[str, float] = {}
        self.avg_cost: Dict[str, float] = {}
        
        try:
            import MetaTrader5 as mt5
            logger.info("Initializing MetaTrader 5...")
            if not mt5.initialize(login=self.login, password=self.password, server=self.server):
                logger.error(f"MT5 Init Failed. Likely wrong credentials or operating system mismatch. Error: {mt5.last_error()}")
            else:
                self.connected = True
                
                # Try to sync real balance from MT5 account
                account_info = mt5.account_info()
                if account_info:
                    self.balance = account_info.balance
                    logger.info(f"MT5 Connected! Real Balance Synced: ${self.balance:.2f}")
                else:
                    logger.warning("Could not retrieve account info.")
                    
        except ImportError:
            logger.error("MetaTrader5 package is not installed (or not supported on this OS). Cannot connect.")

    def execute_trade(self, action: str, ticker: str, amount: float, price: float) -> dict:
        if not self.connected:
            logger.error("MT5 not connected. Trade simulated locally only.")
            return {'success': False}
            
        import MetaTrader5 as mt5
        
        symbol = ticker.replace("-USD", "") # Common fix for standard broker crypto symbols
        
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Symbol {symbol} not available in your MT5 broker.")
            return {'success': False}
            
        order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(amount),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Autonomous AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"MT5 Trade Rejected: {result.comment}")
            return {'success': False}
            
        logger.info(f"MT5 LIVE {action} Executed: {amount} {symbol} at {price}. Ticket: {result.order}")
        
        # Keep local representation updated
        cost = amount * price
        
        if action == 'BUY':
            self.balance -= cost
            
            current_amount = self.portfolio.get(ticker, 0)
            current_avg = self.avg_cost.get(ticker, 0)
            new_total = current_amount + amount
            if new_total > 0:
                self.avg_cost[ticker] = ((current_avg * current_amount) + (price * amount)) / new_total
                
            self.portfolio[ticker] = new_total
            return {'success': True, 'profit': 0.0, 'entry_price': price}
            
        else:
            self.balance += cost
            self.portfolio[ticker] -= amount
            
            entry_price = self.avg_cost.get(ticker, price)
            profit = (price - entry_price) * amount
            
            if self.portfolio[ticker] < 1e-6:
                self.portfolio[ticker] = 0.0
                self.avg_cost[ticker] = 0.0
                
            return {'success': True, 'profit': profit, 'entry_price': entry_price}
