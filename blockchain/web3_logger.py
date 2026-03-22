from web3 import Web3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Web3TradeLogger:
    def __init__(self, rpc_url="http://127.0.0.1:8545", contract_address=None, abi=None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.is_connected = self.w3.is_connected()
        
        if not self.is_connected:
            logger.warning(f"Could not connect to {rpc_url}. Blockchain logging bypassed.")
            return
            
        logger.info(f"Connected to Web3 provider at {rpc_url}")
        self.account = self.w3.eth.accounts[0] if self.w3.eth.accounts else None
        
        if contract_address and abi:
            self.contract = self.w3.eth.contract(address=contract_address, abi=abi)
        else:
            self.contract = None
            logger.info("Contract not fully loaded (missing ABI/Address).")

    def log_trade(self, ticker: str, action: str, amount: float, price: float):
        if not self.is_connected or not self.contract or not self.account:
            logger.info(f"[Mock Chain Log] {action} {amount} {ticker} @ {price}")
            return None

        scale = 10**18
        amt_scaled = int(amount * scale)
        price_scaled = int(price * scale)

        try:
            tx_hash = self.contract.functions.logTrade(
                ticker, action, amt_scaled, price_scaled
            ).transact({'from': self.account})
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info(f"Trade confirmed on chain. Hash: {receipt.transactionHash.hex()}")
            return receipt.transactionHash.hex()
        except Exception as e:
            logger.error(f"Blockchain tx failed: {e}")
            return None

if __name__ == "__main__":
    logger_instance = Web3TradeLogger()
    logger_instance.log_trade("AAPL", "BUY", 10.5, 150.0)
