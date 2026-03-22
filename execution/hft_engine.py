import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFTEngine:
    def __init__(self, threshold=0.5):
        """
        Basic High-Frequency Trading simulator engine.
        Executes mock trades based on unified signal strength crossing a threshold.
        """
        self.threshold = threshold
        self.positions = {}
        logger.info(f"HFT Engine initialized with threshold: {self.threshold}")

    def evaluate_signal(self, ticker: str, unified_signal: float, current_price: float):
        """
        Evaluate if signal translates into BUY, SELL, or HOLD.
        """
        action = 'HOLD'
        if unified_signal > self.threshold:
            action = 'BUY'
            self.positions[ticker] = self.positions.get(ticker, 0) + 1
            logger.info(f"[{ticker}] HFT Engine BUY. Price: {current_price}, Signal: {unified_signal:.2f}")
        elif unified_signal < -self.threshold:
            action = 'SELL'
            if self.positions.get(ticker, 0) > 0:
                self.positions[ticker] -= 1
            logger.info(f"[{ticker}] HFT Engine SELL. Price: {current_price}, Signal: {unified_signal:.2f}")
        else:
            logger.debug(f"[{ticker}] Signal {unified_signal:.2f} weak. HOLD.")
            
        return action

if __name__ == "__main__":
    engine = HFTEngine(threshold=0.6)
    action1 = engine.evaluate_signal("AAPL", 0.75, 150.00)
    action2 = engine.evaluate_signal("AAPL", -0.80, 152.00)
    action3 = engine.evaluate_signal("BTC-USD", 0.20, 95000.00)
    print(f"Final mock positions: {engine.positions}")
