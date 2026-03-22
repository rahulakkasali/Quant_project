import logging
import numpy as np
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedSignalGenerator:
    def __init__(self, use_ml_weights=True):
        """
        Combines predictions from ML, Deep Learning, and FinBERT Sentiment into one signal.
        If use_ml_weights=True, a meta-model learns the optimal weights combining the sub-models.
        """
        self.use_ml_weights = use_ml_weights
        self.meta_model = Ridge(positive=True) # Forces positive coefficients 
        self.is_trained = False
        
        # Fallback equal weights before training
        self.weights = {'ml': 0.333, 'dl': 0.333, 'sentiment': 0.334}
        logger.info(f"Initialized UnifiedSignalGenerator. ML Weight Derivation Active: {use_ml_weights}")

    def fit(self, ml_preds: np.ndarray, dl_preds: np.ndarray, sentiment_preds: np.ndarray, actual_returns: np.ndarray):
        """
        Trains the meta-model to find the optimal ensemble weights based on historical accuracy.
        Inputs should be 1D arrays of historical predictions and the true asset returns.
        """
        X = np.column_stack((ml_preds, dl_preds, sentiment_preds))
        self.meta_model.fit(X, actual_returns)
        
        # Normalize learned coefficients to sum to 1 for transparent weighting
        coefs = self.meta_model.coef_
        coef_sum = np.sum(coefs)
        if coef_sum > 0:
            coefs = coefs / coef_sum
        else:
            coefs = [0.333, 0.333, 0.334] # Fallback if all 0
            
        self.weights = {
            'ml': coefs[0],
            'dl': coefs[1],
            'sentiment': coefs[2]
        }
        self.is_trained = True
        logger.info(f"ML derived optimal signal weights: {self.weights}")

    def generate_signal(self, ml_pred: float, dl_pred: float, sentiment_score: float) -> dict:
        """
        Generates unified signal [-1.0, 1.0] and a confidence score [0.0, 1.0].
        Assumes inputs are already normalized between -1 and 1.
        """
        score = (
            self.weights.get('ml', 0) * ml_pred +
            self.weights.get('dl', 0) * dl_pred +
            self.weights.get('sentiment', 0) * sentiment_score
        )
        final_signal = max(min(score, 1.0), -1.0)
        confidence = abs(final_signal)
        
        action = "HOLD"
        if final_signal > 0.2:
            action = "BUY"
        elif final_signal < -0.2:
            action = "SELL"
            
        return {
            "signal": final_signal,
            "confidence": confidence,
            "action": action
        }

if __name__ == "__main__":
    generator = UnifiedSignalGenerator()
    
    # 1. Simulate historical training data so the ML can derive the weights
    # Imagine ML has 80% accuracy, DL has 60% accuracy, Sentiment is noisy
    np.random.seed(42)
    hist_returns = np.random.normal(0, 0.05, 1000)
    
    # ML is highly correlated with actual returns
    ml_historical = hist_returns + np.random.normal(0, 0.02, 1000)
    # DL is somewhat correlated
    dl_historical = hist_returns + np.random.normal(0, 0.06, 1000)
    # Sentiment is loosely correlated
    sen_historical = hist_returns + np.random.normal(0, 0.1, 1000)
    
    generator.fit(ml_historical, dl_historical, sen_historical, hist_returns)
    
    # 2. Generate a new signal using the ML-derived weights
    result = generator.generate_signal(ml_pred=0.8, dl_pred=-0.2, sentiment_score=0.9)
    print(f"Derived Weights -> ML: {generator.weights['ml']:.2f}, DL: {generator.weights['dl']:.2f}, Sentiment: {generator.weights['sentiment']:.2f}")
    print(f"Test Action: {result['action']} (Confidence: {result['confidence']:.2f})")
