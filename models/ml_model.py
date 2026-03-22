from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShortTermPredictor:
    def __init__(self, model_type="xgboost"):
        """
        Initializes an ML model for short-term price movement predictions.
        Supported types: 'xgboost', 'random_forest'.
        """
        self.model_type = model_type.lower()
        if self.model_type == "xgboost":
            self.model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError("Unsupported model type. Use 'xgboost' or 'random_forest'.")
        
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Train the model on standard 2D feature arrays.
        """
        logger.info(f"Training {self.model_type.upper()} on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.model_type.upper()} Training complete.")

    def predict(self, X):
        """
        Predict returns or prices.
        """
        if not self.is_trained:
            logger.warning("predict() called, but model is not trained. Results usually random if untrained algorithm permits, or throws error.")
        logger.info(f"Predicting for {X.shape[0]} samples.")
        return self.model.predict(X)
        
    def save_model(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"Model successfully saved to {path}.")
        
    def load_model(self, path: str):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"Model successfully loaded from {path}.")
        else:
            logger.error(f"Cannot load model. Path {path} not found.")

if __name__ == "__main__":
    import numpy as np
    
    # Dummy data test
    X_dummy = np.random.rand(100, 10) # 100 samples, 10 features
    y_dummy = np.random.rand(100)
    
    # Train
    predictor = ShortTermPredictor("xgboost")
    predictor.train(X_dummy, y_dummy)
    
    # Predict
    preds = predictor.predict(X_dummy[:5])
    print(f"First 5 predictions: {preds}")
