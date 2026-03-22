from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize FinBERT model for financial sentiment analysis.
        Uses transformers pipeline for streamlined execution.
        """
        logger.info(f"Loading sentiment model: {model_name}. This may take a moment to download weights on first run.")
        self.pipe = pipeline("text-classification", model=model_name)
    
    def analyze(self, texts):
        """
        Input: list of strings or single string.
        Returns: list of dictionaries with 'label' and 'score'.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        logger.info(f"Analyzing sentiment for {len(texts)} texts.")
        return self.pipe(texts)

    def get_aggregate_sentiment_score(self, texts) -> float:
        """
        Returns a single scalar (-1 to 1) representing the average sentiment.
        Useful as a real-time trading signal feature.
        """
        if not texts:
            return 0.0
            
        results = self.analyze(texts)
        score = 0.0
        for res in results:
            if res['label'] == 'positive':
                score += res['score']
            elif res['label'] == 'negative':
                score -= res['score']
        return score / len(texts)

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    news = [
        "Company earnings beat expectations, stock surges.",
        "Market remains uncertain amid inflation fears.",
        "The recent scandal exposes deep financial issues within the firm.",
        "A regular day on Wall Street with flat trading volumes."
    ]
    results = analyzer.analyze(news)
    for t, r in zip(news, results):
        print(f"News: '{t}' | Sentiment: {r['label']} (Score: {r['score']:.2f})")
    
    agg = analyzer.get_aggregate_sentiment_score(news)
    print(f"\nAggregate Sentiment Score: {agg:.2f}")
