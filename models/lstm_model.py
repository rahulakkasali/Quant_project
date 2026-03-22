import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictorLSTM(tf.keras.Model):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        """
        TensorFlow/Keras LSTM Network for time-series forecasting.
        """
        super(PricePredictorLSTM, self).__init__()
        self.lstm_layers = []
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            self.lstm_layers.append(LSTM(hidden_size, return_sequences=return_seq))
            if dropout > 0 and num_layers > 1:
                self.lstm_layers.append(Dropout(dropout))
                
        self.dense = Dense(output_size)

    def call(self, inputs):
        """
        inputs shape: (batch_size, sequence_length, input_size)
        Returns shape: (batch_size, output_size)
        """
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x)
        return self.dense(x)

if __name__ == "__main__":
    model = PricePredictorLSTM(input_size=5)
    dummy_input = tf.random.normal((32, 60, 5)) # Batch of 32, seq_len 60, 5 features
    output = model(dummy_input)
    logger.info(f"LSTM Dummy Output shape: {output.shape}")
