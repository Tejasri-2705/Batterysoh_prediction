import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

print(f"TensorFlow Version: {tf.__version__}")

# Define a custom InputLayer that handles 'batch_shape'
class PatchedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        # The error is "Unrecognized keyword arguments: ['batch_shape']"
        # So we pop it from kwargs before calling super
        if 'batch_shape' in kwargs:
            print(f"Removing batch_shape: {kwargs['batch_shape']}")
            kwargs.pop('batch_shape')
        super().__init__(**kwargs)
    
    # We might need get_config to be safe, but usually loading just needs init
    def get_config(self):
        config = super().get_config()
        return config

models_dir = os.path.join(os.getcwd(), 'models')
lstm_path = os.path.join(models_dir, 'lstm_soh_model.keras')
dqn_path = os.path.join(models_dir, 'best_offline_dqn.keras')

print("\n--- Trying to load LSTM Model with Custom Object ---")
try:
    lstm_model = load_model(lstm_path, custom_objects={'InputLayer': PatchedInputLayer}, compile=False)
    print("SUCCESS: LSTM Model loaded with patch.")
except Exception as e:
    print(f"FAILED: LSTM Model load error: {e}")

print("\n--- Trying to load DQN Model with Custom Object ---")
try:
    dqn_model = load_model(dqn_path, custom_objects={'InputLayer': PatchedInputLayer}, compile=False)
    print("SUCCESS: DQN Model loaded with patch.")
except Exception as e:
    print(f"FAILED: DQN Model load error: {e}")
