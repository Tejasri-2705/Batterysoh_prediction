import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib 
from utils.data_processing import load_scalers

print("Imports successful")
print(f"TensorFlow version: {tf.__version__}")

try:
    print("Attempting to load models...")
    models_dir = os.path.join(os.getcwd(), 'models')
    
    lstm_path = os.path.join(models_dir, 'lstm_soh_model.keras')
    dqn_path = os.path.join(models_dir, 'best_offline_dqn.keras')
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')

    if not os.path.exists(lstm_path):
        print(f"ERROR: LSTM model not found at {lstm_path}")
    if not os.path.exists(dqn_path):
        print(f"ERROR: DQN model not found at {dqn_path}")

    from tensorflow.keras.models import load_model
    
    # Load models with compile=False
    lstm_model = load_model(lstm_path, compile=False)
    print("LSTM Model loaded successfully")
    
    dqn_model = load_model(dqn_path, compile=False)
    print("DQN Model loaded successfully")
    
    scaler_X, scaler_y = load_scalers(scaler_x_path, scaler_y_path)
    print("Scalers loaded successfully")
    
    print("SMOKE TEST PASSED")

except Exception as e:
    print(f"SMOKE TEST FAILED: {e}")
    sys.exit(1)
