import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

def debug_models():
    base_dir = os.getcwd()
    models_dir = os.path.join(base_dir, 'models')
    
    # Load assets
    try:
        scaler_X = joblib.load(os.path.join(models_dir, 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(models_dir, 'scaler_y.pkl'))
        lstm_model = load_model(os.path.join(models_dir, 'lstm_soh_model.keras'), compile=False)
        dqn_model = load_model(os.path.join(models_dir, 'hybrid_online_dqn.keras'), compile=False)
        print("Models loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Test Case 1: Base
    # [IR, Tavg, Tmin, Tmax, ChargeTime, Cycle]
    inputs = [
        [0.05, 25.0, 23.0, 27.0, 1.5, 100],   # Healthy
        [0.10, 35.0, 33.0, 37.0, 1.5, 500],   # Degraded?
        [0.20, 45.0, 43.0, 47.0, 1.5, 1000],  # Critical?
        [0.01, 25.0, 23.0, 27.0, 1.5, 0]      # Brand new
    ]
    
    print("\n--- LSTM Sensitivity Probe ---")
    print(f"{'Input (IR, Temp, Cycle)':<30} | {'Raw SoH':<10} | {'Scaled SoH':<10}")
    
    for inp in inputs:
        raw_feat = np.array([inp])
        scaled_feat = scaler_X.transform(raw_feat)
        lstm_input = np.tile(scaled_feat, (20, 1)).reshape(1, 20, 6)
        
        pred_scaled = lstm_model.predict(lstm_input, verbose=0)[0][0]
        
        pred_raw = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
        
        print(f"{str(inp):<30} | {pred_raw:<10.4f} | {pred_scaled:<10.4f}")

    # Test DQN
    print("\n--- DQN Action Probe ---")
    print(f"{'State (SoH, Temp, Cycle, Curr)':<35} | {'Q-Values':<30} | {'Action'}")
    dqn_cases = [
        [1.0, 25.0, 100, 1.5],
        [0.8, 45.0, 500, 1.5],
        [0.6, 50.0, 1000, 1.5],
        [0.9, 25.0, 100, 5.0], # High current
    ]
    
    for case in dqn_cases:
        state = np.array([case])
        q_vals = dqn_model.predict(state, verbose=0)[0]
        action = np.argmax(q_vals)
        print(f"{str(case):<35} | {str(q_vals)} | {action}")

if __name__ == "__main__":
    debug_models()
