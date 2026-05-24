#!/usr/bin/env python3
"""
Test script to run load_all_models_and_scalers() directly
and display results without starting Streamlit server
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

print("=" * 60)
print("EV Battery Models Loading Test")
print("=" * 60)

# Define file paths
lstm_path = os.path.join(models_dir, 'lstm_soh_model.keras')
dqn_path = os.path.join(models_dir, 'best_offline_dqn.keras')
scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')

# Check files exist
print("\n📁 Checking model files...")
required_files = {
    'LSTM SoH Model': lstm_path,
    'DQN Charging Model': dqn_path,
    'Feature Scaler (X)': scaler_x_path,
    'Target Scaler (y)': scaler_y_path
}

missing = []
for name, path in required_files.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {os.path.basename(path)}")
    if not exists:
        missing.append(name)

if missing:
    print(f"\n❌ Missing files: {missing}")
    sys.exit(1)

# Load Scalers
print("\n📂 Loading Scalers...")
try:
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    print("  ✅ Scalers loaded successfully!")
    print(f"  📊 scaler_X data_min: {scaler_X.data_min_}")
    print(f"  📊 scaler_X data_max: {scaler_X.data_max_}")
    print(f"  📊 scaler_y data_min: {scaler_y.data_min_}")
    print(f"  📊 scaler_y data_max: {scaler_y.data_max_}")
except Exception as e:
    print(f"  ❌ Error loading scalers: {e}")
    sys.exit(1)

# Load LSTM Model
print("\n📂 Loading LSTM SoH Model...")
try:
    lstm_model = load_model(lstm_path, compile=False)
    print("  ✅ LSTM Model loaded successfully!")
    print(f"  🧠 Model input shape: {lstm_model.input_shape}")
    print(f"  🧠 Model output shape: {lstm_model.output_shape}")
except Exception as e:
    print(f"  ❌ Error loading LSTM model: {e}")
    sys.exit(1)

# Load DQN Model
print("\n📂 Loading DQN Charging Model...")
try:
    dqn_model = load_model(dqn_path, compile=False)
    print("  ✅ DQN Model loaded successfully!")
    print(f"  🧠 Model input shape: {dqn_model.input_shape}")
    print(f"  🧠 Model output shape: {dqn_model.output_shape}")
except Exception as e:
    print(f"  ❌ Error loading DQN model: {e}")
    sys.exit(1)

# Warm up models
print("\n⚡ Warming up models...")
try:
    dummy_lstm = np.zeros((1, 20, 6))
    lstm_pred = lstm_model.predict(dummy_lstm, verbose=0)
    print(f"  ✅ LSTM warmup - output shape: {lstm_pred.shape}")
    
    dummy_dqn = np.zeros((1, 4))
    dqn_pred = dqn_model.predict(dummy_dqn, verbose=0)
    print(f"  ✅ DQN warmup - output shape: {dqn_pred.shape}")
    print(f"  📊 DQN Q-values (dummy): {dqn_pred[0]}")
except Exception as e:
    print(f"  ❌ Warmup error: {e}")
    sys.exit(1)

# Test prediction
print("\n🧪 Test Prediction...")
try:
    # Example input features
    test_features = np.array([[0.05, 30.0, 25.0, 35.0, 1.5, 100]])  # IR, Tavg, Tmin, Tmax, chargetime, cycle
    scaled_features = scaler_X.transform(test_features)
    
    # Create sequence for LSTM
    sequence = np.tile(scaled_features, (20, 1))
    lstm_input = sequence.reshape(1, 20, 6)
    
    # Get prediction
    lstm_output = lstm_model.predict(lstm_input, verbose=0)[0][0]
    
    # Inverse scale SoH
    soh_scaled = np.array([[lstm_output]])
    soh_original = scaler_y.inverse_transform(soh_scaled)[0][0]
    soh_percentage = float(np.clip(soh_original * 100, 0, 100))
    
    print(f"  ✅ Prediction successful!")
    print(f"  🔋 Predicted SoH: {soh_percentage:.2f}%")
    
    # Test DQN
    dqn_input = np.array([[soh_percentage/100.0, 30.0, 100, 1.5]])
    q_values = dqn_model.predict(dqn_input, verbose=0)[0]
    best_action = np.argmax(q_values)
    
    action_labels = ["Decrease Current", "Maintain Current", "Increase Current"]
    print(f"  ⚡ Best Action: {action_labels[best_action]}")
    print(f"  📊 Q-values: {q_values}")
    
except Exception as e:
    print(f"  ❌ Prediction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All models and scalers loaded successfully!")
print("=" * 60)

