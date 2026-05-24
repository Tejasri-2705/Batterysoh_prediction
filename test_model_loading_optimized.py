#!/usr/bin/env python3
"""
Optimized Test script for load_all_models_and_scalers()
Removes Streamlit dependency for faster testing and adds timing.
"""

import os
import sys
import time
import numpy as np
from tensorflow.keras.models import load_model
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

print("=" * 60)
print("🚀 EV Battery Models - Optimized Loading Test")
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

all_exist = True
for name, path in required_files.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {os.path.basename(path)}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Missing files - aborting test")
    sys.exit(1)

# Load Scalers (fast)
print("\n📂 Loading Scalers...")
start_time = time.time()
try:
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    scaler_time = time.time() - start_time
    print(f"  ✅ Scalers loaded in {scaler_time:.3f}s")
except Exception as e:
    print(f"  ❌ Error loading scalers: {e}")
    sys.exit(1)

# Load LSTM Model
print("\n📂 Loading LSTM SoH Model...")
start_time = time.time()
try:
    lstm_model = load_model(lstm_path, compile=False)
    lstm_time = time.time() - start_time
    print(f"  ✅ LSTM Model loaded in {lstm_time:.3f}s")
    print(f"  🧠 Input shape: {lstm_model.input_shape}")
    print(f"  🧠 Output shape: {lstm_model.output_shape}")
except Exception as e:
    print(f"  ❌ Error loading LSTM model: {e}")
    sys.exit(1)

# Load DQN Model
print("\n📂 Loading DQN Charging Model...")
start_time = time.time()
try:
    dqn_model = load_model(dqn_path, compile=False)
    dqn_time = time.time() - start_time
    print(f"  ✅ DQN Model loaded in {dqn_time:.3f}s")
    print(f"  🧠 Input shape: {dqn_model.input_shape}")
    print(f"  🧠 Output shape: {dqn_model.output_shape}")
except Exception as e:
    print(f"  ❌ Error loading DQN model: {e}")
    sys.exit(1)

# Warm up models (optional - first prediction is slower)
print("\n⚡ Warming up models...")
start_time = time.time()
try:
    dummy_lstm = np.zeros((1, 20, 6))
    lstm_pred = lstm_model.predict(dummy_lstm, verbose=0)
    
    dummy_dqn = np.zeros((1, 4))
    dqn_pred = dqn_model.predict(dummy_dqn, verbose=0)
    
    warmup_time = time.time() - start_time
    print(f"  ✅ Warmup completed in {warmup_time:.3f}s")
    print(f"  📊 LSTM output shape: {lstm_pred.shape}")
    print(f"  📊 DQN output shape: {dqn_pred.shape}")
except Exception as e:
    print(f"  ❌ Warmup error: {e}")
    sys.exit(1)

# Test prediction
print("\n🧪 Test Prediction...")
try:
    test_features = np.array([[0.05, 30.0, 25.0, 35.0, 1.5, 100]])
    scaled_features = scaler_X.transform(test_features)
    sequence = np.tile(scaled_features, (20, 1))
    lstm_input = sequence.reshape(1, 20, 6)
    
    start_time = time.time()
    lstm_output = lstm_model.predict(lstm_input, verbose=0)[0][0]
    pred_time = time.time() - start_time
    
    soh_scaled = np.array([[lstm_output]])
    soh_original = scaler_y.inverse_transform(soh_scaled)[0][0]
    soh_percentage = float(np.clip(soh_original * 100, 0, 100))
    
    print(f"  ✅ Prediction in {pred_time:.3f}s")
    print(f"  🔋 Predicted SoH: {soh_percentage:.2f}%")
    
    # Test DQN
    dqn_input = np.array([[soh_percentage/100.0, 30.0, 100, 1.5]])
    start_time = time.time()
    q_values = dqn_model.predict(dqn_input, verbose=0)[0]
    dqn_pred_time = time.time() - start_time
    
    best_action = np.argmax(q_values)
    action_labels = ["Decrease Current", "Maintain Current", "Increase Current"]
    print(f"  ⚡ DQN prediction in {dqn_pred_time:.3f}s")
    print(f"  ⚡ Best Action: {action_labels[best_action]}")
    print(f"  📊 Q-values: {q_values}")
    
except Exception as e:
    print(f"  ❌ Prediction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("📊 LOADING TIME SUMMARY")
print("=" * 60)
print(f"  Scalers:     {scaler_time:.3f}s")
print(f"  LSTM Model:  {lstm_time:.3f}s")
print(f"  DQN Model:   {dqn_time:.3f}s")
print(f"  Warmup:      {warmup_time:.3f}s")
print(f"  Prediction:  {pred_time:.3f}s")
print(f"  DQN Pred:    {dqn_pred_time:.3f}s")
print("-" * 60)
total = scaler_time + lstm_time + dqn_time + warmup_time
print(f"  TOTAL:       {total:.3f}s")
print("=" * 60)
print("✅ All tests passed!")
print("=" * 60)

