import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ==========================================
# CREATE MODELS DIRECTORY
# ==========================================
os.makedirs("models", exist_ok=True)

# ==========================================
# SYNTHETIC DATASET
# ==========================================
np.random.seed(42)

samples = 2000

cycle = np.random.randint(0, 3000, samples)
temp = np.random.uniform(20, 50, samples)
ir = 0.02 + cycle * 0.00005 + np.random.normal(0, 0.002, samples)

tmin = temp - np.random.uniform(1, 3, samples)
tmax = temp + np.random.uniform(1, 3, samples)

charge_time = np.random.uniform(0.5, 5, samples)

# Simulated battery SoH
soh = (
    1
    - cycle * 0.00008
    - np.maximum(0, temp - 25) * 0.002
    - (ir - 0.02) * 0.4
)

soh = np.clip(soh, 0.4, 1.0)

# ==========================================
# FEATURES
# ==========================================
X = np.column_stack([
    ir,
    temp,
    tmin,
    tmax,
    charge_time,
    cycle
])

y = soh.reshape(-1, 1)

# ==========================================
# SCALERS
# ==========================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers
joblib.dump(
    scaler_X,
    "models/scaler_X.pkl"
)

joblib.dump(
    scaler_y,
    "models/scaler_y.pkl"
)

# ==========================================
# LSTM INPUT
# ==========================================
X_lstm = np.tile(
    X_scaled[:, np.newaxis, :],
    (1, 20, 1)
)

# ==========================================
# LSTM MODEL
# ==========================================
model = Sequential([
    LSTM(
        64,
        input_shape=(20, 6)
    ),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(0.001),
    loss="mse"
)

# ==========================================
# TRAIN
# ==========================================
model.fit(
    X_lstm,
    y_scaled,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# ==========================================
# SAVE LSTM MODEL
# ==========================================
model.save(
    "models/lstm_soh_model.keras"
)

# ==========================================
# SIMPLE DQN MODEL
# ==========================================
dqn = Sequential([
    Dense(64, activation="relu", input_shape=(4,)),
    Dense(32, activation="relu"),
    Dense(3, activation="linear")
])

dqn.compile(
    optimizer="adam",
    loss="mse"
)

# Dummy training data
states = np.random.rand(1000, 4)
targets = np.random.rand(1000, 3)

dqn.fit(
    states,
    targets,
    epochs=5,
    batch_size=32
)

# Save DQN
dqn.save(
    "models/hybrid_online_dqn.keras"
)

print("✅ All models generated successfully")