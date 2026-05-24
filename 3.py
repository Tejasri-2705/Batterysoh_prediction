# ============================================================
# CALIBRATED GRU + DOUBLE DQN (FINAL INTEGRATION)
# ============================================================

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib

print("🚀 Calibrated GRU + Double DQN Started")

# ------------------------------------------------------------
# LOAD GRU & SCALERS
# ------------------------------------------------------------
GRU_PATH = "/content/drive/MyDrive/models/gru_soh_model.keras"
SCALER_X_PATH = "/content/drive/MyDrive/models/gru_scaler_X.pkl"
SCALER_Y_PATH = "/content/drive/MyDrive/models/gru_scaler_y.pkl"

gru = load_model(GRU_PATH, compile=False)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

WINDOW = 20

# ------------------------------------------------------------
# PHYSICS-INFORMED CALIBRATION
# ------------------------------------------------------------
def calibrate_soh(soh, cycle, cycle_life=800, alpha=0.8):
    factor = np.exp(-alpha * (cycle / cycle_life))
    return float(np.clip(soh * factor, 0.65, 1.0))

# ------------------------------------------------------------
# GRU SoH PREDICTION
# ------------------------------------------------------------
def predict_calibrated_soh(features, cycle):
    seq = np.tile(features, (WINDOW, 1))
    seq = scaler_X.transform(seq)
    seq = seq.reshape(1, WINDOW, 6)

    soh_scaled = gru.predict(seq, verbose=0)
    soh = scaler_y.inverse_transform(soh_scaled)[0][0]
    return calibrate_soh(soh, cycle)

# ------------------------------------------------------------
# DOUBLE DQN MODEL
# ------------------------------------------------------------
def build_q_network():
    inputs = Input(shape=(4,))
    x = Dense(64, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(3, activation="linear")(x)
    return Model(inputs, outputs)

q_net = build_q_network()
target_q_net = build_q_network()
target_q_net.set_weights(q_net.get_weights())

q_net.compile(optimizer=Adam(0.001), loss="mse")

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------
class BatteryEnv:
    def reset(self):
        self.cycle = random.randint(20, 700)
        self.temp = random.uniform(25, 40)
        self.current = random.uniform(6, 14)
        self.features = np.random.uniform(0.2, 0.8, size=6)

        self.soh = predict_calibrated_soh(self.features, self.cycle)
        return self.state()

    def state(self):
        return np.array([self.soh, self.temp, self.cycle, self.current], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.current = max(5, self.current - 1)
        elif action == 2:
            self.current = min(18, self.current + 1)

        self.temp += 0.15 * self.current
        self.cycle += 1

        prev_soh = self.soh
        self.soh = predict_calibrated_soh(self.features, self.cycle)

        # ---------------- Reward ----------------
        reward = 0
        if self.soh >= prev_soh:
            reward += 2
        else:
            reward -= 2

        if self.temp > 40:
            reward -= 3
        if self.temp > 45:
            reward -= 4

        if self.current > 14 and self.soh < 0.8:
            reward -= 2

        done = self.temp > 55 or self.soh <= 0.65
        return self.state(), reward, done

# ------------------------------------------------------------
# TRAINING LOOP (DOUBLE DQN)
# ------------------------------------------------------------
env = BatteryEnv()

MEMORY = []
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
BATCH = 64
EPISODES = 150
TARGET_UPDATE = 10

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for _ in range(50):
        if random.random() < EPSILON:
            action = random.randint(0, 2)
        else:
            action = np.argmax(q_net.predict(state.reshape(1, -1), verbose=0))

        next_state, reward, done = env.step(action)
        MEMORY.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(MEMORY) > BATCH:
            batch = random.sample(MEMORY, BATCH)

            s = np.array([b[0] for b in batch])
            a = np.array([b[1] for b in batch])
            r = np.array([b[2] for b in batch])
            s2 = np.array([b[3] for b in batch])
            d = np.array([b[4] for b in batch])

            q_vals = q_net.predict(s, verbose=0)
            next_actions = np.argmax(q_net.predict(s2, verbose=0), axis=1)
            q_target = target_q_net.predict(s2, verbose=0)

            for i in range(BATCH):
                q_vals[i][a[i]] = r[i] if d[i] else r[i] + GAMMA * q_target[i][next_actions[i]]

            q_net.fit(s, q_vals, verbose=0)

        if done:
            break

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    if ep % TARGET_UPDATE == 0:
        target_q_net.set_weights(q_net.get_weights())

    if (ep + 1) % 20 == 0:
        print(f"Episode {ep+1}/{EPISODES} | Reward: {total_reward:.2f}")

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------
q_net.save("/content/drive/MyDrive/models/double_dqn_calibrated.keras")
print("✅ Double DQN with calibrated SoH saved")