import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import joblib

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="EV Battery Health Manager",
    page_icon="🔋",
    layout="wide",
)

# ==========================================
# MODEL PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_soh_model.keras")
DQN_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_online_dqn.keras")

SCALER_X_PATH = os.path.join(MODELS_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(MODELS_DIR, "scaler_y.pkl")

# ==========================================
# LOAD AI SYSTEM (CACHED)
# ==========================================
@st.cache_resource
def load_ai_system():

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
    dqn_model = load_model(DQN_MODEL_PATH, compile=False)

    # Warmup models
    dummy_lstm = np.zeros((1, 20, 6))
    dummy_dqn = np.zeros((1, 4))

    lstm_model.predict(dummy_lstm, verbose=0)
    dqn_model.predict(dummy_dqn, verbose=0)

    return lstm_model, dqn_model, scaler_X, scaler_y


lstm_model, dqn_model, scaler_X, scaler_y = load_ai_system()

# ==========================================
# SIDEBAR INPUTS
# ==========================================
with st.sidebar:

    st.header("⚙️ Battery Telemetry")

    soc = st.slider("State of Charge (%)", 0.0, 100.0, 80.0)

    voltage = st.number_input(
        "Battery Voltage (V)", 300.0, 450.0, 380.0
    )

    current = st.number_input(
        "Charging Current (A)", 0.0, 20.0, 1.5
    )

    temp = st.slider(
        "Battery Temperature (°C)", -10.0, 60.0, 25.0
    )

    cycle = st.number_input(
        "Cycle Count", 0, 5000, 100
    )

    # simulated internal resistance
    ir_sim = 0.02 + cycle * 0.00008

    with st.expander("Advanced Parameters"):

        ir = st.number_input(
            "Internal Resistance (Ω)",
            0.01,
            1.0,
            ir_sim,
        )

        t_max = st.number_input(
            "Max Temperature",
            -10.0,
            80.0,
            temp + 2
        )

        t_min = st.number_input(
            "Min Temperature",
            -10.0,
            80.0,
            temp - 2
        )

        charge_time = st.number_input(
            "Charge Time (h)",
            0.0,
            24.0,
            1.5
        )

# ==========================================
# PREDICTION PIPELINE
# ==========================================
def run_prediction(ir, tavg, tmin, tmax, ctime, cyclenum, curr):

    start_time = time.time()

    # ---------------------------
    # LSTM INPUT FEATURES
    # ---------------------------
    features = np.array(
        [[ir, tavg, tmin, tmax, ctime, cyclenum]]
    )

    scaled = scaler_X.transform(features)

    lstm_input = np.tile(scaled, (20, 1)).reshape(1, 20, 6)

    encoded_soh = lstm_model.predict(
        lstm_input,
        verbose=0
    )[0][0]

    soh = scaler_y.inverse_transform(
        [[encoded_soh]]
    )[0][0]

    # ---------------------------
    # PHYSICS CORRECTION
    # ---------------------------
    deg_cycle = cyclenum * 0.00010
    deg_temp = max(0, tavg - 25) * 0.003
    deg_ir = (ir - 0.02) * 5.0

    physics_soh = 1.0 - (deg_cycle + deg_temp + deg_ir * 0.05)

    alpha = 0.7
    beta = 0.3

    predicted_soh = alpha * soh + beta * physics_soh
    predicted_soh = np.clip(predicted_soh, 0.4, 1.0)

    # ---------------------------
    # DQN STATE
    # ---------------------------
    dqn_state = np.array(
        [[predicted_soh, tavg, cyclenum, curr]]
    )

    q_values = dqn_model.predict(
        dqn_state,
        verbose=0
    )[0]

    action = np.argmax(q_values)

    latency = (time.time() - start_time) * 1000

    return predicted_soh, action, latency, q_values


# ==========================================
# MAIN UI
# ==========================================
st.title("🔋 EV Battery Health Manager")

st.markdown(
"Hybrid **LSTM + Reinforcement Learning (DQN)** Battery Management System"
)

if st.button("⚡ Run Prediction"):

    soh, action, latency, q_vals = run_prediction(
        ir,
        temp,
        t_min,
        t_max,
        charge_time,
        cycle,
        current
    )

    col1, col2, col3 = st.columns(3)

    # -----------------------------------
    # SOH RESULT
    # -----------------------------------
    with col1:

        st.metric(
            "Predicted Battery SoH",
            f"{soh*100:.2f}%"
        )

        if soh > 0.9:
            st.success("Battery Healthy")

        elif soh > 0.8:
            st.warning("Battery Degrading")

        else:
            st.error("Battery Critical")

    # -----------------------------------
    # ACTION
    # -----------------------------------
    actions = {
        0: ("Decrease Charging", "Reduce current"),
        1: ("Maintain Charging", "Optimal charging"),
        2: ("Increase Charging", "Faster charging allowed")
    }

    label, desc = actions[action]

    with col2:

        st.metric("Recommended Action", label)

        st.info(desc)

    # -----------------------------------
    # LATENCY
    # -----------------------------------
    with col3:

        st.metric(
            "Inference Latency",
            f"{latency:.2f} ms"
        )

    # -----------------------------------
    # Q VALUE CHART
    # -----------------------------------
    st.divider()

    st.subheader("DQN Agent Decision Confidence")

    q_df = pd.DataFrame(
        {
            "Action": [
                "Decrease",
                "Maintain",
                "Increase"
            ],
            "Q Value": q_vals
        }
    )

    st.bar_chart(
        q_df.set_index("Action")
    )

else:

    st.info(
        "Adjust battery telemetry in the sidebar and click **Run Prediction**."
    )
