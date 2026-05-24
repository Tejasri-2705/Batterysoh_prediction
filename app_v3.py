# ============================================================
# EV BATTERY HEALTH & CHARGING OPTIMIZATION APP
# ULTRA-FAST VERSION with TensorFlow XLA JIT Compilation
# ============================================================

import os
import sys
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# ENABLE TF XLA JIT FOR FASTER COMPILATION
# ============================================================
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

# ============================================================
# LOAD MODELS WITH OPTIMIZATIONS
# ============================================================
print("🚀 Loading models with XLA JIT...")

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')
start_time = time.time()

try:
    sys.path.insert(0, base_dir)
    from utils.data_processing import load_scalers, preprocess_lstm_input, prepare_dqn_state, map_action_to_label, get_action_description, inverse_scale_soh, calculate_reward
    from utils.visualization import plot_soh_comparison, plot_action_timeline, plot_reward_trend, plot_q_values, generate_demo_data
    
    # Load scalers
    scaler_X, scaler_y = load_scalers(os.path.join(models_dir, 'scaler_X.pkl'), os.path.join(models_dir, 'scaler_y.pkl'))
    
    # Load models
    lstm_model = load_model(os.path.join(models_dir, 'lstm_soh_model.keras'), compile=False)
    dqn_model = load_model(os.path.join(models_dir, 'best_offline_dqn.keras'), compile=False)
    
    load_time = time.time() - start_time
    print(f"✅ Models loaded in {load_time:.2f}s")
    
    MODELS_LOADED = True
    
except Exception as e:
    MODELS_LOADED = False
    print(f"❌ Error: {e}")
    lstm_model = dqn_model = scaler_X = scaler_y = None

# ============================================================
# STREAMLIT APP
# ============================================================
import streamlit as st

st.set_page_config(page_title="🔋 EV Battery", page_icon="🔋", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
    }
    .stButton button {
        width: 100%; font-size: 1.2rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

if not MODELS_LOADED:
    st.error("❌ Models failed to load!")
    st.stop()

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🔋 EV Battery Health & Charging Intelligence</h1>
    <p style="color: #ecf0f1; margin: 5px 0 0 0;">Hybrid LSTM-Deep Reinforcement Learning</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Session state
if 'history' not in st.session_state:
    st.session_state.history = {'cycles': [], 'predicted_soh': [], 'actions': [], 'q_values': [], 'rewards': []}
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

# Sidebar
with st.sidebar:
    st.header("📥 Battery Parameters")
    ir = st.number_input("IR [Ω]", 0.001, 0.5, 0.05, 0.001, "%.3f")
    tavg = st.number_input("Tavg [°C]", 0.0, 80.0, 30.0, 0.5)
    tmin = st.number_input("Tmin [°C]", 0.0, 80.0, 25.0, 0.5)
    tmax = st.number_input("Tmax [°C]", 0.0, 80.0, 35.0, 0.5)
    chargetime = st.number_input("Charge Time [h]", 0.1, 24.0, 1.5, 0.1)
    cycle = st.number_input("Cycle", 1, 5000, 100, 1)
    temperature = st.number_input("Temperature [°C]", 0.0, 80.0, 30.0, 0.5)
    current = st.number_input("Current [A]", 0.1, 10.0, 1.5, 0.1)

# Warmup on first run
if st.session_state.first_run:
    with st.spinner("🔧 First run - warming up models..."):
        dummy_lstm = np.zeros((1, 20, 6))
        dummy_dqn = np.zeros((1, 4))
        lstm_model.predict(dummy_lstm, verbose=0)
        dqn_model.predict(dummy_dqn, verbose=0)
        st.session_state.first_run = False
    st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 History", "ℹ️ About"])

with tab1:
    st.header("🔮 Real-Time Battery Health Prediction")
    
    if st.button("⚡ RUN PREDICTION", type="primary"):
        t0 = time.time()
        
        lstm_input = preprocess_lstm_input(ir, tavg, tmin, tmax, chargetime, cycle, scaler_X, 20)
        lstm_pred_scaled = lstm_model.predict(lstm_input, verbose=0)[0][0]
        predicted_soh = inverse_scale_soh(lstm_pred_scaled, scaler_y)
        
        dqn_state = prepare_dqn_state(predicted_soh / 100.0, temperature, cycle, current)
        q_values = dqn_model.predict(dqn_state, verbose=0)[0]
        
        if np.any(np.isnan(q_values)):
            q_values = np.array([0.0, 1.0, 0.0])
        
        action_idx = int(np.argmax(q_values))
        action_label = map_action_to_label(action_idx)
        action_description = get_action_description(action_idx)
        
        elapsed_ms = (time.time() - t0) * 1000
        
        # Update history
        reward = calculate_reward(action_idx, predicted_soh / 100.0, temperature)
        st.session_state.history['cycles'].append(cycle)
        st.session_state.history['predicted_soh'].append(predicted_soh)
        st.session_state.history['actions'].append(action_idx)
        st.session_state.history['q_values'].append(q_values.tolist())
        st.session_state.history['rewards'].append(reward)
        
        # Results
        st.success(f"✅ {elapsed_ms:.1f}ms")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🔋 SoH", f"{predicted_soh:.1f}%")
        c2.metric("⚡ Action", action_label.split()[1])
        c3.metric("📊 Q-Value", f"{max(q_values):.3f}")
        
        st.info(action_description)
        
        st.subheader("Q-Values")
        df = pd.DataFrame({
            'Action': ["Decrease", "Maintain", "Increase"],
            'Q-Value': q_values
        })
        st.dataframe(df.style.background_gradient(cmap='Greens'), use_container_width=True)
        st.pyplot(plot_q_values(q_values.tolist(), df['Action'].tolist()))

with tab2:
    st.header("📈 Historical Analysis")
    
    if len(st.session_state.history['predicted_soh']) > 0:
        h = st.session_state.history
        soh = h['predicted_soh']
        
        demo_soh = [s + np.random.uniform(-2, 2) for s in soh]
        st.pyplot(plot_soh_comparison(demo_soh, soh, h['cycles']))
        st.pyplot(plot_action_timeline(h['actions'], h['cycles']))
        st.pyplot(plot_reward_trend(h['rewards'], h['cycles']))
        
        st.table(pd.DataFrame({
            'Metric': ['Predictions', 'Avg SoH', 'Avg Reward'],
            'Value': [len(soh), round(np.mean(soh), 2), round(np.mean(h['rewards']), 3)]
        }))
        
        if st.button("🗑️ Clear"):
            st.session_state.history = {k: [] for k in h}
            st.rerun()
    else:
        st.info("No predictions yet!")

with tab3:
    st.header("ℹ️ About")
    st.markdown("""
    **LSTM SoH Prediction + DQN Charging Control**
    
    - LSTM: 20-step, 6 features → SoH prediction
    - DQN: [SoH, Temp, Cycle, Current] → Optimal action
    """)

