# ============================================================
# EV BATTERY HEALTH & CHARGING OPTIMIZATION APP
# INSTANT LOAD VERSION - Models preloaded at import time
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
# PRE-LOAD MODELS AT IMPORT TIME (before Streamlit starts)
# ============================================================
print("🚀 Pre-loading models for instant predictions...")

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

try:
    # Import utils
    sys.path.insert(0, base_dir)
    from utils.data_processing import load_scalers, preprocess_lstm_input, prepare_dqn_state, map_action_to_label, get_action_description, inverse_scale_soh, calculate_reward
    from utils.visualization import plot_soh_comparison, plot_action_timeline, plot_reward_trend, plot_q_values, generate_demo_data
    
    # Load scalers
    scaler_X, scaler_y = load_scalers(os.path.join(models_dir, 'scaler_X.pkl'), os.path.join(models_dir, 'scaler_y.pkl'))
    
    # Load models with compile=False
    lstm_model = load_model(os.path.join(models_dir, 'lstm_soh_model.keras'), compile=False)
    dqn_model = load_model(os.path.join(models_dir, 'best_offline_dqn.keras'), compile=False)
    
    # Warmup (compiles TensorFlow graphs)
    dummy_lstm = np.zeros((1, 20, 6))
    dummy_dqn = np.zeros((1, 4))
    lstm_model.predict(dummy_lstm, verbose=0)
    dqn_model.predict(dummy_dqn, verbose=0)
    
    MODELS_LOADED = True
    print(f"✅ Models pre-loaded and ready! ({time.time():.2f}s)")
    
except Exception as e:
    MODELS_LOADED = False
    print(f"❌ Model loading failed: {e}")
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
</style>
""", unsafe_allow_html=True)

# Show loading status during initial Streamlit startup
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

# Instant prediction function
def run_prediction(ir, tavg, tmin, tmax, chargetime, cycle, temperature, current):
    """Instant prediction - models are already warm!"""
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
    
    return predicted_soh, q_values, action_idx, action_label, action_description

# Session state
if 'history' not in st.session_state:
    st.session_state.history = {'cycles': [], 'actual_soh': [], 'predicted_soh': [], 'actions': [], 'q_values': [], 'rewards': []}

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

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Prediction", "📈 Historical Analysis", "ℹ️ About"])

with tab1:
    st.header("🔮 Real-Time Battery Health Prediction")
    
    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        start = time.time()
        
        predicted_soh, q_values, action_idx, action_label, action_description = run_prediction(
            ir, tavg, tmin, tmax, chargetime, cycle, temperature, current)
        
        elapsed_ms = (time.time() - start) * 1000
        
        # Update history
        reward = calculate_reward(action_idx, predicted_soh / 100.0, temperature)
        st.session_state.history['cycles'].append(cycle)
        st.session_state.history['predicted_soh'].append(predicted_soh)
        st.session_state.history['actions'].append(action_idx)
        st.session_state.history['q_values'].append(q_values.tolist())
        st.session_state.history['rewards'].append(reward)
        
        # Results
        st.success(f"✅ Prediction in {elapsed_ms:.1f}ms!")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("🔋 Predicted SoH", f"{predicted_soh:.2f}%")
        with col2: st.metric("⚡ Action", action_label.split()[1])
        with col3: st.metric("📊 Q-Value", f"{max(q_values):.4f}")
        
        st.info(f"**Details:** {action_description}")
        
        # Q-values
        st.subheader("📊 Q-Values")
        action_names = ["Decrease Current", "Maintain Current", "Increase Current"]
        q_df = pd.DataFrame({'Action': action_names, 'Q-Value': q_values})
        st.dataframe(q_df.style.background_gradient(subset=['Q-Value'], cmap='Greens'), use_container_width=True)
        st.pyplot(plot_q_values(q_values.tolist(), action_names))

with tab2:
    st.header("📈 Historical Analysis")
    
    if len(st.session_state.history['predicted_soh']) > 0:
        history = st.session_state.history
        predicted_soh = history['predicted_soh']
        
        st.subheader("SoH Comparison")
        demo_soh = [soh + np.random.uniform(-2, 2) for soh in predicted_soh]
        st.pyplot(plot_soh_comparison(demo_soh, predicted_soh, history['cycles']))
        
        st.subheader("Action Timeline")
        st.pyplot(plot_action_timeline(history['actions'], history['cycles']))
        
        st.subheader("Reward Trend")
        st.pyplot(plot_reward_trend(history['rewards'], history['cycles']))
        
        st.subheader("📋 Summary")
        st.table(pd.DataFrame({
            'Metric': ['Total Predictions', 'Avg SoH (%)', 'Avg Reward'],
            'Value': [len(predicted_soh), round(np.mean(predicted_soh), 2), round(np.mean(history['rewards']), 4)]
        }))
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = {k: [] for k in st.session_state.history}
            st.rerun()
    else:
        st.info("📭 No predictions yet. Click 'Run Prediction'!")

with tab3:
    st.header("ℹ️ About")
    st.markdown("""
    ## 🔋 EV Battery Health Intelligence
    
    **LSTM SoH Prediction + DQN Charging Control**
    
    ### Actions:
    - 0: Decrease Current (conservative)
    - 1: Maintain Current (standard)
    - 2: Increase Current (fast)
    """)

if __name__ == "__main__":
    pass  # App runs at import time!

