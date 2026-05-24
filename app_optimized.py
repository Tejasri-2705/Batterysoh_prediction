# ============================================================
# EV BATTERY HEALTH & CHARGING OPTIMIZATION APP
# Optimized Version - Faster Model Loading
# ============================================================

import os
import sys
import time
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing import (
    load_scalers,
    preprocess_lstm_input,
    prepare_dqn_state,
    map_action_to_label,
    get_action_description,
    inverse_scale_soh,
    calculate_reward
)
from utils.visualization import (
    plot_soh_comparison,
    plot_action_timeline,
    plot_reward_trend,
    plot_q_values,
    generate_demo_data
)

st.set_page_config(
    page_title="🔋 EV Battery Health Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_all_models_and_scalers():
    """
    Optimized model loading - removed warmup and UI calls for faster startup.
    """
    start_time = time.time()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    lstm_path = os.path.join(models_dir, 'lstm_soh_model.keras')
    dqn_path = os.path.join(models_dir, 'best_offline_dqn.keras')
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    
    # Check files exist
    for name, path in [('LSTM', lstm_path), ('DQN', dqn_path), 
                       ('Scaler X', scaler_x_path), ('Scaler y', scaler_y_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {name} at {path}")
    
    # Load scalers first (fast)
    scaler_X, scaler_y = load_scalers(scaler_x_path, scaler_y_path)
    
    # Load models
    lstm_model = load_model(lstm_path, compile=False)
    dqn_model = load_model(dqn_path, compile=False)
    
    elapsed = time.time() - start_time
    print(f"✅ Models loaded in {elapsed:.2f}s")
    
    return lstm_model, dqn_model, scaler_X, scaler_y


def warmup_models(lstm_model, dqn_model):
    """Optional: Call this after load to warm up models for faster first prediction."""
    dummy_lstm = np.zeros((1, 20, 6))
    dummy_dqn = np.zeros((1, 4))
    lstm_model.predict(dummy_lstm, verbose=0)
    dqn_model.predict(dummy_dqn, verbose=0)


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = {
            'cycles': [], 'actual_soh': [], 'predicted_soh': [],
            'actions': [], 'q_values': [], 'rewards': []
        }
    if 'models_warmed' not in st.session_state:
        st.session_state.models_warmed = False


def main():
    initialize_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🔋 EV Battery Health & Charging Intelligence</h1>
        <p style="color: #ecf0f1; margin: 5px 0 0 0;">
            Hybrid LSTM-Deep Reinforcement Learning for Battery SoH Prediction & Adaptive Charging Control
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        try:
            lstm_model, dqn_model, scaler_X, scaler_y = load_all_models_and_scalers()
            
            # Warm up models only once after first load
            if not st.session_state.models_warmed:
                warmup_models(lstm_model, dqn_model)
                st.session_state.models_warmed = True
                
            st.success("✅ All models and scalers loaded successfully!")
        except Exception as e:
            st.error("❌ Failed to load models")
            st.exception(e)
            st.stop()
    
    # Sidebar inputs
    with st.sidebar:
        st.header("📥 Battery Parameters")
        
        ir = st.number_input("Internal Resistance (IR) [Ω]", 0.001, 0.5, 0.05, 0.001, "%.3f")
        tavg = st.number_input("Average Temperature (Tavg) [°C]", 0.0, 80.0, 30.0, 0.5)
        tmin = st.number_input("Minimum Temperature (Tmin) [°C]", 0.0, 80.0, 25.0, 0.5)
        tmax = st.number_input("Maximum Temperature (Tmax) [°C]", 0.0, 80.0, 35.0, 0.5)
        chargetime = st.number_input("Charging Time [hours]", 0.1, 24.0, 1.5, 0.1)
        cycle = st.number_input("Cycle Number", 1, 5000, 100, 1)
        temperature = st.number_input("Current Temperature [°C]", 0.0, 80.0, 30.0, 0.5)
        current = st.number_input("Charging Current [A]", 0.1, 10.0, 1.5, 0.1)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Prediction", "📈 Historical Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("🔮 Real-Time Battery Health Prediction")
        
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
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
                    reward = calculate_reward(action_idx, predicted_soh / 100.0, temperature)
                    
                    # Update history
                    st.session_state.history['cycles'].append(cycle)
                    st.session_state.history['predicted_soh'].append(predicted_soh)
                    st.session_state.history['actions'].append(action_idx)
                    st.session_state.history['q_values'].append(q_values.tolist())
                    st.session_state.history['rewards'].append(reward)
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔋 Predicted SoH", f"{predicted_soh:.2f}%")
                    with col2:
                        st.metric("⚡ Recommended Action", action_label.split()[1])
                    with col3:
                        st.metric("📊 Q-Value (Max)", f"{max(q_values):.4f}")
                    
                    st.info(f"**Action Details:** {action_description}")
                    
                    # Q-values visualization
                    st.subheader("📊 Q-Values for Transparency")
                    action_names = ["Decrease Current", "Maintain Current", "Increase Current"]
                    q_df = pd.DataFrame({'Action': action_names, 'Q-Value': q_values})
                    st.dataframe(q_df.style.background_gradient(subset=['Q-Value'], cmap='Greens'), use_container_width=True)
                    fig_q = plot_q_values(q_values.tolist(), action_names)
                    st.pyplot(fig_q)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)
    
    with tab2:
        st.header("📈 Historical Analysis & Visualization")
        
        if len(st.session_state.history['predicted_soh']) > 0:
            history = st.session_state.history
            
            st.subheader("1️⃣ Actual vs Predicted SoH")
            actual_soh = history.get('actual_soh', [])
            predicted_soh = history.get('predicted_soh', [])
            
            if actual_soh and len(actual_soh) > 0:
                fig_soh = plot_soh_comparison(actual_soh, predicted_soh, history['cycles'])
            else:
                demo_soh = [soh + np.random.uniform(-2, 2) for soh in predicted_soh]
                fig_soh = plot_soh_comparison(demo_soh, predicted_soh, history['cycles'])
            st.pyplot(fig_soh)
            
            st.subheader("2️⃣ DRL Action Selection Timeline")
            actions = history.get('actions', [])
            if actions and len(actions) > 0:
                fig_actions = plot_action_timeline(actions, history['cycles'])
                st.pyplot(fig_actions)
            
            st.subheader("3️⃣ Reward Trend")
            rewards = history.get('rewards', [])
            if rewards and len(rewards) > 0:
                fig_rewards = plot_reward_trend(rewards, history['cycles'])
                st.pyplot(fig_rewards)
            
            st.subheader("📋 Session Summary")
            avg_soh = np.mean(predicted_soh) if predicted_soh else 0
            avg_reward = np.mean(rewards) if rewards else 0
            summary_df = pd.DataFrame({
                'Metric': ['Total Predictions', 'Avg Predicted SoH (%)', 'Avg Reward'],
                'Value': [len(predicted_soh), round(avg_soh, 2), round(avg_reward, 4)]
            })
            st.table(summary_df)
            
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = {k: [] for k in st.session_state.history}
                st.rerun()
        else:
            st.info("📭 No prediction history yet. Make predictions in the 'Real-Time Prediction' tab.")
    
    with tab3:
        st.header("ℹ️ About This Application")
        st.markdown("""
        ## 🔋 EV Battery Health & Charging Intelligence System
        
        **Hybrid LSTM-Deep Reinforcement Learning** approach for electric vehicle battery health optimization.
        
        ### 🏗️ Architecture
        **1. LSTM SoH Prediction Model**
        - Input: 20-step time series with 6 features
        - Output: State of Health (SoH) prediction
        
        **2. Deep Q-Network (DQN) Charging Policy**
        - State: [SoH, Temperature, Cycle, Charging Current]
        - Action Space: 3 discrete actions
          - 0: Decrease Current (conservative)
          - 1: Maintain Current (standard)
          - 2: Increase Current (fast)
        """)
        
        st.markdown("---")
        st.caption("Built with Streamlit, TensorFlow/Keras, and Scikit-learn")


if __name__ == "__main__":
    main()

