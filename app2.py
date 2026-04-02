import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="AI Network Controller", page_icon="📡", layout="wide")

# Custom CSS for beautification
st.markdown("""
    <style>
    .stProgress .st-bo { background-color: #ff4b4b; }
    .big-font { font-size:24px !important; font-weight:bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOADING & STATE INITIALIZATION
# ==========================================
@st.cache_resource
def load_models():
    xgb = joblib.load("xgb_model.pkl")
    rf = joblib.load("rf_optimizer.pkl")
    return xgb, rf

try:
    xgb_model, rf_optimizer = load_models()
except:
    st.error("🚨 Models not found! Ensure .joblib files are in the same folder.")
    st.stop()

# Initialize session state for the live simulation history
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['time', 'throughput', 'latency', 'prob', 'status'])
if 'sim_time' not in st.session_state:
    st.session_state.sim_time = 0

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Cell_tower.jpg/1200px-Cell_tower.jpg", use_container_width=True)
st.sidebar.title("System Navigation")
page = st.sidebar.radio("Select Module:", ["🏠 Landing / System Overview", "⚡ Live Autonomous Control", "🎛️ Scenario Lab"])

# ==========================================
# PAGE 1: LANDING PAGE
# ==========================================
if page == "🏠 Landing / System Overview":
    st.title("📡 Autonomous Telecom Network Controller")
    st.markdown("### Final Year Project Deployment Simulation")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Welcome to the live deployment dashboard.**
        This system utilizes a dual-engine Artificial Intelligence architecture to monitor, predict, and prescribe solutions for communication network hardware.
        
        * **🧠 Predictive Brain (XGBoost):** Scans 10 minutes into the future to forecast hardware queue saturation and buffer overflows.
        * **🛠️ Prescriptive Brain (Random Forest):** Acts as a surrogate deterministic engine to autonomously deploy network routing commands.
        """)
        st.info("👈 Please select a module from the sidebar to begin the simulation.")
        
    with col2:
        st.success("✅ XGBoost Predictive Model: Loaded")
        st.success("✅ Random Forest Optimization: Loaded")
        st.success("✅ SDN Controller API: Connected")

# ==========================================
# PAGE 2: LIVE AUTONOMOUS CONTROL (REAL-TIME)
# ==========================================
elif page == "⚡ Live Autonomous Control":
    st.title("⚡ Live Network Telemetry & Autonomous AI")
    st.markdown("This dashboard simulates real-time network traffic. Click the button below to start the live feed.")
    
    run_sim = st.button("▶️ Start Live Network Feed (Generate Next 10 Mins)")
    
    # Placeholders for dynamic UI updates
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    chart_placeholder = st.empty()
    action_placeholder = st.empty()
    
    if run_sim:
        with st.spinner("Syncing with hardware endpoints..."):
            time.sleep(1) # Fake loading page delay
            
            # Generate slightly randomized but realistic progression data
            st.session_state.sim_time += 10
            base_load = 6.0 + np.sin(st.session_state.sim_time / 50) * 4 # Creates a wave pattern
            spike = np.random.choice([0, 0, 0, 8.0]) # Random massive spikes
            
            current_tp = base_load + spike + np.random.uniform(0, 1)
            current_lat = 40 + (current_tp**2.5)*0.2 if current_tp > 8 else 40 + np.random.uniform(-5, 5)
            
            # Build input for XGBoost
            sim_features = pd.DataFrame({
                'hour': [18], 'weekday': [4], 'throughput': [current_tp],
                'traffic_norm': [current_tp/15.0], 'traffic_diff': [spike],
                'burst_intensity': [spike/15.0], 'expected_throughput': [base_load],
                'latency_ms': [current_lat], 'packet_loss_pct': [0.0 if current_lat < 150 else 5.0]
            })
            
            # 1. PREDICT
            prob = xgb_model.predict_proba(sim_features)[0][1]
            status = "SAFE" if prob < 0.5 else "WARNING" if prob < 0.7 else "CRITICAL"
            
            # Append to history
            new_data = pd.DataFrame([{'time': f"T+{st.session_state.sim_time}m", 'throughput': current_tp, 'latency': current_lat, 'prob': prob, 'status': status}])
            st.session_state.history = pd.concat([st.session_state.history, new_data], ignore_index=True).tail(20) # Keep last 20
            
            # 2. UPDATE METRICS
            metric_col1.metric("Throughput (Normalized)", f"{current_tp:.2f}", f"{spike:.1f} Burst")
            metric_col2.metric("Simulated Latency", f"{current_lat:.0f} ms")
            metric_col3.metric("AI Congestion Forecast", f"{prob*100:.1f}%")
            
            # 3. DRAW PLOTLY CHART
            fig = px.line(st.session_state.history, x='time', y='throughput', title='Real-Time Network Load (Last 200 mins)', markers=True)
            fig.add_scatter(x=st.session_state.history['time'], y=st.session_state.history['latency']/10, name="Latency (Scaled)", mode="lines")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # 4. DEPLOY AUTONOMOUS ACTION
            if status == "CRITICAL":
                action_placeholder.error(f"🚨 CRITICAL THRESHOLD BREACHED (Prob: {prob*100:.1f}%)")
                with st.spinner("AI Controller finding optimal solution..."):
                    time.sleep(1.5) # Simulate AI thinking
                    dt_features = sim_features[['traffic_norm', 'burst_intensity', 'latency_ms', 'traffic_diff']]
                    action = rf_optimizer.predict(dt_features)[0]
                    action_placeholder.success(f"🛠️ **AUTONOMOUS ACTION EXECUTED:** Initiating `{action}` API Protocol.")
            elif status == "WARNING":
                action_placeholder.warning("🟡 Elevated stress detected. System is actively monitoring.")
            else:
                action_placeholder.success("🟢 Network is stable. No action required.")

# ==========================================
# PAGE 3: SCENARIO LAB (MANUAL CONFIG)
# ==========================================
elif page == "🎛️ Scenario Lab":
    st.title("🎛️ Manual Scenario Configuration")
    st.markdown("Manually inject hardware constraints to test the AI's edge-case decision logic.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Inject Parameters")
        tp = st.slider("Throughput", 0.0, 15.0, 8.5)
        diff = st.slider("Traffic Burst Rate", -2.0, 10.0, 4.0)
        lat = st.slider("Latency (ms)", 20.0, 300.0, 160.0)
        norm = st.slider("Capacity Utilization", 0.0, 1.0, 0.85)
        
        test_features = pd.DataFrame({
            'hour': [18], 'weekday': [4], 'throughput': [tp], 'traffic_norm': [norm],
            'traffic_diff': [diff], 'burst_intensity': [diff/15.0], 'expected_throughput': [7.0],
            'latency_ms': [lat], 'packet_loss_pct': [0.0 if lat < 150 else 2.5]
        })
        
    with col2:
        st.subheader("AI Analysis")
        prob = xgb_model.predict_proba(test_features)[0][1]
        
        # BEAUTIFUL GAUGE CHART
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Failure Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
            }))
        st.plotly_chart(fig, use_container_width=True)
        
        if prob >= 0.7:
            dt_features = test_features[['traffic_norm', 'burst_intensity', 'latency_ms', 'traffic_diff']]
            action = rf_optimizer.predict(dt_features)[0]
            st.error(f"**PRESCRIBED ACTION:** {action}")
        else:
            st.success("**PRESCRIBED ACTION:** Monitor Only")