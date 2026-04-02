# import streamlit as st
# import pandas as pd
# import numpy as np

# # TITLE
# st.title("📡 AI Network Congestion Prediction & Optimization System")

# st.markdown("Predicts network congestion and recommends optimization actions in real-time.")

# # SIDEBAR INPUTS
# st.sidebar.header("📊 Input Network Parameters")

# hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
# weekday = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)

# throughput = st.sidebar.number_input("Throughput", value=10.0)
# traffic_norm = st.sidebar.slider("Traffic Normalization (0–1)", 0.0, 1.0, 0.5)
# traffic_diff = st.sidebar.number_input("Traffic Change", value=0.1)
# burst_intensity = st.sidebar.slider("Burst Intensity", 0.0, 1.0, 0.3)
# latency_ms = st.sidebar.slider("Latency (ms)", 0, 250, 60)
# packet_loss_pct = st.sidebar.slider("Packet Loss (%)", 0.0, 1.0, 0.05)

# # CREATE INPUT DATAFRAME
# input_data = pd.DataFrame([{
#     'hour': hour,
#     'weekday': weekday,
#     'throughput': throughput,
#     'traffic_norm': traffic_norm,
#     'traffic_diff': traffic_diff,
#     'burst_intensity': burst_intensity,
#     'latency_ms': latency_ms,
#     'packet_loss_pct': packet_loss_pct
# }])

# st.subheader("📥 Input Data")
# st.write(input_data)

# # LOAD MODELS (Assumes you've saved them)
# import joblib

# xgb_model = joblib.load("xgb_model.pkl")
# rf_optimizer = joblib.load("rf_optimizer.pkl")

# # PREDICT BUTTON
# if st.button("🚀 Predict Network Status"):

#     # Prediction
#     prob = xgb_model.predict_proba(input_data)[0][1]

#     st.subheader("📈 Prediction Result")

#     st.metric("Congestion Probability", f"{prob:.2f}")

#     # DECISION LOGIC
#     if prob < 0.5:
#         level = "🟢 SAFE"
#         action = "No Action Required"

#     elif prob < 0.7:
#         level = "🟡 WARNING"
#         action = "Monitor Network"

#     else:
#         dt_features = ['traffic_norm', 'burst_intensity', 'latency_ms', 'traffic_diff']
#         action_pred = rf_optimizer.predict(input_data[dt_features])[0]

#         level = "🔴 CRITICAL"
#         action = action_pred

#     # DISPLAY
#     st.subheader("🧠 Decision Engine Output")

#     st.write(f"**Risk Level:** {level}")
#     st.write(f"**Recommended Action:** {action}")

#     # VISUAL FEEDBACK
#     st.progress(int(prob * 100))


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration for a wider, sleeker layout
st.set_page_config(page_title="Telecom AI Controller", layout="wide")

# ==========================================
# 1. Load the Trained Models (Cached for speed)
# ==========================================
@st.cache_resource
def load_models():
    try:
        xgb = joblib.load("xgb_model.pkl")
        rf = joblib.load("rf_optimizer.pkl")
        return xgb, rf
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'predictive_xgb_model.joblib' and 'prescriptive_rf_model.joblib' are in the same folder.")
        st.stop()

xgb_model, rf_optimizer = load_models()

# ==========================================
# 2. UI Layout: Header
# ==========================================
st.title("📡 Autonomous Network AI Controller")
st.markdown("""
This dashboard simulates real-time network telemetry. 
Adjust the sliders on the left to simulate different network conditions. The **Predictive AI** will forecast congestion, and the **Prescriptive AI** will deploy an autonomous fix if necessary.
""")
st.divider()

# ==========================================
# 3. UI Layout: Sidebar (Telemetry Inputs)
# ==========================================
st.sidebar.header("⚙️ Network Telemetry Inputs")
st.sidebar.markdown("Simulate current hardware state:")

# Temporal Features
hour = st.sidebar.slider("Hour of Day", 0, 23, 18)
weekday = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 4)

# Workload Features
throughput = st.sidebar.slider("Current Throughput", 0.0, 20.0, 8.5)
expected_throughput = st.sidebar.slider("Expected Throughput (Baseline)", 0.0, 20.0, 7.0)
traffic_norm = st.sidebar.slider("Capacity Utilization (0-1)", 0.0, 1.0, 0.85)
traffic_diff = st.sidebar.slider("Traffic Burst Rate (diff)", -5.0, 15.0, 2.5)
burst_intensity = st.sidebar.slider("Burst Intensity (0-1)", 0.0, 1.0, 0.6)

# Physical QoS Features
latency_ms = st.sidebar.slider("Latency (ms)", 20.0, 500.0, 65.0)
packet_loss_pct = st.sidebar.slider("Packet Loss (%)", 0.0, 20.0, 0.0)

# Build the input dataframe exactly how the models expect it
input_features = pd.DataFrame({
    'hour': [hour],
    'weekday': [weekday],
    'throughput': [throughput],
    'traffic_norm': [traffic_norm],
    'traffic_diff': [traffic_diff],
    'burst_intensity': [burst_intensity],
    'expected_throughput': [expected_throughput],
    'latency_ms': [latency_ms],
    'packet_loss_pct': [packet_loss_pct]
})

# ==========================================
# 4. Main Dashboard Area
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric(label="Current Utilization", value=f"{traffic_norm*100:.1f}%", delta=f"{traffic_diff:.2f} Burst")
col2.metric(label="Current Latency", value=f"{latency_ms:.0f} ms")
col3.metric(label="Packet Loss", value=f"{packet_loss_pct:.1f}%")

st.markdown("### AI Controller Status")

# Run the AI logic when the user clicks the button
if st.button("🚀 Analyze Network State & Deploy AI", use_container_width=True):
    with st.spinner('AI is analyzing the next 10 minutes...'):
        
        # Step 1: Predictive Layer (XGBoost)
        prob = xgb_model.predict_proba(input_features)[0][1]
        
        st.divider()
        
        # Step 2: The Tiered Logic Controller
        if prob < 0.5:
            st.success(f"🟢 **SAFE** | Congestion Probability: **{prob*100:.1f}%**")
            st.info("Network is stable. No autonomous action required.")
            
        elif prob < 0.7:
            st.warning(f"🟡 **WARNING** | Congestion Probability: **{prob*100:.1f}%**")
            st.info("Pressure is rising, but within acceptable limits. Monitoring closely.")
            
        else:
            st.error(f"🔴 **CRITICAL** | Congestion Probability: **{prob*100:.1f}%**")
            st.error("⚠️ Hardware failure predicted in < 10 minutes. Handing over to Prescriptive Layer...")
            
            # Step 3: Prescriptive Layer (Random Forest)
            dt_features = input_features[['traffic_norm', 'burst_intensity', 'latency_ms', 'traffic_diff']]
            action = rf_optimizer.predict(dt_features)[0]
            
            st.markdown("### 🛠️ Autonomous Mitigation Deployed")
            if action == 'Throttle_Call_Rate':
                st.error(f"**ACTION TAKEN: {action}** (Severing low-priority connections to prevent buffer overflow)")
            elif action == 'Reallocate_Bandwidth':
                st.warning(f"**ACTION TAKEN: {action}** (Shifting resources from healthy cells to node)")
            elif action == 'Optimize_Routing':
                st.info(f"**ACTION TAKEN: {action}** (Rerouting new handshakes to adjacent towers)")
            else:
                st.success(f"**ACTION TAKEN: {action}** (Restricting background app data to ease congestion)")