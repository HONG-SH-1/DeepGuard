import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from tensorflow.keras.models import load_model

@st.cache_resource
def load_resources():
    model = load_model('../models/deepguard_model.keras')
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../models/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return model, scaler, threshold

model, scaler, threshold = load_resources()

st.title("🛡️ DeepGuard - 실시간 네트워크 침입 탐지")
st.markdown(f"**임계값 (Threshold):** `{threshold:.6f}`")

if 'mse_history' not in st.session_state:
    st.session_state.mse_history = []
if 'running' not in st.session_state:
    st.session_state.running = False

def generate_traffic(anomaly=False):
    if anomaly:
        return np.random.uniform(0.8, 1.0, (1, 78))
    return np.random.uniform(0.0, 0.3, (1, 78))

def predict(raw_data):
    scaled = scaler.transform(raw_data)
    sequence = np.tile(scaled, (10, 1)).reshape(1, 10, 78)
    pred = model.predict(sequence, verbose=0)
    mse = np.mean(np.square(sequence - pred))
    return mse, mse > threshold

col1, col2, col3 = st.columns(3)
normal_btn = col1.button("▶ 정상 모니터링")
attack_btn = col2.button("🚨 공격 주입 (10회)")
stop_btn = col3.button("⏹ 정지")

chart_placeholder = st.empty()
status_placeholder = st.empty()

if stop_btn:
    st.session_state.running = False
    st.session_state.mse_history = []

if normal_btn or attack_btn:
    st.session_state.running = True
    for i in range(10):
        if not st.session_state.running:
            break
        
        # 공격 주입은 i=3~6 구간만
        is_attack_phase = attack_btn and (3 <= i <= 6)
        raw = generate_traffic(anomaly=is_attack_phase)
        mse, is_anomaly = predict(raw)
        st.session_state.mse_history.append(mse)

        with chart_placeholder.container():
            st.line_chart(pd.DataFrame({'MSE': st.session_state.mse_history}))
            if is_anomaly:
                status_placeholder.error(f"🚨 ANOMALY 탐지! MSE: {mse:.6f}")
            else:
                status_placeholder.success(f"✅ 정상 트래픽 MSE: {mse:.6f}")
        time.sleep(1)
    
    st.session_state.running = False