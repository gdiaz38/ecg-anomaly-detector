import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3

st.set_page_config(
    page_title="FDA-Grade ECG Anomaly Detector",
    page_icon="🫀",
    layout="wide"
)

API_URL = "http://localhost:8001"

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #ff6b6b; }
    .sub-header  { color: #888; margin-bottom: 1.5rem; }
    .normal-box  { background:#0d2b1f; border:1px solid #00d4aa;
                   border-radius:10px; padding:1rem; text-align:center; }
    .anomaly-box { background:#2b0d0d; border:1px solid #ff4444;
                   border-radius:10px; padding:1rem; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🫀 FDA-Grade ECG Anomaly Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Two-stage detection: Isolation Forest + LSTM Autoencoder | SHAP explainability | 21 CFR Part 11 audit logging</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Patient & Device")
patient_id = st.sidebar.text_input("Patient ID", value="PT-001")
device_id  = st.sidebar.text_input("Device ID",  value="MASIMO-4821")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Stage 1: Isolation Forest (AUC 0.75)")
st.sidebar.markdown("- Stage 2: LSTM Autoencoder")
st.sidebar.markdown("- Explainability: SHAP KernelExplainer")
st.sidebar.markdown("- Compliance: FDA 21 CFR Part 11")

# ── Load real test waveforms ──────────────────────────────────────────────────
@st.cache_data
def load_test_data():
    X = np.load("X_test.npy")
    y = np.load("y_test.npy")
    return X, y

X_test, y_test = load_test_data()

classes = {0:"Normal", 1:"Supraventricular", 2:"Ventricular", 3:"Fusion", 4:"Unknown"}
y_binary = (y_test != 0).astype(int)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Detection", "📊 Batch Analysis", "📋 Audit Trail"])

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Single Beat Analysis")

    col1, col2 = st.columns([1, 2])
    with col1:
        beat_class = st.selectbox(
            "Select ECG beat type",
            options=list(classes.keys()),
            format_func=lambda x: f"Class {x}: {classes[x]}"
        )
        beat_idx = st.slider("Sample index", 0, 49, 0)

    # Get sample
    class_indices = np.where(y_test == beat_class)[0]
    sample_idx    = class_indices[beat_idx]
    waveform      = X_test[sample_idx].tolist()

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=waveform, mode='lines',
            line=dict(color='#00d4aa', width=2),
            name='ECG Waveform'
        ))
        fig.update_layout(
            title=f"ECG Waveform — {classes[beat_class]}",
            xaxis_title="Timestep", yaxis_title="Amplitude",
            template="plotly_dark", height=250,
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🔬 Analyze Beat", type="primary"):
        with st.spinner("Running two-stage detection + SHAP..."):
            payload = {
                "patient_id": patient_id,
                "device_id":  device_id,
                "waveform":   waveform
            }
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                result = r.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # ── Result display ────────────────────────────────────────────────────
        pred  = result["prediction"]
        conf  = result["confidence"]
        color = "#ff4444" if pred == "ANOMALY" else "#00d4aa"
        box   = "anomaly-box" if pred == "ANOMALY" else "normal-box"

        st.markdown(f"""
        <div class="{box}">
            <h2 style="color:{color}; margin:0">{pred}</h2>
            <p style="color:#aaa; margin:4px 0">Confidence: {conf} | Record: {result['record_id'][:8]}...</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Isolation Forest Score", f"{result['iso_score']:.4f}")
        m2.metric("LSTM Reconstruction Error", f"{result['recon_error']:.6f}")
        m3.metric("Ensemble Score", f"{result['ensemble_score']:.4f}")

        # ── SHAP visualization ────────────────────────────────────────────────
        st.markdown("#### 🧠 SHAP Explanation — Why this prediction?")
        st.caption("Which parts of the ECG waveform drove the decision")

        shap_data = result["shap_top5"]
        shap_fig  = make_subplots(specs=[[{"secondary_y": True}]])

        shap_fig.add_trace(go.Bar(
            x=[f"t={s['timestep']}" for s in shap_data],
            y=[s['shap_value'] for s in shap_data],
            marker_color=['#ff4444' if s['shap_value'] > 0 else '#00d4aa' for s in shap_data],
            name='SHAP Value'
        ), secondary_y=False)

        shap_fig.update_layout(
            title="Top 5 Contributing ECG Timesteps",
            template="plotly_dark", height=300
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        st.markdown("**Interpretation:**")
        for s in shap_data:
            arrow = "🔴" if s['direction'] == 'anomaly' else "🟢"
            st.markdown(f"{arrow} Timestep **{s['timestep']}** pushed prediction toward **{s['direction']}** (SHAP={s['shap_value']:+.4f})")

        st.success(f"✅ Prediction logged to FDA audit trail | Record ID: {result['record_id']}")

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Analysis — 20 Random Beats")

    if st.button("▶ Run Batch Analysis", type="primary"):
        np.random.seed(99)
        sample_indices = np.random.choice(len(X_test), 20, replace=False)

        results = []
        progress = st.progress(0)
        for i, idx in enumerate(sample_indices):
            wf = X_test[idx].tolist()
            try:
                r = requests.post(f"{API_URL}/predict", json={
                    "patient_id": f"PT-{idx:04d}",
                    "device_id":  device_id,
                    "waveform":   wf
                }, timeout=30)
                d = r.json()
                results.append({
                    "Sample":     idx,
                    "True Class": classes[y_test[idx]],
                    "Prediction": d["prediction"],
                    "Confidence": d["confidence"],
                    "ISO Score":  round(d["iso_score"], 4),
                    "Recon Error":round(d["recon_error"], 6),
                    "Correct":    (d["prediction"]=="ANOMALY") == (y_test[idx]!=0)
                })
            except Exception:
                pass
            progress.progress((i+1)/20)

        df = pd.DataFrame(results)
        accuracy = df["Correct"].mean() * 100
        st.metric("Batch Accuracy", f"{accuracy:.1f}%")

        def color_pred(val):
            return "color: #ff4444" if val == "ANOMALY" else "color: #00d4aa"
        def color_correct(val):
            return "color: #00d4aa" if val else "color: #ff4444"

        st.dataframe(
            df.style
              .map(color_pred, subset=["Prediction"])
              .map(color_correct, subset=["Correct"]),
            use_container_width=True
        )

        # Distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["ISO Score"],
            marker_color='#00d4aa', opacity=0.7, name="ISO Score"
        ))
        fig.update_layout(template="plotly_dark", title="Isolation Forest Score Distribution",
                          height=300)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("FDA 21 CFR Part 11 — Audit Trail")
    st.caption("Every prediction is immutably logged with timestamp, model version, SHAP values, and waveform hash")

    audit_patient = st.text_input("Query patient ID", value="PT-001")
    if st.button("🔍 Retrieve Audit Log"):
        try:
            r    = requests.get(f"{API_URL}/audit/{audit_patient}", timeout=10)
            data = r.json()
            st.metric("Total Records", data["total_records"])
            if data["records"]:
                df_audit = pd.DataFrame(data["records"])
                st.dataframe(df_audit[[
                    "record_id","timestamp_utc","prediction",
                    "confidence","iso_score","recon_error","model_version"
                ]], use_container_width=True)
                st.download_button(
                    "⬇ Download Audit Log (CSV)",
                    df_audit.to_csv(index=False),
                    file_name=f"audit_{audit_patient}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not fetch audit log: {e}")

    st.markdown("---")
    st.markdown("""
    ### FDA 21 CFR Part 11 Compliance Features
    - **Unique record ID** per prediction (UUID4)
    - **UTC timestamp** on every record
    - **Immutable SQLite log** — INSERT only, no updates
    - **Waveform hash** for data integrity verification
    - **Model version** pinned per record
    - **SHAP explanation** stored with every decision
    """)
