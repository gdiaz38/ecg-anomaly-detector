# 🫀 FDA-Grade ECG Anomaly Detector

Two-stage cardiac anomaly detection system trained on real MIT-BIH ECG data.
Built to reflect production requirements in regulated medical device environments.

---

## Architecture
```
MIT-BIH ECG Data (87k beats, 187 timesteps each)
            ↓
  Train on normal beats only (72k)
            ↓
  ┌─────────────────────────────┐
  │  Stage 1: Isolation Forest  │  ← global statistical outliers
  │  Stage 2: LSTM Autoencoder  │  ← temporal waveform anomalies
  └─────────────────────────────┘
            ↓
  SHAP Explainability Layer
  (which ECG timesteps drove the decision)
            ↓
  FastAPI REST Endpoint
            ↓
  FDA 21 CFR Part 11 Audit Log (SQLite)
```

---

## Why unsupervised?

Real medical devices rarely have labeled anomaly data at deployment time.
Training only on normal beats means the system flags anything that doesn't
look like a healthy heartbeat — without needing examples of every arrhythmia type.

---

## FDA 21 CFR Part 11 Compliance

Every prediction is logged with:

| Field | Purpose |
|-------|---------|
| `record_id` | UUID4 — unique immutable identifier |
| `timestamp_utc` | UTC timestamp — tamper-evident time record |
| `waveform_hash` | Data integrity verification |
| `shap_top5` | Explainability — required for regulatory review |
| `model_version` | Reproducibility — which model made this decision |
| `patient_id` / `device_id` | Chain of custody |

---

## Results

| Model | AUC |
|-------|-----|
| Isolation Forest | 0.75 |
| LSTM Autoencoder | 0.71 |

Trained exclusively on normal beats — no anomaly labels used at train time.

---

## Stack

- **Models:** Scikit-learn IsolationForest, PyTorch LSTM Autoencoder
- **Explainability:** SHAP KernelExplainer
- **API:** FastAPI + Uvicorn
- **Audit DB:** SQLite (append-only)
- **Dashboard:** Streamlit + Plotly
- **Data:** MIT-BIH Arrhythmia Database via PhysioNet

---

## Run It
```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn torch fastapi uvicorn streamlit plotly shap joblib kagglehub

python3 download_data.py
python3 features.py
python3 train.py
python3 explain.py

# Terminal 1
python3 api.py

# Terminal 2
streamlit run dashboard.py
```

## API
```bash
POST /predict
{
  "patient_id": "PT-001",
  "device_id": "MASIMO-4821",
  "waveform": [0.12, 0.34, ...]  # 187 timesteps
}

GET /audit/{patient_id}  # retrieve full audit trail
```
