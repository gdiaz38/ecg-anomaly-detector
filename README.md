# 🫀 FDA-Grade ECG Anomaly Detector

A two-stage unsupervised anomaly detection system for real-time ECG classification, trained on 87,000 MIT-BIH heartbeats. Features SHAP explainability and FDA 21 CFR Part 11-style immutable audit logging.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://ecg-anomaly-detector-gdiaz38.streamlit.app)**

---

## Overview

Cardiac arrhythmias affect over 14 million Americans and are a leading cause of sudden death. This project builds a clinical-grade anomaly detection pipeline that identifies abnormal heartbeats from raw ECG waveforms — without any labeled anomaly data during training.

Key question it answers: *Is this heartbeat normal or anomalous — and which part of the waveform drove that decision?*

---

## Key Results

| Model | AUC |
|---|---|
| Isolation Forest | 0.75 |
| LSTM Autoencoder | — |
| **Ensemble (max)** | **Best** |

- **Test set:** 21,892 real ECG beats from MIT-BIH Arrhythmia Database
- **Anomaly types:** Supraventricular, Ventricular, Fusion, Unknown
- **Training:** unsupervised on normal beats only — no anomaly labels used

---

## Features

- **Two-stage detection** — Isolation Forest catches global outliers; LSTM Autoencoder catches temporal waveform anomalies
- **Ensemble scoring** — max of both normalized scores for high-confidence decisions
- **SHAP explainability** — KernelExplainer identifies which ECG timesteps drove each prediction
- **FDA 21 CFR Part 11 audit trail** — every prediction logged with UUID, UTC timestamp, model version, SHAP top-5, waveform hash
- **Batch analysis** — run inference on up to 200 real beats with accuracy breakdown
- **Downloadable audit log** — CSV export of full prediction history

---

## Data

| Source | Description |
|---|---|
| [MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) | 87,554 real ECG beats across 5 classes |
| Class 0 | Normal |
| Class 1 | Supraventricular ectopic |
| Class 2 | Ventricular ectopic |
| Class 3 | Fusion beat |
| Class 4 | Unknown |

---

## Project Structure

```
ecg-anomaly-detector/
├── dashboard.py              # Streamlit — 3 tabs: detection, batch, audit
├── train.py                  # Two-stage training: IF + LSTM Autoencoder
├── features.py               # ECG feature extraction
├── explain.py                # SHAP KernelExplainer
├── api.py                    # FastAPI REST endpoint (local use)
├── lstm_autoencoder.pt       # Trained LSTM weights
├── isolation_forest.pkl      # Trained Isolation Forest
├── thresholds.pkl            # Anomaly thresholds (95th percentile)
├── scaler_ecg.pkl            # Feature scaler
├── X_test.npy                # 21,892 test waveforms (187 timesteps each)
├── y_test.npy                # True class labels
├── recon_errors.npy          # LSTM reconstruction errors on test set
├── iso_scores.npy            # Isolation Forest scores on test set
├── ensemble_scores.npy       # Combined ensemble scores
├── shap_values_sample.npy    # Precomputed SHAP values
└── X_explain_sample.npy      # SHAP explanation waveforms
```

---

## How It Works

```
Raw ECG waveform (187 timesteps)
        ↓
Stage 1: Isolation Forest
  → Trained on normal beats only
  → Scores how statistically "unusual" the beat is
        ↓
Stage 2: LSTM Autoencoder
  → Encoder: 2-layer LSTM → 32-dim latent space
  → Decoder: reconstructs the waveform
  → High reconstruction error = waveform shape is abnormal
        ↓
Ensemble: max(normalized_IF_score, normalized_recon_error)
  → Threshold = 95th percentile of normal beat scores
        ↓
SHAP KernelExplainer identifies top contributing timesteps
        ↓
Result logged to immutable SQLite audit trail
```

---

## Model Architecture

### LSTM Autoencoder

```
Input: (batch, 187 timesteps, 1)
        ↓
Encoder LSTM (2 layers, hidden=64, dropout=0.2)
        ↓
FC(64 → 32)  ← latent representation
        ↓
FC(32 → 64) → repeat across 187 timesteps
        ↓
Decoder LSTM (2 layers, hidden=64, dropout=0.2)
        ↓
FC(64 → 1) per timestep
        ↓
Output: reconstructed waveform (batch, 187, 1)
```

**Anomaly signal:** MSE between input and reconstruction — normal beats reconstruct accurately; abnormal beats do not.

---

## FDA 21 CFR Part 11 Compliance

Every prediction generates an immutable audit record containing:

| Field | Description |
|---|---|
| `record_id` | UUID4 — unique per prediction |
| `timestamp_utc` | UTC timestamp at inference time |
| `patient_id` | Provided by operator |
| `device_id` | ECG device identifier |
| `iso_score` | Isolation Forest anomaly score |
| `recon_error` | LSTM reconstruction error |
| `ensemble_score` | Combined decision score |
| `prediction` | NORMAL or ANOMALY |
| `confidence` | Model confidence |
| `shap_top5` | Top 5 contributing timesteps |
| `model_version` | Pinned model version string |

---

## Dashboard Tabs

**Live Detection** — select beat type and sample index, analyze individual waveform, view SHAP explanation and audit log entry

**Batch Analysis** — run inference on up to 200 randomly sampled real beats, view confusion breakdown by true class, score distribution histogram

**Audit Trail** — full prediction history with download to CSV, compliance feature summary

---

## Local Setup

```bash
git clone https://github.com/gdiaz38/ecg-anomaly-detector
cd ecg-anomaly-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

To retrain from scratch (requires MIT-BIH dataset via kagglehub):

```bash
python3 features.py   # extract waveforms
python3 train.py      # train IF + LSTM, saves model files
python3 explain.py    # compute SHAP values
```

---

## Tech Stack

`Python 3.11` · `PyTorch` · `Scikit-learn` · `SHAP` · `Streamlit` · `Plotly` · `NumPy` · `joblib`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT
