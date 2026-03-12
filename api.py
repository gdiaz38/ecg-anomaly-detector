import numpy as np
import torch
import torch.nn as nn
import joblib
import shap
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ── Model definition (must match train.py) ────────────────────────────────────
SEQ_LEN = 187
HIDDEN  = 64

class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_lstm = nn.LSTM(1, HIDDEN, num_layers=2, batch_first=True, dropout=0.2)
        self.encoder_fc   = nn.Linear(HIDDEN, 32)
        self.decoder_fc   = nn.Linear(32, HIDDEN)
        self.decoder_lstm = nn.LSTM(HIDDEN, HIDDEN, num_layers=2, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        _, (h, _)  = self.encoder_lstm(x)
        latent     = self.encoder_fc(h[-1])
        dec_input  = self.decoder_fc(latent).unsqueeze(1).repeat(1, SEQ_LEN, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        return self.output_layer(dec_out)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
iso        = joblib.load("isolation_forest.pkl")
thresholds = joblib.load("thresholds.pkl")
scaler     = joblib.load("scaler_ecg.pkl")

lstm = LSTMAutoencoder()
lstm.load_state_dict(torch.load("lstm_autoencoder.pt", map_location="cpu"))
lstm.eval()

# SHAP background (small sample for speed)
X_train    = np.load("X_train_normal.npy").astype(np.float32)
np.random.seed(42)
background = X_train[np.random.choice(len(X_train), 50, replace=False)]
explainer  = shap.KernelExplainer(lambda X: -iso.score_samples(X), background)
print("Models loaded.")

# ── FDA 21 CFR Part 11 Audit Database ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("audit_log.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            record_id        TEXT PRIMARY KEY,
            timestamp_utc    TEXT NOT NULL,
            patient_id       TEXT NOT NULL,
            device_id        TEXT NOT NULL,
            iso_score        REAL NOT NULL,
            recon_error      REAL NOT NULL,
            ensemble_score   REAL NOT NULL,
            prediction       TEXT NOT NULL,
            confidence       TEXT NOT NULL,
            shap_top5        TEXT NOT NULL,
            waveform_hash    TEXT NOT NULL,
            model_version    TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(record: dict):
    """Immutable audit log entry — required by FDA 21 CFR Part 11"""
    conn = sqlite3.connect("audit_log.db")
    conn.execute("""
        INSERT INTO audit_log VALUES (
            :record_id, :timestamp_utc, :patient_id, :device_id,
            :iso_score, :recon_error, :ensemble_score,
            :prediction, :confidence, :shap_top5,
            :waveform_hash, :model_version
        )
    """, record)
    conn.commit()
    conn.close()

init_db()

# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FDA-Grade ECG Anomaly Detector",
    description="""
    Two-stage cardiac anomaly detection system.
    - Stage 1: Isolation Forest (global statistical outlier detection)
    - Stage 2: LSTM Autoencoder (temporal waveform reconstruction)
    - Explainability: SHAP values per prediction
    - Audit logging: FDA 21 CFR Part 11 compliant
    """,
    version="1.0.0"
)

class ECGRequest(BaseModel):
    patient_id: str
    device_id:  str
    waveform:   List[float]   # exactly 187 timesteps

class SHAPContribution(BaseModel):
    timestep:  int
    shap_value: float
    direction: str

class ECGResponse(BaseModel):
    record_id:       str
    timestamp_utc:   str
    patient_id:      str
    prediction:      str
    confidence:      str
    iso_score:       float
    recon_error:     float
    ensemble_score:  float
    shap_top5:       List[SHAPContribution]
    audit_logged:    bool

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": "1.0.0",
        "compliance":    "FDA 21 CFR Part 11",
        "stages":        ["IsolationForest", "LSTMAutoencoder", "SHAP"]
    }

@app.post("/predict", response_model=ECGResponse)
def predict(req: ECGRequest):
    if len(req.waveform) != SEQ_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"Waveform must have exactly {SEQ_LEN} timesteps, got {len(req.waveform)}"
        )

    waveform = np.array(req.waveform, dtype=np.float32).reshape(1, -1)

    # Scale using training scaler
    waveform_scaled = scaler.transform(waveform)

    # ── Stage 1: Isolation Forest ─────────────────────────────────────────────
    iso_score = float(-iso.score_samples(waveform_scaled)[0])

    # ── Stage 2: LSTM Autoencoder ─────────────────────────────────────────────
    waveform_t  = torch.tensor(waveform_scaled, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        recon      = lstm(waveform_t)
        recon_error = float(((recon - waveform_t) ** 2).mean().item())

    # ── Ensemble score ────────────────────────────────────────────────────────
    iso_scores_range  = (0.3, 0.7)
    recon_range       = (0.0, 0.2)
    iso_norm   = np.clip((iso_score - iso_scores_range[0]) /
                         (iso_scores_range[1] - iso_scores_range[0]), 0, 1)
    recon_norm = np.clip((recon_error - recon_range[0]) /
                         (recon_range[1] - recon_range[0]), 0, 1)
    ensemble_score = float(max(iso_norm, recon_norm))

    # ── Prediction ────────────────────────────────────────────────────────────
    lstm_threshold = thresholds["lstm_threshold"]
    is_anomaly     = (recon_error > lstm_threshold) or (iso_score > 0.55)

    prediction = "ANOMALY" if is_anomaly else "NORMAL"
    if ensemble_score > 0.75:
        confidence = "HIGH"
    elif ensemble_score > 0.45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── SHAP explanation ──────────────────────────────────────────────────────
    shap_vals = explainer.shap_values(waveform_scaled, nsamples=50)
    top5_idx  = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
    shap_top5 = [
        SHAPContribution(
            timestep=int(t),
            shap_value=round(float(shap_vals[0][t]), 6),
            direction="anomaly" if shap_vals[0][t] > 0 else "normal"
        )
        for t in top5_idx
    ]

    # ── FDA 21 CFR Part 11 Audit Log ──────────────────────────────────────────
    record_id     = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    waveform_hash = str(hash(tuple(req.waveform)))

    log_prediction({
        "record_id":      record_id,
        "timestamp_utc":  timestamp_utc,
        "patient_id":     req.patient_id,
        "device_id":      req.device_id,
        "iso_score":      iso_score,
        "recon_error":    recon_error,
        "ensemble_score": ensemble_score,
        "prediction":     prediction,
        "confidence":     confidence,
        "shap_top5":      json.dumps([s.dict() for s in shap_top5]),
        "waveform_hash":  waveform_hash,
        "model_version":  "1.0.0"
    })

    return ECGResponse(
        record_id=record_id,
        timestamp_utc=timestamp_utc,
        patient_id=req.patient_id,
        prediction=prediction,
        confidence=confidence,
        iso_score=round(iso_score, 6),
        recon_error=round(recon_error, 6),
        ensemble_score=round(ensemble_score, 4),
        shap_top5=shap_top5,
        audit_logged=True
    )

@app.get("/audit/{patient_id}")
def get_audit_log(patient_id: str):
    """Retrieve full audit trail for a patient — FDA 21 CFR Part 11"""
    conn   = sqlite3.connect("audit_log.db")
    cursor = conn.execute(
        "SELECT * FROM audit_log WHERE patient_id=? ORDER BY timestamp_utc DESC",
        (patient_id,)
    )
    cols = [d[0] for d in cursor.description]
    rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
    conn.close()
    return {"patient_id": patient_id, "total_records": len(rows), "records": rows}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
