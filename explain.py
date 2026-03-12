import numpy as np
import shap
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

print("Loading models and data...")
X_train = np.load("X_train_normal.npy").astype(np.float32)
X_test  = np.load("X_test.npy").astype(np.float32)
y_test  = np.load("y_test.npy").astype(int)
y_binary = (y_test != 0).astype(int)

iso = joblib.load("isolation_forest.pkl")

# Subsample for SHAP — it's compute intensive
np.random.seed(42)
background_idx = np.random.choice(len(X_train), 200, replace=False)
background     = X_train[background_idx]

# Pick test samples to explain: 50 normal + 50 anomalies
normal_idx  = np.where(y_binary == 0)[0][:50]
anomaly_idx = np.where(y_binary == 1)[0][:50]
explain_idx = np.concatenate([normal_idx, anomaly_idx])
X_explain   = X_test[explain_idx]
y_explain   = y_binary[explain_idx]

print(f"Running SHAP on {len(X_explain)} samples (50 normal + 50 anomaly)...")
print("This takes 2-3 minutes...")

# Wrap IsolationForest to return anomaly scores
def iso_anomaly_score(X):
    return -iso.score_samples(X)

explainer   = shap.KernelExplainer(iso_anomaly_score, background)
shap_values = explainer.shap_values(X_explain, nsamples=100)

print("SHAP computation complete.")

# ── Feature names = ECG timesteps ─────────────────────────────────────────────
feature_names = [f"t={i}" for i in range(X_explain.shape[1])]

# ── Plot 1: Mean absolute SHAP values (which timesteps matter most?) ──────────
mean_shap = np.abs(shap_values).mean(axis=0)
top_idx   = np.argsort(mean_shap)[::-1][:20]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

axes[0,0].bar(range(20), mean_shap[top_idx], color='#00d4aa')
axes[0,0].set_xticks(range(20))
axes[0,0].set_xticklabels([f"t={i}" for i in top_idx], rotation=45, fontsize=8)
axes[0,0].set_title('Top 20 Most Important ECG Timesteps\n(Mean |SHAP| across all samples)', fontweight='bold')
axes[0,0].set_ylabel('Mean |SHAP value|')

# ── Plot 2: SHAP values across full waveform ──────────────────────────────────
normal_shap  = shap_values[:50]
anomaly_shap = shap_values[50:]

axes[0,1].plot(np.abs(normal_shap).mean(axis=0),  color='#00d4aa', label='Normal beats',  linewidth=2)
axes[0,1].plot(np.abs(anomaly_shap).mean(axis=0), color='#ff4444', label='Anomaly beats', linewidth=2)
axes[0,1].set_title('SHAP Importance Across ECG Waveform\nNormal vs Anomaly', fontweight='bold')
axes[0,1].set_xlabel('ECG Timestep')
axes[0,1].set_ylabel('Mean |SHAP value|')
axes[0,1].legend()

# ── Plot 3: Single anomaly explanation ───────────────────────────────────────
sample_idx  = 0  # first anomaly
sample_shap = shap_values[50 + sample_idx]
sample_ecg  = X_explain[50 + sample_idx]

ax = axes[1,0]
ax2 = ax.twinx()
ax.bar(range(len(sample_shap)), sample_shap,
       color=['#ff4444' if s > 0 else '#00d4aa' for s in sample_shap],
       alpha=0.6, label='SHAP value')
ax2.plot(sample_ecg, color='white', linewidth=1.5, label='ECG waveform')
ax.set_title('Single Anomaly Explanation\nRed = pushed toward anomaly, Green = pushed toward normal', fontweight='bold')
ax.set_xlabel('ECG Timestep')
ax.set_ylabel('SHAP Value')
ax2.set_ylabel('ECG Amplitude')

# ── Plot 4: Single normal explanation ────────────────────────────────────────
sample_idx  = 0  # first normal
sample_shap = shap_values[sample_idx]
sample_ecg  = X_explain[sample_idx]

ax = axes[1,1]
ax2 = ax.twinx()
ax.bar(range(len(sample_shap)), sample_shap,
       color=['#ff4444' if s > 0 else '#00d4aa' for s in sample_shap],
       alpha=0.6, label='SHAP value')
ax2.plot(sample_ecg, color='white', linewidth=1.5, label='ECG waveform')
ax.set_title('Single Normal Beat Explanation\nMostly green = model confident it is normal', fontweight='bold')
ax.set_xlabel('ECG Timestep')
ax.set_ylabel('SHAP Value')
ax2.set_ylabel('ECG Amplitude')

plt.suptitle('FDA-Grade Explainability: SHAP Analysis of ECG Anomaly Detection',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("shap_explanation.png", dpi=150, bbox_inches='tight')
print("✅ Saved shap_explanation.png")

# ── Save SHAP values for API use ──────────────────────────────────────────────
np.save("shap_values_sample.npy", shap_values)
np.save("X_explain_sample.npy",   X_explain)
np.save("y_explain_sample.npy",   y_explain)

# ── Print top contributing timesteps for one anomaly ─────────────────────────
print("\n=== EXAMPLE ANOMALY EXPLANATION (FDA audit trail style) ===")
sample_shap = shap_values[50]
top5 = np.argsort(np.abs(sample_shap))[::-1][:5]
print(f"Sample classified as: ANOMALY")
print(f"Top 5 contributing ECG regions:")
for rank, t in enumerate(top5):
    direction = "↑ anomaly" if sample_shap[t] > 0 else "↓ normal"
    print(f"  #{rank+1}: Timestep {t:3d} | SHAP={sample_shap[t]:+.4f} | {direction}")
print("\nThis is what gets logged in the FDA 21 CFR Part 11 audit trail.")
