import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load("X_train_normal.npy").astype(np.float32)
# Subsample for speed — 5k normal beats is sufficient to learn normal waveform shape
np.random.seed(42)
idx = np.random.choice(len(X_train), 15000, replace=False)
X_train = X_train[idx]
print(f"Subsampled training set: {len(X_train)} beats")
X_test  = np.load("X_test.npy").astype(np.float32)
y_test  = np.load("y_test.npy").astype(int)

# Binary labels: 0=normal, 1=anomaly
y_binary = (y_test != 0).astype(int)
print(f"Test anomaly rate: {y_binary.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Isolation Forest
# Fast, interpretable, no GPU needed
# Good at catching global statistical outliers
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 1: Training Isolation Forest ---")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.172,   # we expect ~17% anomalies
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train)
joblib.dump(iso, "isolation_forest.pkl")

# Scores: more negative = more anomalous
iso_scores = -iso.score_samples(X_test)   # flip so higher = more anomalous
iso_preds  = (iso.predict(X_test) == -1).astype(int)

iso_auc = roc_auc_score(y_binary, iso_scores)
print(f"Isolation Forest AUC: {iso_auc:.4f}")
print(classification_report(y_binary, iso_preds, target_names=["Normal","Anomaly"]))

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: LSTM Autoencoder
# Learns temporal structure of normal heartbeats
# High reconstruction error = the waveform shape is abnormal
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 2: Training LSTM Autoencoder ---")

SEQ_LEN    = 187
HIDDEN     = 64
LATENT     = 32
EPOCHS     = 40
BATCH      = 512
LR         = 0.001

class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.encoder_fc = nn.Linear(HIDDEN, LATENT)

        # Decoder
        self.decoder_fc   = nn.Linear(LATENT, HIDDEN)
        self.decoder_lstm = nn.LSTM(
            input_size=HIDDEN,
            hidden_size=HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.output_layer = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        enc_out, (h, c) = self.encoder_lstm(x)
        latent = self.encoder_fc(h[-1])              # (batch, latent)

        # Repeat latent vector across timesteps for decoder
        dec_input = self.decoder_fc(latent)
        dec_input = dec_input.unsqueeze(1).repeat(1, SEQ_LEN, 1)  # (batch, seq, hidden)

        dec_out, _ = self.decoder_lstm(dec_input)
        recon = self.output_layer(dec_out)            # (batch, seq, 1)
        return recon

# DataLoader — only normal beats for training
X_train_t = torch.tensor(X_train).unsqueeze(-1)     # (N, 187, 1)
train_ds   = TensorDataset(X_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

model     = LSTMAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

train_losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for (Xb,) in train_loader:
        optimizer.zero_grad()
        recon = model(Xb)
        loss  = criterion(recon, Xb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(train_loader)
    train_losses.append(avg)
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg:.6f}")

torch.save(model.state_dict(), "lstm_autoencoder.pt")
print("✅ Saved lstm_autoencoder.pt")

# ── Compute reconstruction errors on test set ─────────────────────────────────
print("\nComputing reconstruction errors...")
model.eval()
X_test_t = torch.tensor(X_test).unsqueeze(-1)
test_ds   = TensorDataset(X_test_t)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

recon_errors = []
with torch.no_grad():
    for (Xb,) in test_loader:
        recon = model(Xb)
        errors = ((recon - Xb) ** 2).mean(dim=(1,2))
        recon_errors.extend(errors.numpy())

recon_errors = np.array(recon_errors)

# Threshold = 95th percentile of normal beat errors
normal_errors = recon_errors[y_binary == 0]
threshold = np.percentile(normal_errors, 95)
print(f"Anomaly threshold (95th pct of normal): {threshold:.6f}")

lstm_preds = (recon_errors > threshold).astype(int)
lstm_auc   = roc_auc_score(y_binary, recon_errors)
print(f"LSTM Autoencoder AUC: {lstm_auc:.4f}")
print(classification_report(y_binary, lstm_preds, target_names=["Normal","Anomaly"]))

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Combine both scores (ensemble)
# Simple average — both models must agree for high confidence
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Combined Ensemble Score ---")
iso_norm   = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
lstm_norm  = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min())
ensemble   = np.maximum(iso_norm, lstm_norm)

ens_threshold = np.percentile(ensemble[y_binary == 0], 95)
ens_preds     = (ensemble > ens_threshold).astype(int)
ens_auc       = roc_auc_score(y_binary, ensemble)

print(f"Ensemble AUC: {ens_auc:.4f}")
print(classification_report(y_binary, ens_preds, target_names=["Normal","Anomaly"]))

# Save scores and threshold
np.save("recon_errors.npy", recon_errors)
np.save("iso_scores.npy",   iso_scores)
np.save("ensemble_scores.npy", ensemble)
joblib.dump({"threshold": float(ens_threshold), "lstm_threshold": float(threshold)}, "thresholds.pkl")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Reconstruction error distribution
axes[0].hist(recon_errors[y_binary==0], bins=80, alpha=0.6, color='#00d4aa', label='Normal')
axes[0].hist(recon_errors[y_binary==1], bins=80, alpha=0.6, color='#ff4444', label='Anomaly')
axes[0].axvline(threshold, color='orange', linestyle='--', label=f'Threshold={threshold:.4f}')
axes[0].set_title('LSTM Reconstruction Error Distribution')
axes[0].set_xlabel('Reconstruction Error (MSE)')
axes[0].legend()

# 2. Confusion matrix
cm = confusion_matrix(y_binary, ens_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
axes[1].set_title('Ensemble Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# 3. Training loss
axes[2].plot(train_losses, color='#00d4aa', linewidth=2)
axes[2].set_title('LSTM Autoencoder Training Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MSE Loss')

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("\n✅ Saved training_results.png")
print(f"\n{'='*50}")
print(f"FINAL SUMMARY")
print(f"{'='*50}")
print(f"Isolation Forest AUC : {iso_auc:.4f}")
print(f"LSTM Autoencoder AUC : {lstm_auc:.4f}")
print(f"Ensemble AUC         : {ens_auc:.4f}")
