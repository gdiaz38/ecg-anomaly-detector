import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/shayanfazeli/heartbeat/versions/1"

print("Loading MIT-BIH training data...")
train = pd.read_csv(f"{DATA_PATH}/mitbih_train.csv", header=None)
test  = pd.read_csv(f"{DATA_PATH}/mitbih_test.csv",  header=None)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# Last column is the label
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values.astype(int)

print(f"\nFeatures per sample: {X_train.shape[1]} (ECG timesteps)")
print(f"Training samples:    {len(X_train)}")

# Class distribution
print("\n=== CLASS DISTRIBUTION ===")
classes = {
    0: "Normal",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
}
for k, v in classes.items():
    count = (y_train == k).sum()
    pct   = count / len(y_train) * 100
    print(f"  Class {k} — {v:<30} {count:>6} samples ({pct:.1f}%)")

# Plot one sample of each class
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
for cls in range(5):
    idx = np.where(y_train == cls)[0][0]
    axes[cls].plot(X_train[idx], color='#00d4aa', linewidth=1.2)
    axes[cls].set_title(f"Class {cls}: {classes[cls]}", fontsize=9)
    axes[cls].set_xlabel("Timestep")
    axes[cls].axhline(0, color='gray', linewidth=0.5)
    axes[cls].set_ylim(-0.2, 1.2)

plt.suptitle("ECG Waveform Samples by Class", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("ecg_samples.png", dpi=150)
print("\n✅ Saved ecg_samples.png")

# This is critical for anomaly detection framing:
print("\n=== ANOMALY DETECTION FRAMING ===")
normal_count   = (y_train == 0).sum()
abnormal_count = (y_train != 0).sum()
print(f"Normal beats:   {normal_count} ({normal_count/len(y_train)*100:.1f}%)")
print(f"Abnormal beats: {abnormal_count} ({abnormal_count/len(y_train)*100:.1f}%)")
