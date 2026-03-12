import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/shayanfazeli/heartbeat/versions/1"

print("Loading data...")
train = pd.read_csv(f"{DATA_PATH}/mitbih_train.csv", header=None)
test  = pd.read_csv(f"{DATA_PATH}/mitbih_test.csv",  header=None)

X_train_all = train.iloc[:, :-1].values.astype(np.float32)
y_train_all = train.iloc[:, -1].values.astype(int)
X_test_all  = test.iloc[:, :-1].values.astype(np.float32)
y_test_all  = test.iloc[:, -1].values.astype(int)

# ── KEY DECISION: train only on normal beats ──────────────────────────────────
normal_mask_train = y_train_all == 0
X_train_normal    = X_train_all[normal_mask_train]
print(f"Normal training beats: {len(X_train_normal)}")

# ── Scale to [0, 1] per feature ───────────────────────────────────────────────
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)

# Scale full test set with same scaler
X_test_scaled  = scaler.transform(X_test_all)

# ── Save everything ───────────────────────────────────────────────────────────
np.save("X_train_normal.npy", X_train_scaled)
np.save("X_test.npy",         X_test_scaled)
np.save("y_test.npy",         y_test_all)
joblib.dump(scaler, "scaler_ecg.pkl")

print(f"Train set shape: {X_train_scaled.shape}")
print(f"Test set shape:  {X_test_scaled.shape}")
print("\n=== TEST SET CLASS BREAKDOWN ===")
classes = {0:"Normal", 1:"Supraventricular", 2:"Ventricular", 3:"Fusion", 4:"Unknown"}
for k, v in classes.items():
    count = (y_test_all == k).sum()
    print(f"  {v:<20} {count:>5} samples")

print("\n✅ Saved X_train_normal.npy, X_test.npy, y_test.npy, scaler_ecg.pkl")
