import os
import random
import pandas as pd

# === Settings ===
data_root = "clustered_seasons_kmeans"
train_csv = "csv/train.csv"
val_csv = "csv/val.csv"
test_csv = "csv/test.csv"
ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
assert abs(sum(ratios.values()) - 1.0) < 1e-6

# === Collect all image paths and labels ===
samples = []
for season in os.listdir(data_root):
    season_path = os.path.join(data_root, season)
    if not os.path.isdir(season_path):
        continue
    for subtype in os.listdir(season_path):
        label = f"{season}_{subtype}"
        subtype_path = os.path.join(season_path, subtype)
        if not os.path.isdir(subtype_path):
            continue
        for fname in os.listdir(subtype_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                file_path = os.path.join(season_path, subtype, fname)
                samples.append((file_path, label))

# === Shuffle and split ===
random.shuffle(samples)
n = len(samples)
n_train = int(ratios["train"] * n)
n_val = int(ratios["val"] * n)
n_test = n - n_train - n_val

train_samples = samples[:n_train]
val_samples = samples[n_train:n_train+n_val]
test_samples = samples[n_train+n_val:]

# === Create DataFrames and Save ===
pd.DataFrame(train_samples, columns=["file_path", "label"]).to_csv(train_csv, index=False)
pd.DataFrame(val_samples, columns=["file_path", "label"]).to_csv(val_csv, index=False)
pd.DataFrame(test_samples, columns=["file_path", "label"]).to_csv(test_csv, index=False)

print(f"CSVs saved: {train_csv} ({len(train_samples)}), {val_csv} ({len(val_samples)}), {test_csv} ({len(test_samples)})")