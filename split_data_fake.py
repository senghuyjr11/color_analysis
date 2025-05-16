import os
import pandas as pd
from sklearn.model_selection import train_test_split

# === Set your actual dataset path ===
data_dir = "dataset_fake_parsing"
seasons = ["Spring", "Summer", "Autumn", "Winter"]

# === Collect only existing image files ===
data_entries = []
for season in seasons:
    season_folder = os.path.join(data_dir, season)
    if not os.path.exists(season_folder):
        continue
    for fname in os.listdir(season_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            relative_path = os.path.join(season, fname)
            full_path = os.path.join(data_dir, relative_path)
            if os.path.isfile(full_path):  # ✅ Check existence
                data_entries.append((relative_path, season))

# === Safety check ===
if not data_entries:
    raise ValueError("❌ No valid image files found in the dataset!")

# === Stratified split: 80% train, 10% val, 10% test ===
train_val, test = train_test_split(data_entries, test_size=0.1, random_state=42, stratify=[s for _, s in data_entries])
train, val = train_test_split(train_val, test_size=0.1111, random_state=42, stratify=[s for _, s in train_val])
# (0.1111 of 90% ≈ 10%, so final split is ~80/10/10)

# === Save to CSV files ===
for name, split in zip(["train", "val", "test"], [train, val, test]):
    df = pd.DataFrame(split, columns=["filename", "season"])
    df.to_csv(f"dataset_fake_parsing/{name}.csv", index=False)

print("✅ CSVs created: train.csv, val.csv, test.csv")
