import os
import pandas as pd
from sklearn.model_selection import train_test_split

# === PATHS ===
root_dir = "merge_dataset_cropped_segmented"
train_csv = "merge_dataset_cropped_segmented/train.csv"
val_csv = "merge_dataset_cropped_segmented/val.csv"
test_csv = "merge_dataset_cropped_segmented/test.csv"

# === COLLECT FILE PATHS ===
data = []
exts = (".jpg", ".jpeg", ".png", ".webp")

for season in os.listdir(root_dir):
    season_path = os.path.join(root_dir, season)
    if not os.path.isdir(season_path):
        continue

    for subtone in os.listdir(season_path):
        subtone_path = os.path.join(season_path, subtone)
        if not os.path.isdir(subtone_path):
            continue

        for file in os.listdir(subtone_path):
            if file.lower().endswith(exts):
                rel_path = os.path.join(season, subtone, file)  # keep relative path
                label = f"{season}_{subtone}"  # ✅ combine season and subtone
                data.append([rel_path, label])

df = pd.DataFrame(data, columns=["filename", "label"])
print(f"📂 Total images found: {len(df)}")

# === SPLIT 70/15/15 ===
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

# === SAVE CSVs ===
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"✅ train.csv: {len(train_df)} images")
print(f"✅ val.csv: {len(val_df)} images")
print(f"✅ test.csv: {len(test_df)} images")
