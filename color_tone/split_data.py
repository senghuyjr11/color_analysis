import os
import pandas as pd
from sklearn.model_selection import train_test_split

def collect_image_paths(base_dir):
    data = []
    for season in os.listdir(base_dir):
        season_path = os.path.join(base_dir, season)
        if not os.path.isdir(season_path):
            continue
        for tone in os.listdir(season_path):
            tone_path = os.path.join(season_path, tone)
            if not os.path.isdir(tone_path):
                continue
            label = f"{season}_{tone}"
            for fname in os.listdir(tone_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    full_path = os.path.join(base_dir, season, tone, fname)
                    data.append((full_path.replace("\\", "/"), label))
    return data

# Base directory
base = "ORIGINAL_RGB_NOT_PROCESSED"
train_data = collect_image_paths(os.path.join(base, "train"))
test_data = collect_image_paths(os.path.join(base, "test"))

# Split 15% validation from train
train_list, val_list = train_test_split(train_data, test_size=0.15, stratify=[x[1] for x in train_data], random_state=42)

# Convert to DataFrame
df_train = pd.DataFrame(train_list, columns=["path", "label"])
df_val = pd.DataFrame(val_list, columns=["path", "label"])
df_test = pd.DataFrame(test_data, columns=["path", "label"])

# Save
df_train.to_csv("ORIGINAL_RGB_NOT_PROCESSED/train.csv", index=False)
df_val.to_csv("ORIGINAL_RGB_NOT_PROCESSED/val.csv", index=False)
df_test.to_csv("ORIGINAL_RGB_NOT_PROCESSED/test.csv", index=False)

print("CSV files created successfully.")
