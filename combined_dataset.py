import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Config ---
fake_dir = "dataset_fake_rm_bg"
real_dir = "dataset_real"
output_dir = "combined_dataset"
splits = ["train", "val", "test"]
seasons = ["Spring", "Summer", "Autumn", "Winter"]
resize_to = (224, 224)

# --- Create output folders ---
for split in splits:
    for season in seasons:
        os.makedirs(os.path.join(output_dir, split, season), exist_ok=True)

# --- Load CSVs ---
df_fake_train = pd.read_csv(os.path.join(fake_dir, "train.csv"))
df_fake_val = pd.read_csv(os.path.join(fake_dir, "val.csv"))
df_fake_test = pd.read_csv(os.path.join(fake_dir, "test.csv"))
df_real_train = pd.read_csv(os.path.join(real_dir, "train.csv"))
df_real_test = pd.read_csv(os.path.join(real_dir, "test.csv"))

# --- Split real train into train + val ---
df_real_train, df_real_val = train_test_split(
    df_real_train,
    test_size=0.2,
    stratify=df_real_train["season"],
    random_state=42
)

# --- Resize + Copy Function ---
def copy_and_resize(df, split, source_label, root_path):
    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {source_label} → {split}"):
        filename = row["filename"]
        season = row["season"].capitalize()
        src_path = os.path.join(root_path, season, filename)
        new_name = f"{source_label}_{filename}"
        dst_path = os.path.join(output_dir, split, season, new_name)

        if os.path.exists(src_path):
            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(resize_to, Image.BILINEAR)
                img.save(dst_path)
                new_rows.append({
                    "filename": new_name,
                    "season": row["season"],
                    "source": source_label
                })
            except Exception as e:
                print(f"❌ Failed to process {src_path}: {e}")
    return pd.DataFrame(new_rows)

# --- Copy all ---
train_df = pd.concat([
    copy_and_resize(df_fake_train, "train", "fake", fake_dir),
    copy_and_resize(df_real_train, "train", "real", os.path.join(real_dir, "train"))
])

val_df = pd.concat([
    copy_and_resize(df_fake_val, "val", "fake", fake_dir),
    copy_and_resize(df_real_val, "val", "real", os.path.join(real_dir, "train"))
])

test_df = pd.concat([
    copy_and_resize(df_fake_test, "test", "fake", fake_dir),
    copy_and_resize(df_real_test, "test", "real", os.path.join(real_dir, "test"))
])

# --- Save CSVs ---
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("\n✅ Combined dataset created and resized at:", output_dir)
