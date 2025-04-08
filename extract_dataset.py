import csv
import os
import shutil

# Input directory (your raw structured dataset)
input_dir = "dataset"
# Output directory (flattened + ready for training)
output_dir = "dataset_real"

# Step 1: Flatten folder structure
for split in ["train", "test"]:
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        os.makedirs(os.path.join(output_dir, split, season), exist_ok=True)

    input_split = os.path.join(input_dir, split)

    for season in os.listdir(input_split):
        season_path = os.path.join(input_split, season)

        if os.path.isdir(season_path):
            for root, dirs, files in os.walk(season_path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        src = os.path.join(root, file)
                        dst = os.path.join(output_dir, split, season, file)
                        shutil.copy2(src, dst)

print("✅ Flattening complete.")

# Step 2: Create CSV files
for split in ["train", "test"]:
    csv_path = os.path.join(output_dir, f"{split}.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "season"])

        for season in os.listdir(os.path.join(output_dir, split)):
            season_dir = os.path.join(output_dir, split, season)
            for filename in os.listdir(season_dir):
                writer.writerow([filename, season])

print("✅ CSV files created for train and test.")
