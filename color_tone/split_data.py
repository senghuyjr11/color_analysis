import pandas as pd
from sklearn.model_selection import train_test_split

# === 1. Folder name fix: Italian → English ===
folder_map = {
    'autunno': 'autumn',
    'primavera': 'spring',
    'estate': 'summer',
    'inverno': 'winter'
}

def fix_paths(df):
    for it, en in folder_map.items():
        df['path'] = df['path'].str.replace(f"/{it}/", f"/{en}/", regex=False)
    return df

# Load original train CSV
train_df = pd.read_csv('./ORIGINAL_RGB_NOT_PROCESSED/cleaned_train.csv')
train_df = fix_paths(train_df)

# === 2. Stratified Train/Val split ===
train_part, val_part = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df['label'],
    random_state=42
)

# === 3. Load and fix test CSV too ===
val_part = fix_paths(val_part)
test_df = pd.read_csv('./ORIGINAL_RGB_NOT_PROCESSED/cleaned_test.csv')
test_df = fix_paths(test_df)

# === 4. Save all cleaned/fixed files ===
train_part.to_csv('./ORIGINAL_RGB_NOT_PROCESSED/cleaned_train.csv', index=False)
val_part.to_csv('./ORIGINAL_RGB_NOT_PROCESSED/cleaned_val.csv', index=False)
test_df.to_csv('./ORIGINAL_RGB_NOT_PROCESSED/cleaned_test.csv', index=False)

print(f"✅ Folder names fixed and new val split created.")
print(f"✅ Train: {len(train_part)} | Val: {len(val_part)} | Test: {len(test_df)}")
