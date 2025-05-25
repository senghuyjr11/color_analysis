import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load full dataset
df = pd.read_csv('res_parsed/hsv_skin_features.csv')

# Step 1: Split train + temp (val+test)
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)

# Step 2: Split val + test from temp
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

# Make sure output folder exists
os.makedirs("res_parsed", exist_ok=True)

# Save res_parsed
train_df.to_csv('res_parsed/train.csv', index=False)
val_df.to_csv('res_parsed/val.csv', index=False)
test_df.to_csv('res_parsed/test.csv', index=False)

print(f"Split complete:")
print(f" - Train: {len(train_df)}")
print(f" - Val:   {len(val_df)}")
print(f" - Test:  {len(test_df)}")
