import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset.csv
df = pd.read_csv("dataset_fake/dataset.csv")

# Shuffle and split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['season'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['season'], random_state=42)

# Save the splits
train_df.to_csv("dataset_fake_rm_bg/train.csv", index=False)
val_df.to_csv("dataset_fake_rm_bg/val.csv", index=False)
test_df.to_csv("dataset_fake_rm_bg/test.csv", index=False)

print("Splits saved:")
print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")
