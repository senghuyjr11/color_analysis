import pandas as pd
from sklearn.model_selection import train_test_split

# Load original training CSV
df = pd.read_csv("raf_db_dataset/train_labels.csv")

# Split: 85% train, 15% val
train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)

# Save the new splits
train_df.to_csv("raf_db_dataset/train_split.csv", index=False)
val_df.to_csv("raf_db_dataset/val_split.csv", index=False)
