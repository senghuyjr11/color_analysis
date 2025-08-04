import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# CONFIG
# ======================
DATASET_DIR = "merge_dataset_cropped_segmented"
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
MODEL_PATH = os.path.join(DATASET_DIR, "convnext_season_large.pth")
HISTORY_PATH = os.path.join(DATASET_DIR, "season_train_history.json")

BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD TEST DATA
# ======================
df_test = pd.read_csv(TEST_CSV)
df_test['season'] = df_test['label'].apply(lambda x: x.split('_')[0])

# Label Encoder
season_labels = df_test['season'].tolist()
label_encoder = LabelEncoder()
label_encoder.fit(season_labels)
class_names = list(label_encoder.classes_)
print("Classes:", class_names)

# ======================
# DATASET CLASS
# ======================
class SeasonDataset(Dataset):
    def __init__(self, df, root_dir, target="season", transform=None, label_encoder=None):
        self.df = df
        self.root_dir = root_dir
        self.target = target
        self.transform = transform
        self.le = label_encoder
        self.df['label_encoded'] = self.le.transform(self.df[self.target])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename']).replace("\\", "/")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row['label_encoded'], dtype=torch.long)

# ======================
# TRANSFORMS & LOADER
# ======================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = SeasonDataset(df_test, DATASET_DIR, target="season", transform=test_transform, label_encoder=label_encoder)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ======================
# LOAD MODEL
# ======================
model = convnext_large(weights=None)  # no weights because we load trained ones
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ======================
# EVALUATION
# ======================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ======================
# METRICS
# ======================
print("\n✅ Test Accuracy: {:.2f}%".format((np.array(all_preds) == np.array(all_labels)).mean() * 100))

# Classification report
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\n📊 Classification Report:\n", report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix (Season Model)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ======================
# PLOT TRAINING CURVES (OPTIONAL)
# ======================
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Validation Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["val_acc"], label="Val Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No training history file found.")
