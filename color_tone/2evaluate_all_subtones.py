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
from torchvision.models import convnext_large
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# CONFIG
# ======================
DATASET_DIR = "merge_dataset_cropped_segmented"
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

SEASONS = ["spring", "summer", "autumn", "winter"]

BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD TEST DATA
# ======================
df_test = pd.read_csv(TEST_CSV)
df_test['season'] = df_test['label'].apply(lambda x: x.split('_')[0])
df_test['subtone'] = df_test['label'].apply(lambda x: x.split('_')[1])

# ======================
# DATASET CLASS
# ======================
class SubtoneDataset(Dataset):
    def __init__(self, df, root_dir, target="subtone", transform=None, label_encoder=None):
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
# TRANSFORM
# ======================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======================
# EVALUATE FUNCTION
# ======================
def evaluate_subtone_model(season):
    print(f"\n🔍 Evaluating Subtone Model for {season.upper()}")

    # Filter test data for this season
    season_test_df = df_test[df_test['season'] == season].reset_index(drop=True)

    # Label encoder for this season's subtone labels
    label_encoder = LabelEncoder()
    label_encoder.fit(season_test_df['subtone'])
    class_names = list(label_encoder.classes_)
    print(f"Classes for {season.upper()}: {class_names}")

    # Dataset and loader
    test_dataset = SubtoneDataset(season_test_df, DATASET_DIR, target="subtone",
                                  transform=test_transform, label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Load model
    model_path = os.path.join(DATASET_DIR, f"convnext_subtone_{season}.pth")
    model = convnext_large(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Prediction loop
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {season}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Test Accuracy
    test_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    print(f"✅ Test Accuracy for {season.upper()}: {test_acc:.2f}%")

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title(f"{season.upper()} Subtone Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Plot training curves (if available)
    history_path = os.path.join(DATASET_DIR, f"train_history_subtone_{season}.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

        epochs_range = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(12, 5))

        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss")
        plt.title(f"{season.upper()} Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history["val_acc"], label="Val Accuracy")
        plt.title(f"{season.upper()} Validation Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ No training history for {season.upper()}.")

# ======================
# RUN EVALUATION FOR ALL SEASONS
# ======================
if __name__ == "__main__":
    for season in SEASONS:
        evaluate_subtone_model(season)
