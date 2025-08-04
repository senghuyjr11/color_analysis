import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ======================
# CONFIG
# ======================
DATASET_DIR = "merge_dataset_cropped_segmented"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 50
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEASONS = ["spring", "summer", "autumn", "winter"]

# ======================
# LOAD DATA
# ======================
df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_test = pd.read_csv(TEST_CSV)

# Split columns
for df in [df_train, df_val, df_test]:
    df['season'] = df['label'].apply(lambda x: x.split('_')[0])
    df['subtone'] = df['label'].apply(lambda x: x.split('_')[1])

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
# TRANSFORMS
# ======================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======================
# TRAIN LOOP
# ======================
def train_subtone_model(season):
    print(f"\n===== Training Subtone Model for {season.upper()} =====")

    # Filter data for the season
    train_df = df_train[df_train['season'] == season].reset_index(drop=True)
    val_df = df_val[df_val['season'] == season].reset_index(drop=True)
    test_df = df_test[df_test['season'] == season].reset_index(drop=True)

    # Label encoder for subtones of this season
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['subtone'])
    num_classes = len(label_encoder.classes_)
    print(f"{season.upper()} subtones:", label_encoder.classes_)

    # Compute class weights for imbalance
    class_weights = compute_class_weight('balanced', classes=label_encoder.classes_, y=train_df['subtone'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # Datasets and loaders
    train_dataset = SubtoneDataset(train_df, DATASET_DIR, target="subtone", transform=train_transform, label_encoder=label_encoder)
    val_dataset = SubtoneDataset(val_df, DATASET_DIR, target="subtone", transform=val_transform, label_encoder=label_encoder)
    test_dataset = SubtoneDataset(test_df, DATASET_DIR, target="subtone", transform=val_transform, label_encoder=label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model = model.to(DEVICE)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # === TRAIN ===
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"[{season}] Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # === VALIDATE ===
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total

        print(f"{season.upper()} Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(DATASET_DIR, f"convnext_subtone_{season}.pth"))
            print(f"✅ Best {season} model saved with val_acc: {val_acc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"🛑 Early stopping for {season.upper()} model.")
            break

        scheduler.step()

    # Save training history
    with open(os.path.join(DATASET_DIR, f"train_history_subtone_{season}.json"), "w") as f:
        json.dump(history, f)

    print(f"📁 {season.upper()} subtone history saved.")

    # Final test accuracy
    model.load_state_dict(torch.load(os.path.join(DATASET_DIR, f"convnext_subtone_{season}.pth")))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"✅ Final Test Accuracy for {season.upper()}: {test_acc:.4f}")

# ======================
# TRAIN ALL SEASONS
# ======================
if __name__ == "__main__":
    for season in SEASONS:
        train_subtone_model(season)
