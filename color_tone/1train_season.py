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

torch.backends.cudnn.benchmark = True

# ======================
# CONFIGURATION
# ======================
DATASET_DIR = "merge_dataset_cropped_segmented"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
MODEL_PATH = os.path.join(DATASET_DIR, "convnext_season_large.pth")
HISTORY_PATH = os.path.join(DATASET_DIR, "season_train_history.json")

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 50
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STEP 1: Split dataset into season/subtone
df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_test = pd.read_csv(TEST_CSV)

for df in [df_train, df_val, df_test]:
    df['season'] = df['label'].apply(lambda x: x.split('_')[0])
    df['subtone'] = df['label'].apply(lambda x: x.split('_')[1])

# ======================
# DATASET CLASS
# ======================
# STEP 2: Define separate datasets for season & subtone
class SeasonSubtoneDataset(Dataset):
    def __init__(self, df, root_dir, target="season", transform=None, label_encoder=None):
        self.df = df
        self.root_dir = root_dir
        self.target = target
        self.transform = transform
        self.le = label_encoder or LabelEncoder()
        self.df['label_encoded'] = self.le.fit_transform(self.df[self.target])

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
# DATA TRANSFORMS
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
# LABEL ENCODER
# ======================
season_labels = df_train['season'].tolist()
label_encoder = LabelEncoder()
label_encoder.fit(season_labels)

num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)

# ======================
# LOAD DATASETS
# ======================
train_dataset = SeasonSubtoneDataset(df_train, DATASET_DIR, target="season", transform=train_transform)
val_dataset   = SeasonSubtoneDataset(df_val, DATASET_DIR, target="season", transform=val_transform)
test_dataset  = SeasonSubtoneDataset(df_test, DATASET_DIR, target="season", transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ======================
# MODEL
# ======================
# STEP 3: Train Season Model
# num_classes = 4
model_season = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model_season.classifier[2] = nn.Linear(model_season.classifier[2].in_features, 4)
in_features = model_season.classifier[2].in_features
model = model_season.to(DEVICE)

# ======================
# LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ======================
# TRAINING LOOP
# ======================
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    if epoch == 15:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
        print("🔽 Learning rate dropped to 1e-5 for fine-tuning")

    model.train()
    running_loss = 0.0
    total_train_samples = 0

    for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        # MixUp applied here
        images, targets_a, targets_b, lam = mixup_data(images, labels)
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_train_samples += labels.size(0)

    train_loss = running_loss / len(train_loader)

    # ----- VALIDATE -----
    model.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

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

    # ----- PRINT EPOCH RESULTS -----
    print(f"Epoch {epoch+1}/{EPOCHS} | "
      f"Train Loss: {train_loss:.4f} | "
      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


    # ----- SAVE HISTORY -----
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # ----- SAVE BEST MODEL -----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Best model saved with val_acc: {val_acc:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # ----- EARLY STOPPING -----
    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered.")
        break

    scheduler.step()

# ======================
# SAVE HISTORY
# ======================
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f)
print(f"Training history saved to {HISTORY_PATH}")

# ======================
# TEST EVALUATION
# ======================
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
print(f"Final Test Accuracy: {test_acc:.4f}")
