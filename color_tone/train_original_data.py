import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from tqdm import tqdm

# === Config ===
DATA_DIR = './ORIGINAL_RGB_NOT_PROCESSED'
TRAIN_CSV = os.path.join(DATA_DIR, 'cropped_4class_train.csv')
VAL_CSV   = os.path.join(DATA_DIR, 'cropped_4class_val.csv')

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Label Encoding
def encode_labels(labels):
    classes = sorted(list(set(labels)))
    label2idx = {cls: idx for idx, cls in enumerate(classes)}
    idx2label = {idx: cls for cls, idx in label2idx.items()}
    encoded = [label2idx[label] for label in labels]
    return encoded, label2idx, idx2label

# === Dataset
class ColorToneDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.labels, self.label2idx, self.idx2label = encode_labels(self.df['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, self.df.iloc[idx]['path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# === Transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# === Load Datasets
train_set = ColorToneDataset(TRAIN_CSV, transform=train_transform)
val_set = ColorToneDataset(VAL_CSV, transform=val_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# === Model
weights = EfficientNet_B2_Weights.IMAGENET1K_V1
model = efficientnet_b2(weights=weights).to(DEVICE)
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
).to(DEVICE)

# === Class Weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_set.labels),
                                     y=train_set.labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# === Loss, Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# === Training Loop
best_acc = 0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0

    for images, labels in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[Val Epoch {epoch+1}]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'best_efficientnet_b2.pth')
        print("  🔥 Best model updated.")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("  🛑 Early stopping.")
            break

# === Save History
with open("efficientnet_b2_history.json", "w") as f:
    json.dump(history, f)

print("✅ Training complete. Best model saved to best_efficientnet_b2.pth.")
