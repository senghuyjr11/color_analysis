import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import timm
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd
import pickle
from pathlib import Path

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
class FastEmotionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        unique_labels = sorted(self.data["label"].dropna().unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.data["label_idx"] = self.data["label"].map(self.label_map)
        self.data = self.data.dropna(subset=["label_idx"]).copy()
        self.data["label_idx"] = self.data["label_idx"].astype('int64')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / row["label"] / row["filename"]

        try:
            image = Image.open(img_path).convert("RGB")
            if image.size != (224, 224):
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
        except:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, row["label_idx"]

# Model with dropout
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
        in_features = self.base_model.num_features
        self.base_model.reset_classifier(0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        return self.classifier(x)

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=6, delta=0.001, save_path='models/emotion_model_efficientnet.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def __call__(self, val_loss, val_acc, model):
        score = val_acc
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"New best model saved. Val Acc: {val_acc:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def update_history(self, train_loss, val_loss, train_acc, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)

# Training
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device).long()
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val"):
            imgs, labels = imgs.to(device), labels.to(device).long()
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# Main
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocessed_path = "archive/preprocessed_faces"
    base_path = Path(preprocessed_path)

    train_set = FastEmotionDataset(base_path / "train_split.csv", base_path / "train_split", transform=train_transform)
    val_set = FastEmotionDataset(base_path / "val_split.csv", base_path / "val_split", transform=val_transform)

    label_counts = train_set.data['label_idx'].value_counts().sort_index()
    weights = 1.0 / (label_counts ** 0.5)
    sample_weights = train_set.data['label_idx'].map(weights).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)

    train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = CustomEfficientNet(train_set.data['label_idx'].nunique()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    scaler = torch.cuda.amp.GradScaler()
    stopper = EarlyStopping(patience=6)

    for epoch in range(1, 31):
        print(f"Epoch {epoch}/30")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        stopper.update_history(train_loss, val_loss, train_acc, val_acc)
        stopper(val_loss, val_acc, model)

        if stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    os.makedirs('models', exist_ok=True)
    with open('models/training_history_efficientnet.pkl', 'wb') as f:
        pickle.dump(stopper.history, f)

    print(f"Training complete. Best validation accuracy: {stopper.best_score:.4f}")
