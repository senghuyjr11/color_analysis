import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd

# ------------------ EarlyStopping Class (FIXED) ------------------
class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, save_path='models/best_emotion_model.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def __call__(self, val_loss, val_acc, model):
        score = val_acc
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"🎯 New best model saved! Val Acc: {val_acc:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def update_history(self, train_loss, val_loss, train_acc, val_acc):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)


# ------------------ EmotionDataset Class (FIXED) ------------------
class EmotionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mtcnn=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mtcnn = mtcnn  # MTCNN detector

        unique_labels = sorted(self.data["label"].dropna().unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        print(f"[{os.path.basename(csv_file)}] {self.num_classes} classes: {unique_labels}")

        self.data["label_idx"] = self.data["label"].map(self.label_map)
        self.data = self.data.dropna(subset=["label_idx"]).copy()
        self.data["label_idx"] = self.data["label_idx"].astype('int64')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["label"], row["filename"])

        try:
            image = Image.open(img_path).convert("RGB")
            if self.mtcnn:
                face = self.mtcnn(image)
                if face is not None:
                    image = transforms.ToPILImage()(face)
        except Exception as e:
            print(f"[Warning] Failed to load {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, row["label_idx"]


# ------------------ Image Transforms ------------------
common_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    common_normalize
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    common_normalize
])


# ------------------ Data Setup ------------------
def setup_data():
    mtcnn = None  # Disabled for debugging
    base_path = "archive/data_balanced_1x"
    train_csv = os.path.join(base_path, "train_split.csv")
    val_csv = os.path.join(base_path, "val_split.csv")
    test_csv = os.path.join(base_path, "test.csv")
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")

    for file_path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")

    train_set = EmotionDataset(train_csv, train_dir, transform=train_transform, mtcnn=mtcnn)
    val_set = EmotionDataset(val_csv, train_dir, transform=val_transform, mtcnn=mtcnn)
    test_set = EmotionDataset(test_csv, test_dir, transform=val_transform, mtcnn=mtcnn)

    num_classes = train_set.num_classes
    assert val_set.num_classes == test_set.num_classes == num_classes, "Inconsistent class counts"

    return train_set, val_set, test_set, num_classes


# ------------------ Simple Model for Debugging ------------------
class SimpleEmotionClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

        # Simple initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)


def create_model(num_classes, device):
    print("Creating model with SimpleEmotionClassifier for debugging...")
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = base.fc.in_features

    # Use simple classifier for debugging
    base.fc = SimpleEmotionClassifier(in_features, num_classes)
    model = base.to(device)

    # Test with eval mode to avoid BatchNorm issues
    model.eval()
    test_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(test_input)
        print(f"Test output shape: {output.shape}")
        print(f"Test output range: {output.min().item():.3f} to {output.max().item():.3f}")
    model.train()

    return model


# ------------------ Training (ENHANCED WITH DEBUGGING) ------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for i, (imgs, labels) in enumerate(tqdm(loader, desc="Train")):
        imgs, labels = imgs.to(device, memory_format=torch.channels_last), labels.to(device)

        # ENSURE labels are Long type
        labels = labels.long()

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # ADD DETAILED LOGGING FOR FIRST FEW BATCHES
            if i < 5 or (i % 100 == 0 and i > 0):
                print(f"Batch {i}:")
                print(f"Loss: {loss.item():.6f}")
                print(f"Output range: {outputs.min().item():.3f} to {outputs.max().item():.3f}")
                print(f"Output mean: {outputs.mean().item():.3f}")
                unique_preds = torch.unique(outputs.argmax(1))
                print(f"Unique predictions: {unique_preds.cpu().numpy()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # ADD GRADIENT MONITORING
        if i < 5 or (i % 100 == 0 and i > 0):
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Gradient norm: {total_norm:.6f}")

            # Check if gradients are actually flowing
            if total_norm < 1e-8:
                print("WARNING: Gradients are too small!")
            elif total_norm > 100:
                print("WARNING: Gradients are exploding!")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        # Monitor prediction distribution
        if i % 100 == 0 and i > 0:
            pred_dist = torch.bincount(outputs.argmax(1), minlength=8)
            emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            pred_counts = pred_dist.cpu().numpy()
            print(f"   Prediction distribution: {dict(zip(emotions, pred_counts))}")

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val"):
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.long()

            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    # Show validation prediction distribution
    pred_dist = torch.bincount(torch.tensor(all_preds), minlength=8)
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    print(f"Val predictions: {dict(zip(emotions, pred_dist.numpy()))}")

    return total_loss / len(loader), correct / total


# ------------------ Main (ENHANCED) ------------------
def main():
    print("Starting Emotion Classification Training")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set, val_set, test_set, num_classes = setup_data()

    label_counts = train_set.data['label_idx'].value_counts().sort_index()
    weights = 1.0 / label_counts
    sample_weights = train_set.data['label_idx'].map(weights).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)

    train_loader = DataLoader(train_set, batch_size=48, sampler=sampler,
                              num_workers=6, pin_memory=True, prefetch_factor=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=48, shuffle=False,
                            num_workers=6, pin_memory=True, prefetch_factor=2)

    model = create_model(num_classes, device)

    # DEBUGGING: Higher learning rate and standard CrossEntropy
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=4, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    stopper = EarlyStopping(patience=10)

    print(f"   Training Configuration:")
    print(f"   Learning Rate: 5e-4")
    print(f"   Loss Function: CrossEntropyLoss (standard)")
    print(f"   Model: SimpleEmotionClassifier")
    print(f"   MTCNN: Disabled")
    print(f"   Batch Size: 48")
    print(f"   Classes: {num_classes}")
    print("=" * 60)

    for epoch in range(1, 41):
        print(f"\nEpoch {epoch}/40")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        # Update history and check early stopping
        stopper.update_history(train_loss, val_loss, train_acc, val_acc)
        stopper(val_loss, val_acc, model)

        if stopper.early_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {stopper.best_score:.4f}")
            break

    # Load best model
    if os.path.exists(stopper.save_path):
        model.load_state_dict(torch.load(stopper.save_path, map_location=device))
        print(f"Loaded best model from: {stopper.save_path}")

    # Save complete training history
    os.makedirs('models', exist_ok=True)
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(stopper.history, f)

    print("\nTraining Complete!")
    print("=" * 60)
    print("Files saved:")
    print("- models/best_emotion_model.pth (best model)")
    print("- models/training_history.pkl (complete history)")
    print(f"Final best validation accuracy: {stopper.best_score:.4f}")


if __name__ == '__main__':
    main()