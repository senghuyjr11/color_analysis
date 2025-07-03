# COMPREHENSIVE FIX FOR EMOTION RECOGNITION
# Addresses both MTCNN issues and fundamental accuracy problems

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm.auto import tqdm
from facenet_pytorch import MTCNN
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np


# ------------------ FIXED Dataset with Better MTCNN Handling ------------------
class RobustEmotionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, use_mtcnn=True, debug=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.debug = debug
        self.use_mtcnn = use_mtcnn

        # Initialize MTCNN with more lenient settings
        if self.use_mtcnn:
            self.mtcnn = MTCNN(
                image_size=224,
                margin=40,  # More margin for context
                min_face_size=30,  # Smaller minimum face size
                thresholds=[0.5, 0.6, 0.6],  # More lenient thresholds
                factor=0.8,  # Better scale factor
                post_process=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.mtcnn = None

        unique_labels = sorted(self.data["label"].dropna().unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        print(f"[{os.path.basename(csv_file)}] {self.num_classes} classes: {unique_labels}")

        self.data["label_idx"] = self.data["label"].map(self.label_map)
        self.data = self.data.dropna(subset=["label_idx"]).copy()
        self.data["label_idx"] = self.data["label_idx"].astype('int64')

        # Track statistics
        self.mtcnn_success = 0
        self.mtcnn_failure = 0
        self.load_failure = 0

    def enhance_image_quality(self, image):
        """Enhance image quality before processing"""
        # Enhance contrast and brightness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["label"], row["filename"])

        try:
            image = Image.open(img_path).convert("RGB")

            # Enhance image quality
            image = self.enhance_image_quality(image)

            # Ensure minimum size
            if image.size[0] < 100 or image.size[1] < 100:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

            if self.mtcnn:
                try:
                    face = self.mtcnn(image)
                    if face is not None:
                        # Success: Convert tensor back to PIL
                        image = transforms.ToPILImage()(face)
                        self.mtcnn_success += 1

                        if self.debug and idx < 10:
                            print(f"Sample {idx}: MTCNN success")
                    else:
                        # MTCNN failed: Use original image with smart cropping
                        self.mtcnn_failure += 1
                        if self.debug and idx < 10:
                            print(f"Sample {idx}: MTCNN failed, using smart crop")

                        # Smart center crop focusing on upper portion (where faces usually are)
                        w, h = image.size
                        crop_size = min(w, h)

                        # Focus on upper-center area
                        left = (w - crop_size) // 2
                        top = max(0, (h - crop_size) // 3)  # Upper third

                        image = image.crop((left, top, left + crop_size, top + crop_size))
                        image = image.resize((224, 224), Image.Resampling.LANCZOS)

                except Exception as e:
                    if self.debug:
                        print(f"MTCNN error for {img_path}: {e}")
                    self.mtcnn_failure += 1
                    # Fall back to center crop
                    w, h = image.size
                    crop_size = min(w, h)
                    left = (w - crop_size) // 2
                    top = (h - crop_size) // 2
                    image = image.crop((left, top, left + crop_size, top + crop_size))
                    image = image.resize((224, 224), Image.Resampling.LANCZOS)
            else:
                # No MTCNN: Smart center crop
                w, h = image.size
                crop_size = min(w, h)
                left = (w - crop_size) // 2
                top = (h - crop_size) // 2
                image = image.crop((left, top, left + crop_size, top + crop_size))
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

        except Exception as e:
            if self.debug:
                print(f"[ERROR] Failed to load {img_path}: {e}")
            self.load_failure += 1

            # Create a neutral gray image instead of black
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, row["label_idx"]

    def print_stats(self):
        """Print processing statistics"""
        total = len(self.data)
        if self.use_mtcnn:
            print(f"MTCNN Stats: Success={self.mtcnn_success}, Failure={self.mtcnn_failure}")
            print(f"Success rate: {self.mtcnn_success / (self.mtcnn_success + self.mtcnn_failure) * 100:.1f}%")
        print(f"Load failures: {self.load_failure}/{total}")


# ------------------ Enhanced Transforms ------------------
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def setup_data(use_mtcnn=True, debug=False):
    base_path = "archive/data_balanced_1x"
    train_csv = os.path.join(base_path, "train_split.csv")
    val_csv = os.path.join(base_path, "val_split.csv")
    test_csv = os.path.join(base_path, "test.csv")
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")

    for file_path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")

    train_set = RobustEmotionDataset(train_csv, train_dir, transform=train_transform,
                                     use_mtcnn=use_mtcnn, debug=debug)
    val_set = RobustEmotionDataset(val_csv, train_dir, transform=val_transform,
                                   use_mtcnn=use_mtcnn, debug=debug)
    test_set = RobustEmotionDataset(test_csv, test_dir, transform=val_transform,
                                    use_mtcnn=use_mtcnn, debug=debug)

    num_classes = train_set.num_classes
    return train_set, val_set, test_set, num_classes


# ------------------ Better Model Architecture ------------------
class RobustEmotionClassifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)


def create_robust_model(num_classes, device):
    print("🔧 Creating robust emotion model...")

    # Use ResNet18 for faster training and less overfitting
    from torchvision.models import resnet18, ResNet18_Weights
    base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    in_features = base.fc.in_features
    base.fc = RobustEmotionClassifier(in_features, num_classes)

    model = base.to(device)

    # Test forward pass
    model.eval()
    test_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(test_input)
        print(f"✅ Model test successful: {output.shape}")
    model.train()

    return model


# ------------------ Improved Training Loop ------------------
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, save_path='models/emotion_model_v2.pth'):
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
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for i, (imgs, labels) in enumerate(tqdm(loader, desc="Train")):
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.long()

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
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

        # Monitor prediction diversity
        if i % 200 == 0 and i > 0:
            pred_classes = torch.unique(outputs.argmax(1))
            print(f"📊 Batch {i}: Predicting {len(pred_classes)}/8 classes")

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

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

    return total_loss / len(loader), correct / total


# ------------------ Main Function ------------------
def main():
    print("🚀 COMPREHENSIVE EMOTION RECOGNITION FIX")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # STEP 1: Try without MTCNN first
    print("\n🧪 STEP 1: Testing without MTCNN...")
    try:
        train_set, val_set, test_set, num_classes = setup_data(use_mtcnn=False, debug=True)
        print("✅ Data loading successful without MTCNN")

        # Print a few samples for debugging
        for i in range(3):
            img, label = train_set[i]
            print(f"Sample {i}: shape={img.shape}, label={label}")

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return

    # STEP 2: Test with MTCNN
    print("\n🧪 STEP 2: Testing with improved MTCNN...")
    try:
        train_set_mtcnn, val_set_mtcnn, test_set_mtcnn, _ = setup_data(use_mtcnn=True, debug=True)
        print("✅ Data loading successful with MTCNN")

        # Check MTCNN success rate after loading a few samples
        for i in range(20):
            _ = train_set_mtcnn[i]
        train_set_mtcnn.print_stats()

        # Use MTCNN version if success rate is good
        if train_set_mtcnn.mtcnn_success > train_set_mtcnn.mtcnn_failure:
            print("🎯 Using MTCNN version (good success rate)")
            train_set, val_set, test_set = train_set_mtcnn, val_set_mtcnn, test_set_mtcnn
        else:
            print("⚠️  Using non-MTCNN version (low success rate)")

    except Exception as e:
        print(f"⚠️  MTCNN failed: {e}, using non-MTCNN version")

    # STEP 3: Setup training
    print(f"\n🎯 STEP 3: Setting up training...")

    # Balanced sampling
    label_counts = train_set.data['label_idx'].value_counts().sort_index()
    weights = 1.0 / (label_counts + 1)  # Add 1 to avoid division by zero
    sample_weights = train_set.data['label_idx'].map(weights).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)

    train_loader = DataLoader(train_set, batch_size=32, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = create_robust_model(num_classes, device)

    # Robust training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    stopper = EarlyStopping(patience=12)

    print(f"📋 Configuration:")
    print(f"   Model: ResNet18 + RobustEmotionClassifier")
    print(f"   Optimizer: AdamW(lr=2e-4)")
    print(f"   Loss: CrossEntropyLoss with label smoothing")
    print(f"   Batch size: 32")
    print(f"   Expected accuracy: 60-75%")

    # STEP 4: Training
    print(f"\n🚀 STEP 4: Starting training...")

    for epoch in range(1, 31):
        print(f"\n🔄 Epoch {epoch}/30")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"📈 Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"📊 Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"🎯 LR: {current_lr:.2e}")

        stopper.update_history(train_loss, val_loss, train_acc, val_acc)
        stopper(val_loss, val_acc, model)

        if stopper.early_stop:
            print(f"\n⏹️ Early stopping at epoch {epoch}")
            break

    # Save results
    os.makedirs('models', exist_ok=True)
    with open('models/training_history_v2.pkl', 'wb') as f:
        pickle.dump(stopper.history, f)

    print(f"\n🎉 Training Complete!")
    print(f"🏆 Best validation accuracy: {stopper.best_score:.4f}")
    print(f"🎯 Target achieved: {'✅' if stopper.best_score > 0.6 else '❌'}")


if __name__ == '__main__':
    main()
