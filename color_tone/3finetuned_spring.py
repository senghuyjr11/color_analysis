import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from torchvision.models import convnext_large
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ======================
# CONFIG
# ======================
DATASET_DIR = "merge_dataset_cropped_segmented"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

SEASON = "spring"  # Only boosting Spring subtone
MODEL_PATH = os.path.join(DATASET_DIR, f"convnext_subtone_{SEASON}.pth")
HISTORY_PATH = os.path.join(DATASET_DIR, f"train_history_subtone_{SEASON}_boost.json")

GRAD_ACCUM_STEPS = 2
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 35
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD DATA
# ======================
df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_test = pd.read_csv(TEST_CSV)

# Split by Spring only
for df in [df_train, df_val, df_test]:
    df['season'] = df['label'].apply(lambda x: x.split('_')[0])
    df['subtone'] = df['label'].apply(lambda x: x.split('_')[1])

train_df = df_train[df_train['season'] == SEASON].reset_index(drop=True)
val_df   = df_val[df_val['season'] == SEASON].reset_index(drop=True)
test_df  = df_test[df_test['season'] == SEASON].reset_index(drop=True)

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
# AUGMENTATION (boosted for Spring)
# ======================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.ToTensor(),  # ⬅️ Must come before RandomErasing
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======================
# LABEL ENCODER & CLASS WEIGHTS
# ======================
label_encoder = LabelEncoder()
label_encoder.fit(train_df['subtone'])
num_classes = len(label_encoder.classes_)
print(f"{SEASON.upper()} Subtones:", label_encoder.classes_)

class_weights = compute_class_weight('balanced', classes=label_encoder.classes_, y=train_df['subtone'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ======================
# LOAD DATASETS
# ======================
train_dataset = SubtoneDataset(train_df, DATASET_DIR, target="subtone", transform=train_transform, label_encoder=label_encoder)
val_dataset   = SubtoneDataset(val_df, DATASET_DIR, target="subtone", transform=val_transform, label_encoder=label_encoder)
test_dataset  = SubtoneDataset(test_df, DATASET_DIR, target="subtone", transform=val_transform, label_encoder=label_encoder)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ======================
# MODEL - LOAD PREVIOUS SPRING MODEL
# ======================
model = convnext_large(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

# Load previous Spring model weights
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

# ======================
# OPTIMIZER & LOSS
# ======================
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.2)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# ======================
# MixUp Function
# ======================
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
# BOOST-PHASE TRAINING LOOP
# ======================
history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
epochs_no_improve = 0

print(f"\n🚀 Starting Spring BOOST PHASE (target: 70%+ accuracy)\n")
for epoch in range(EPOCHS):

    if epoch == 5:
        for name, param in model.named_parameters():
            if "features.6" in name or "features.7" in name:  # last two stages
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        print("🔓 Partially unfrozen backbone (stages 6 & 7) — LR dropped to 1e-5")
        
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Spring Boost] Epoch {epoch+1}/{EPOCHS}")):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / GRAD_ACCUM_STEPS  # normalize loss for accumulation
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * GRAD_ACCUM_STEPS  # restore original loss scale
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # === Validation ===
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

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Best Spring model improved & saved (val_acc: {val_acc:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print("🛑 Early stopping triggered — no improvement.")
        break
    
    scheduler.step(epoch + 1)

# ======================
# SAVE HISTORY
# ======================
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f)
print(f"\n📁 Training history saved to {HISTORY_PATH}")

# ======================
# TEST EVALUATION
# ======================
model.load_state_dict(torch.load(MODEL_PATH))
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
print(f"\n🌸 Final Test Accuracy (Spring BOOST Phase): {test_acc:.4f}")
