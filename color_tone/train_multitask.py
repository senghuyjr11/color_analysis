import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pickle
import warnings

warnings.filterwarnings('ignore')

class EarlyStopping:
    def __init__(self, patience, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class MultiTaskColorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.parse_labels()
        print(f"Dataset loaded with {len(self.annotations)} samples")
        print(f"Season classes: {self.season_classes}")
        print(f"Tone classes: {self.tone_classes}")

    def parse_labels(self):
        seasons = []
        tones = []

        for label in self.annotations['label']:
            parts = label.split('_')
            season = parts[0]
            tone = parts[1] if len(parts) > 1 else 'unknown'
            seasons.append(season)
            tones.append(tone)

        self.annotations['season'] = seasons
        self.annotations['tone'] = tones

        self.season_classes = sorted(list(set(seasons)))
        self.tone_classes = sorted(list(set(tones)))

        self.season_to_idx = {s: i for i, s in enumerate(self.season_classes)}
        self.tone_to_idx = {t: i for i, t in enumerate(self.tone_classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['path'])
        image = Image.open(img_path).convert('RGB')

        season = self.annotations.iloc[idx]['season']
        tone = self.annotations.iloc[idx]['tone']
        season_idx = self.season_to_idx[season]
        tone_idx = self.tone_to_idx[tone]

        if self.transform:
            image = self.transform(image)

        return image, season_idx, tone_idx


class MultiTaskColorModel(nn.Module):
    def __init__(self, num_season_classes, num_tone_classes, model_name, pretrained):
        super(MultiTaskColorModel, self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        print(f"Successfully loaded model: {model_name}")

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Shared feature extractor with moderate regularization (was too aggressive)
        self.shared_features = nn.Sequential(
            nn.Dropout(0.4),  # Reduced from 0.7
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Reduced from 0.8
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)  # Reduced from 0.6
        )

        # Task-specific heads with lighter regularization
        self.season_classifier = nn.Sequential(
            nn.Dropout(0.3),  # Reduced from 0.5
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced from 0.4
            nn.Linear(64, num_season_classes)
        )

        self.tone_classifier = nn.Sequential(
            nn.Dropout(0.3),  # Reduced from 0.5
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced from 0.4
            nn.Linear(64, num_tone_classes)
        )

    def forward(self, x):
        backbone_features = self.backbone(x)
        shared_features = self.shared_features(backbone_features)

        season_logits = self.season_classifier(shared_features)
        tone_logits = self.tone_classifier(shared_features)

        return season_logits, tone_logits


class MultiTaskLoss(nn.Module):
    def __init__(self, season_weights=None, tone_weights=None):
        super(MultiTaskLoss, self).__init__()
        # Reduce label smoothing and use stronger class weights for summer issue
        self.season_criterion = nn.CrossEntropyLoss(weight=season_weights, label_smoothing=0.05)
        self.tone_criterion = nn.CrossEntropyLoss(weight=tone_weights, label_smoothing=0.05)

    def forward(self, season_pred, tone_pred, season_true, tone_true):
        season_loss = self.season_criterion(season_pred, season_true)
        tone_loss = self.tone_criterion(tone_pred, tone_true)
        # Equal weighting to focus on season class balance first
        total_loss = season_loss + tone_loss
        return total_loss, season_loss, tone_loss


def get_transforms(image_size=240):  # Default to 240 for B1
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # Less aggressive crop
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Reduced rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Gentler color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Less aggressive
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Reduced erasing
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def calculate_class_weights(dataset, device):
    season_labels = []
    tone_labels = []

    original_transform = dataset.transform
    dataset.transform = None

    try:
        for i in range(len(dataset)):
            _, season_idx, tone_idx = dataset[i]
            season_labels.append(season_idx)
            tone_labels.append(tone_idx)
    finally:
        dataset.transform = original_transform

    season_weights = compute_class_weight('balanced', classes=np.unique(season_labels), y=season_labels)
    tone_weights = compute_class_weight('balanced', classes=np.unique(tone_labels), y=tone_labels)

    return torch.FloatTensor(season_weights).to(device), torch.FloatTensor(tone_weights).to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    season_correct = 0
    tone_correct = 0
    combined_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, season_labels, tone_labels in pbar:
        images = images.to(device)
        season_labels = season_labels.to(device)
        tone_labels = tone_labels.to(device)

        optimizer.zero_grad()

        season_pred, tone_pred = model(images)
        loss, season_loss, tone_loss = criterion(season_pred, tone_pred, season_labels, tone_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, season_predicted = torch.max(season_pred, 1)
        _, tone_predicted = torch.max(tone_pred, 1)

        season_correct += (season_predicted == season_labels).sum().item()
        tone_correct += (tone_predicted == tone_labels).sum().item()
        combined_correct += ((season_predicted == season_labels) & (tone_predicted == tone_labels)).sum().item()
        total_samples += season_labels.size(0)

        combined_acc = 100 * combined_correct / total_samples
        pbar.set_postfix({'Loss': f'{running_loss / (pbar.n + 1):.4f}', 'Acc': f'{combined_acc:.1f}%'})

    epoch_loss = running_loss / len(train_loader)
    season_acc = 100 * season_correct / total_samples
    tone_acc = 100 * tone_correct / total_samples
    combined_acc = 100 * combined_correct / total_samples

    return epoch_loss, season_acc, tone_acc, combined_acc


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    season_correct = 0
    tone_correct = 0
    combined_correct = 0
    total_samples = 0

    all_season_preds = []
    all_tone_preds = []
    all_season_labels = []
    all_tone_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, season_labels, tone_labels in pbar:
            images = images.to(device)
            season_labels = season_labels.to(device)
            tone_labels = tone_labels.to(device)

            season_pred, tone_pred = model(images)
            loss, season_loss, tone_loss = criterion(season_pred, tone_pred, season_labels, tone_labels)

            running_loss += loss.item()
            _, season_predicted = torch.max(season_pred, 1)
            _, tone_predicted = torch.max(tone_pred, 1)

            season_correct += (season_predicted == season_labels).sum().item()
            tone_correct += (tone_predicted == tone_labels).sum().item()
            combined_correct += ((season_predicted == season_labels) & (tone_predicted == tone_labels)).sum().item()
            total_samples += season_labels.size(0)

            all_season_preds.extend(season_predicted.cpu().numpy())
            all_tone_preds.extend(tone_predicted.cpu().numpy())
            all_season_labels.extend(season_labels.cpu().numpy())
            all_tone_labels.extend(tone_labels.cpu().numpy())

            combined_acc = 100 * combined_correct / total_samples
            pbar.set_postfix({'Loss': f'{running_loss / (pbar.n + 1):.4f}', 'Acc': f'{combined_acc:.1f}%'})

    epoch_loss = running_loss / len(val_loader)
    season_acc = 100 * season_correct / total_samples
    tone_acc = 100 * tone_correct / total_samples
    combined_acc = 100 * combined_correct / total_samples

    return epoch_loss, season_acc, tone_acc, combined_acc, all_season_preds, all_tone_preds, all_season_labels, all_tone_labels


def save_training_history(history, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)


def main():
    config = {
        'data_root': '',
        'train_csv': 'ORIGINAL_RGB_NOT_PROCESSED/train.csv',
        'val_csv': 'ORIGINAL_RGB_NOT_PROCESSED/val.csv',
        'test_csv': 'ORIGINAL_RGB_NOT_PROCESSED/test.csv',
        'model_name': 'efficientnet_b1',
        'image_size': 240,
        'batch_size': 24,  # Increase batch size back
        'num_epochs': 60,
        'learning_rate': 1e-3,  # Much higher learning rate
        'weight_decay': 3e-4,  # Reduce weight decay
        'early_stopping_patience': 5,
        'save_dir': 'model_checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    config['save_path'] = os.path.join(config['save_dir'], 'best_multitask_model.pth')
    config['history_path'] = os.path.join(config['save_dir'], 'training_history.pkl')

    print(f"Device: {config['device']}")
    print(f"Model: {config['model_name']} (240x240 resolution)")
    print(f"Save directory: {config['save_dir']}")

    train_transform, val_transform = get_transforms(config['image_size'])

    # Create datasets
    train_dataset = MultiTaskColorDataset(
        csv_file=config['train_csv'],
        root_dir=config['data_root'],
        transform=train_transform
    )

    val_dataset = MultiTaskColorDataset(
        csv_file=config['val_csv'],
        root_dir=config['data_root'],
        transform=val_transform
    )

    # Calculate class weights
    season_weights, tone_weights = calculate_class_weights(train_dataset, config['device'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model = MultiTaskColorModel(
        num_season_classes=len(train_dataset.season_classes),
        num_tone_classes=len(train_dataset.tone_classes),
        model_name=config['model_name'],
        pretrained=True
    ).to(config['device'])

    print(f"Season classes: {len(train_dataset.season_classes)}")
    print(f"Tone classes: {len(train_dataset.tone_classes)}")

    # Loss function and optimizer with higher learning rates to escape local minimum
    criterion = MultiTaskLoss(season_weights=season_weights, tone_weights=tone_weights)

    # Higher learning rates to help model learn all classes properly
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['learning_rate'] * 0.3},  # Higher backbone LR
        {'params': list(model.shared_features.parameters()) +
                   list(model.season_classifier.parameters()) +
                   list(model.tone_classifier.parameters()), 'lr': config['learning_rate']}  # Full LR for new layers
    ], weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    print(f"Class weights being used:")
    print(f"Season weights: {season_weights}")
    print(f"Tone weights: {tone_weights}")

    # Training history
    training_history = {
        'train_losses': [], 'val_losses': [],
        'train_season_accs': [], 'train_tone_accs': [], 'train_combined_accs': [],
        'val_season_accs': [], 'val_tone_accs': [], 'val_combined_accs': [],
        'epochs_completed': 0, 'best_epoch': 0, 'early_stopped': False
    }

    best_combined_acc = 0.0

    print("\nStarting training...")
    print("=" * 50)

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Train
        train_loss, train_season_acc, train_tone_acc, train_combined_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )

        # Validate
        val_loss, val_season_acc, val_tone_acc, val_combined_acc, val_season_preds, val_tone_preds, val_season_labels, val_tone_labels = validate_epoch(
            model, val_loader, criterion, config['device']
        )

        scheduler.step()

        # Store metrics
        training_history['train_losses'].append(train_loss)
        training_history['val_losses'].append(val_loss)
        training_history['train_season_accs'].append(train_season_acc)
        training_history['train_tone_accs'].append(train_tone_acc)
        training_history['train_combined_accs'].append(train_combined_acc)
        training_history['val_season_accs'].append(val_season_acc)
        training_history['val_tone_accs'].append(val_tone_acc)
        training_history['val_combined_accs'].append(val_combined_acc)
        training_history['epochs_completed'] = epoch + 1

        # Print results with overfitting monitoring
        overfitting_gap = train_combined_acc - val_combined_acc
        print(
            f"Train: Loss={train_loss:.4f} | Season={train_season_acc:.1f}% | Tone={train_tone_acc:.1f}% | Combined={train_combined_acc:.1f}%")
        print(
            f"Val:   Loss={val_loss:.4f} | Season={val_season_acc:.1f}% | Tone={val_tone_acc:.1f}% | Combined={val_combined_acc:.1f}%")
        print(f"Overfitting gap: {overfitting_gap:.1f}% (target: <8%)")

        # Check for class collapse (all predictions same class)
        unique_season_preds = len(set(val_season_preds))
        unique_tone_preds = len(set(val_tone_preds))
        if unique_season_preds == 1:
            print(f"WARNING: Season predictions collapsed to single class!")
        if unique_tone_preds == 1:
            print(f"WARNING: Tone predictions collapsed to single class!")

        # Show prediction distribution and check for summer class issue
        season_pred_dist = {train_dataset.season_classes[i]: val_season_preds.count(i) for i in
                            range(len(train_dataset.season_classes))}
        print(f"Season pred distribution: {season_pred_dist}")

        # Check individual season accuracies
        from collections import Counter
        season_true_dist = Counter([train_dataset.season_classes[i] for i in val_season_labels])
        print(f"Season true distribution: {dict(season_true_dist)}")

        # Calculate per-class precision for seasons
        for i, season in enumerate(train_dataset.season_classes):
            season_mask = [True if val_season_labels[j] == i else False for j in range(len(val_season_labels))]
            if any(season_mask):
                season_acc = sum([1 for j, mask in enumerate(season_mask) if mask and val_season_preds[j] == i]) / sum(
                    season_mask)
                print(f"{season} accuracy: {season_acc:.1%}")

        # Save best model
        if val_combined_acc > best_combined_acc:
            best_combined_acc = val_combined_acc
            training_history['best_epoch'] = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_combined_acc': best_combined_acc,
                'season_classes': train_dataset.season_classes,
                'tone_classes': train_dataset.tone_classes,
                'season_to_idx': train_dataset.season_to_idx,
                'tone_to_idx': train_dataset.tone_to_idx,
                'config': config
            }, config['save_path'])

            print(f"NEW BEST MODEL! Combined Acc: {best_combined_acc:.1f}%")

        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping after {config['early_stopping_patience']} epochs without improvement")
            training_history['early_stopped'] = True
            break

        save_training_history(training_history, config['history_path'])

    print(f"\nTraining completed!")
    print(f"Best combined accuracy: {best_combined_acc:.1f}% (Epoch {training_history['best_epoch']})")
    print(f"Files saved in: {config['save_dir']}")

    # Final classification reports using actual predictions
    print(f"\nFinal Season Classification Report:")
    print(classification_report(val_season_labels, val_season_preds, target_names=train_dataset.season_classes,
                                zero_division=0))

    print(f"\nFinal Tone Classification Report:")
    print(classification_report(val_tone_labels, val_tone_preds, target_names=train_dataset.tone_classes,
                                zero_division=0))

    # Test evaluation if available
    if os.path.exists(config['test_csv']):
        print(f"\nEvaluating on test set...")
        test_dataset = MultiTaskColorDataset(config['test_csv'], config['data_root'], val_transform)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

        checkpoint = torch.load(config['save_path'])
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_season_acc, test_tone_acc, test_combined_acc, _, _, _, _ = validate_epoch(
            model, test_loader, criterion, config['device']
        )

        print(f"Test results:")
        print(f"Season: {test_season_acc:.1f}% | Tone: {test_tone_acc:.1f}% | Combined: {test_combined_acc:.1f}%")

if __name__ == "__main__":
    main()
