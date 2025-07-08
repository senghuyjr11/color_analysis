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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import json
import warnings

warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
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
        """Saves model when validation loss decreases."""
        self.best_weights = model.state_dict().copy()


class ColorAnalysisDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Dataset for seasonal color analysis

        Args:
            csv_file (str): Path to CSV file with annotations
            root_dir (str): Root directory containing images
            transform (callable): Optional transforms to apply
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Create label mapping
        self.labels = sorted(self.annotations['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        print(f"Dataset loaded with {len(self.annotations)} samples")
        print(f"Classes: {self.labels}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['path'])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations.iloc[idx]['label']
        label_idx = self.label_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx


class ColorAnalysisModel(nn.Module):
    def __init__(self, num_classes, model_name='convnext_tiny', pretrained=True):
        """
        ConvNeXt-based model for color analysis

        Args:
            num_classes (int): Number of output classes
            model_name (str): ConvNeXt model variant
            pretrained (bool): Use pretrained weights
        """
        super(ColorAnalysisModel, self).__init__()

        # Load pretrained ConvNeXt
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Custom classifier for color analysis
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def get_transforms(image_size=224):
    """Get data augmentation transforms"""

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        accuracy = 100 * correct_predictions / total_samples
        pbar.set_postfix({
            'Loss': f'{running_loss / len(train_loader):.4f}',
            'Acc': f'{accuracy:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_predictions / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = 100 * correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{running_loss / len(val_loader):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct_predictions / total_samples
    return epoch_loss, epoch_acc, all_predictions, all_labels


def save_training_history(history, filepath):
    """Save training history to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {filepath}")


def main():
    # Configuration
    config = {
        'data_root': '',
        'train_csv': 'ORIGINAL_RGB_NOT_PROCESSED/train.csv',
        'val_csv': 'ORIGINAL_RGB_NOT_PROCESSED/val.csv',
        'test_csv': 'ORIGINAL_RGB_NOT_PROCESSED/test.csv',
        'model_name': 'convnext_tiny',  # Options: convnext_tiny, convnext_small, convnext_base
        'image_size': 224,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 5,
        'save_path': 'best_color_analysis_model.pth',
        'history_path': 'training_history.pkl',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Using device: {config['device']}")
    print(f"Model: {config['model_name']}")

    # Get transforms
    train_transform, val_transform = get_transforms(config['image_size'])

    # Create datasets
    train_dataset = ColorAnalysisDataset(
        csv_file=config['train_csv'],
        root_dir=config['data_root'],
        transform=train_transform
    )

    val_dataset = ColorAnalysisDataset(
        csv_file=config['val_csv'],
        root_dir=config['data_root'],
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    num_classes = len(train_dataset.labels)
    model = ColorAnalysisModel(
        num_classes=num_classes,
        model_name=config['model_name'],
        pretrained=True
    ).to(config['device'])

    print(f"Model created with {num_classes} classes")
    print(f"Classes: {train_dataset.labels}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=0.001,
        restore_best_weights=True
    )

    # Training loop
    best_val_acc = 0.0
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'learning_rates': [],
        'epochs_completed': 0,
        'best_epoch': 0,
        'early_stopped': False
    }

    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(config['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']} | LR: {current_lr:.2e}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, config['device']
        )

        # Update scheduler
        scheduler.step()

        # Store metrics
        training_history['train_losses'].append(train_loss)
        training_history['val_losses'].append(val_loss)
        training_history['train_accs'].append(train_acc)
        training_history['val_accs'].append(val_acc)
        training_history['learning_rates'].append(current_lr)
        training_history['epochs_completed'] = epoch + 1

        # Print epoch results
        print(f"Train - Loss: {train_loss:.6f} | Accuracy: {train_acc:.4f}%")
        print(f"Val   - Loss: {val_loss:.6f} | Accuracy: {val_acc:.4f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            training_history['best_epoch'] = epoch + 1

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'class_names': train_dataset.labels,
                'label_to_idx': train_dataset.label_to_idx,
                'config': config
            }, config['save_path'])

            print(f"★ NEW BEST MODEL SAVED! Val Acc: {best_val_acc:.4f}% ★")

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\n🛑 EARLY STOPPING triggered after {config['early_stopping_patience']} epochs without improvement")
            training_history['early_stopped'] = True
            break

        print(f"Best Val Acc so far: {best_val_acc:.4f}% (Epoch {training_history['best_epoch']})")

        # Save training history after each epoch
        save_training_history(training_history, config['history_path'])

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📈 Best Validation Accuracy: {best_val_acc:.4f}% (Epoch {training_history['best_epoch']})")
    print(f"📁 Best model saved to: {config['save_path']}")
    print(f"📊 Training history saved to: {config['history_path']}")

    if training_history['early_stopped']:
        print(f"⏰ Training stopped early after {training_history['epochs_completed']} epochs")
    else:
        print(f"⏰ Training completed all {training_history['epochs_completed']} epochs")

    # Final evaluation on validation set
    print(f"\n📋 FINAL VALIDATION REPORT:")
    print("-" * 50)
    print(classification_report(
        val_labels, val_preds,
        target_names=train_dataset.labels,
        digits=4
    ))

    # Test on test set if available
    if os.path.exists(config['test_csv']):
        print(f"\n📋 EVALUATING ON TEST SET:")
        print("-" * 50)
        test_dataset = ColorAnalysisDataset(
            csv_file=config['test_csv'],
            root_dir=config['data_root'],
            transform=val_transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Load best model
        checkpoint = torch.load(config['save_path'])
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_preds, test_labels = validate_epoch(
            model, test_loader, criterion, config['device']
        )

        print(f"🎯 Test Accuracy: {test_acc:.4f}%")
        print(f"📋 Test Classification Report:")
        print("-" * 50)
        print(classification_report(
            test_labels, test_preds,
            target_names=train_dataset.labels,
            digits=4
        ))

        # Add test results to history
        training_history['test_acc'] = test_acc
        training_history['test_loss'] = test_loss
        save_training_history(training_history, config['history_path'])

    print(f"\n✅ All training artifacts saved!")
    print(f"   📁 Model: {config['save_path']}")
    print(f"   📊 History: {config['history_path']}")


def predict_single_image(model_path, image_path, transform=None):
    """Predict color category for a single image"""
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    model = ColorAnalysisModel(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    predicted_class = class_names[predicted_idx]

    return predicted_class, confidence, probabilities[0].tolist()


if __name__ == "__main__":
    main()
