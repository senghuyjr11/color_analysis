import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


# ------------------ Reuse Dataset and Model Classes (FIXED) ------------------
class EmotionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mtcnn=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mtcnn = mtcnn

        unique_labels = sorted(self.data["label"].dropna().unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.class_names = unique_labels
        self.num_classes = len(unique_labels)

        self.data["label_idx"] = self.data["label"].map(self.label_map)
        self.data = self.data.dropna(subset=["label_idx"]).copy()
        # FIXED: Use int64 to match training
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


# ADDED: SimpleEmotionClassifier to match training
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


class EmotionClassifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm((in_features,), eps=1e-6),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def create_model_complex(num_classes, device):
    """Create model with complex EmotionClassifier"""
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = base.fc.in_features
    base.fc = EmotionClassifier(in_features, num_classes)
    return base.to(device)


def create_model_simple(num_classes, device):
    """Create model with SimpleEmotionClassifier"""
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = base.fc.in_features
    base.fc = SimpleEmotionClassifier(in_features, num_classes)
    return base.to(device)


def load_model_smart(model_path, num_classes, device):
    """Intelligently load model by detecting its architecture"""
    print(f"🔍 Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load state dict to inspect structure
    state_dict = torch.load(model_path, map_location=device)
    fc_keys = [k for k in state_dict.keys() if k.startswith('fc.')]
    print(f"📋 FC layer keys found: {fc_keys}")

    # Determine architecture and create matching model
    if 'fc.classifier.1.weight' in state_dict and 'fc.classifier.3.weight' in state_dict:
        print("✅ Detected: Complex EmotionClassifier architecture")
        model = create_model_complex(num_classes, device)

    elif 'fc.classifier.2.weight' in state_dict and 'fc.classifier.2.bias' in state_dict:
        print("✅ Detected: SimpleEmotionClassifier architecture")
        # This is SimpleEmotionClassifier: Flatten(0), Dropout(1), Linear(2)
        model = create_model_simple(num_classes, device)

    elif 'fc.1.weight' in state_dict and 'fc.1.bias' in state_dict:
        print("⚠️  Detected: Sequential(Flatten, Linear) architecture")
        # Recreate the simple structure
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        fc1_weight_shape = state_dict['fc.1.weight'].shape
        in_features = fc1_weight_shape[1]
        out_features = fc1_weight_shape[0]

        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, out_features)
        )
        model = model.to(device)

    elif 'fc.weight' in state_dict:
        print("⚠️  Detected: Simple Linear fc architecture")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    else:
        raise ValueError(f"Unrecognized model architecture. FC keys: {fc_keys}")

    # Load the state dict
    model.load_state_dict(state_dict)
    print(f"✅ Model loaded successfully!")

    return model


# ------------------ Evaluation Functions ------------------
def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("🧪 Running evaluation...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            # ENSURE labels are Long type
            labels = labels.long()

            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_metrics(y_true, y_pred, class_names):
    """Print detailed classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 60}")
    print(f"🎯 OVERALL ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 60}")

    # Per-class accuracy
    print(f"\n📊 PER-CLASS ACCURACY:")
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).mean()
            print(f"   {class_name:>10}: {class_acc:.4f} ({class_acc * 100:.1f}%)")

    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.show()


def analyze_predictions(y_true, y_pred, y_probs, class_names):
    """Analyze prediction patterns and confidence"""
    correct_mask = (y_true == y_pred)
    max_probs = np.max(y_probs, axis=1)

    print(f"\n{'=' * 60}")
    print("🔍 PREDICTION ANALYSIS:")
    print(f"{'=' * 60}")
    print(f"✅ Correct predictions: {correct_mask.sum()}/{len(y_true)} ({correct_mask.mean():.4f})")
    print(f"📈 Average confidence (correct): {max_probs[correct_mask].mean():.4f}")
    print(f"📉 Average confidence (incorrect): {max_probs[~correct_mask].mean():.4f}")

    # Confidence distribution
    print(f"\n📊 CONFIDENCE DISTRIBUTION:")
    confidence_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in confidence_ranges:
        mask = (max_probs >= low) & (max_probs < high)
        accuracy_in_range = correct_mask[mask].mean() if mask.sum() > 0 else 0
        print(f"   {low:.1f}-{high:.1f}: {mask.sum():4d} samples, accuracy: {accuracy_in_range:.3f}")

    # Most confused classes
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # Remove correct predictions
    most_confused = np.unravel_index(cm.argmax(), cm.shape)
    print(f"\n🤔 Most confused classes:")
    print(f"   {class_names[most_confused[0]]} → {class_names[most_confused[1]]}: {cm[most_confused]} times")

    # Find other highly confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((cm[i, j], class_names[i], class_names[j]))

    confused_pairs.sort(reverse=True)
    print(f"   Top confusion pairs:")
    for count, true_class, pred_class in confused_pairs[:3]:
        print(f"     {true_class} → {pred_class}: {count} times")


def plot_training_history(history_path='models/training_history.pkl', save_path='results/training_curves.png'):
    """Plot comprehensive training history"""
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)

        print(f"\n📈 TRAINING HISTORY ANALYSIS:")

        if isinstance(history, dict) and 'train_loss' in history:
            # Complete history with train/val loss and accuracy
            epochs = range(1, len(history['val_loss']) + 1)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Loss curves
            ax1.plot(epochs, history['train_loss'], 'r-', label='Train Loss', linewidth=2)
            ax1.plot(epochs, history['val_loss'], 'b-', label='Validation Loss', linewidth=2)
            ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Accuracy curves
            ax2.plot(epochs, history['train_acc'], 'r-', label='Train Accuracy', linewidth=2)
            ax2.plot(epochs, history['val_acc'], 'b-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Learning curve (val accuracy vs train accuracy)
            ax3.plot(history['train_acc'], history['val_acc'], 'go-', alpha=0.7)
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Perfect correlation line
            ax3.set_title('Learning Curve (Train vs Val Accuracy)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Train Accuracy')
            ax3.set_ylabel('Validation Accuracy')
            ax3.grid(True, alpha=0.3)

            # Loss difference (overfitting indicator)
            loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
            ax4.plot(epochs, loss_diff, 'purple', linewidth=2)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax4.set_title('Overfitting Indicator (Train - Val Loss)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Difference')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Print summary statistics
            best_val_acc = max(history['val_acc'])
            best_val_epoch = history['val_acc'].index(best_val_acc) + 1
            final_val_acc = history['val_acc'][-1]

            print(f"📊 Training Summary:")
            print(f"   Total epochs: {len(epochs)}")
            print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_epoch})")
            print(f"   Final validation accuracy: {final_val_acc:.4f}")
            print(f"   Improvement: {(best_val_acc - history['val_acc'][0]):.4f}")

        elif isinstance(history, list):
            # Simple validation loss history (from old format)
            val_losses = history
            epochs = range(1, len(val_losses) + 1)

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, val_losses, 'b-', label='Validation Loss', linewidth=2)
            plt.title('Training History (Validation Loss)', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            print(f"📊 Training Summary (Loss only):")
            print(f"   Total epochs: {len(val_losses)}")
            print(f"   Best validation loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")
            print(f"   Final validation loss: {val_losses[-1]:.4f}")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to: {save_path}")
        plt.show()

    except FileNotFoundError:
        print(f"⚠️  Training history not found at {history_path}")
    except Exception as e:
        print(f"❌ Error loading training history: {e}")


def generate_prediction_samples(model, test_loader, device, class_names, save_path='results/prediction_samples.png'):
    """Generate sample predictions for visual inspection"""
    model.eval()

    # Get a few samples for visualization
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images[:16])  # Take first 16 samples
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(1)

    # Convert back to CPU for plotting
    images = images[:16].cpu()
    labels = labels[:16].cpu()
    preds = preds.cpu()
    probs = probs.cpu()

    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    images = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    images = torch.clamp(images, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        ax = axes[i // 4, i % 4]

        # Convert CHW to HWC for matplotlib
        img = images[i].permute(1, 2, 0)
        ax.imshow(img)

        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence = probs[i, preds[i]].item()

        # Color code: green for correct, red for incorrect
        color = 'green' if labels[i] == preds[i] else 'red'

        ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}',
                     fontsize=10, color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️  Prediction samples saved to: {save_path}")
    plt.show()


# ------------------ Main Evaluation Function ------------------
def main():
    print("🎯 EMOTION MODEL EVALUATION (FIXED)")
    print("=" * 60)

    # CREATE RESULTS DIRECTORY FIRST
    os.makedirs('results', exist_ok=True)
    print("📁 Created results directory")

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data setup - MATCH TRAINING (NO MTCNN)
    mtcnn = None  # Disabled to match training
    base_path = "archive/data_balanced_1x"
    test_csv = os.path.join(base_path, "test.csv")
    test_dir = os.path.join(base_path, "test")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_set = EmotionDataset(test_csv, test_dir, transform=val_transform, mtcnn=mtcnn)
    test_loader = DataLoader(test_set, batch_size=48, shuffle=False,
                             num_workers=6, pin_memory=True)

    print(f"📋 Test set: {len(test_set)} samples, {test_set.num_classes} classes")
    print(f"📋 Classes: {test_set.class_names}")
    print(f"⚠️  MTCNN: {'Enabled' if mtcnn else 'Disabled'} (matching training)")

    # Load model with smart detection
    model_path = 'models/best_emotion_model.pth'
    model = load_model_smart(model_path, test_set.num_classes, device)

    # Run evaluation
    print(f"\n{'=' * 60}")
    print("🚀 STARTING EVALUATION")
    print(f"{'=' * 60}")

    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device, test_set.class_names)

    # Generate all analysis
    print_metrics(y_true, y_pred, test_set.class_names)
    plot_confusion_matrix(y_true, y_pred, test_set.class_names)
    analyze_predictions(y_true, y_pred, y_probs, test_set.class_names)
    plot_training_history()
    generate_prediction_samples(model, test_loader, device, test_set.class_names)

    # Save comprehensive results
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'predictions': y_pred,
        'true_labels': y_true,
        'probabilities': y_probs,
        'class_names': test_set.class_names,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=test_set.class_names,
                                                       output_dict=True)
    }

    np.save('results/evaluation_results.npy', results)

    print(f"\n{'=' * 60}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'=' * 60}")
    print("📁 Files generated:")
    print("   - results/confusion_matrix.png (performance heatmap)")
    print("   - results/training_curves.png (learning progression)")
    print("   - results/prediction_samples.png (visual predictions)")
    print("   - results/evaluation_results.npy (complete results)")
    print(
        f"\n🏆 Final Test Accuracy: {accuracy_score(y_true, y_pred):.4f} ({accuracy_score(y_true, y_pred) * 100:.2f}%)")


if __name__ == '__main__':
    main()
