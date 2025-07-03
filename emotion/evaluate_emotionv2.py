import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from pathlib import Path


class FastEmotionDataset(Dataset):
    """Fast dataset that loads pre-processed images - matches training exactly"""

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        unique_labels = sorted(self.data["label"].dropna().unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.class_names = unique_labels
        self.num_classes = len(unique_labels)

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
            # Safety check for consistent size
            if image.size != (224, 224):
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
        except Exception as e:
            # Create fallback image
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, row["label_idx"]


# IMPORTANT: Use the same model architecture as training
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


def load_model_smart(model_path, num_classes, device):
    """Load the trained model with correct architecture"""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create model with same architecture as training
    model = CustomEfficientNet(num_classes)
    model = model.to(device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    return model


def get_tta_transforms():
    """Create multiple transforms for test-time augmentation"""
    base_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms_list = [
        # Original
        transforms.Compose([
            transforms.ToTensor(),
            base_normalize
        ]),

        # Horizontal flip
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            base_normalize
        ]),

        # Slight rotation
        transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            base_normalize
        ]),

        # Color jitter
        transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            base_normalize
        ]),

        # Center crop
        transforms.Compose([
            transforms.CenterCrop(200),
            transforms.Resize(224),
            transforms.ToTensor(),
            base_normalize
        ]),
    ]

    return transforms_list


def test_time_augmentation_prediction(model, image, device, num_augmentations=5):
    """Apply test-time augmentation and average predictions"""
    tta_transforms = get_tta_transforms()

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(min(num_augmentations, len(tta_transforms))):
            # Apply transform
            transformed_image = tta_transforms[i](image).unsqueeze(0).to(device)

            # Get prediction
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(transformed_image)
                prob = F.softmax(output, dim=1)
                predictions.append(prob.cpu())

    # Average all predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction


def calibrate_predictions(y_probs, temperature=1.3):
    """Apply temperature scaling to calibrate confidence"""
    # Temperature scaling
    scaled_probs = np.exp(np.log(y_probs + 1e-8) / temperature)
    scaled_probs = scaled_probs / np.sum(scaled_probs, axis=1, keepdims=True)
    return scaled_probs


def evaluate_model_baseline(model, test_loader, device, class_names):
    """Standard model evaluation (baseline)"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("Running baseline evaluation...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Baseline"):
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.long()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_model_improved(model, dataset, device, class_names):
    """Improved evaluation with TTA and temperature scaling"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("Running improved evaluation with TTA...")

    for idx in tqdm(range(len(dataset)), desc="TTA Evaluation"):
        # Get raw image (before transform)
        row = dataset.data.iloc[idx]
        img_path = dataset.img_dir / row["label"] / row["filename"]

        try:
            image = Image.open(img_path).convert("RGB")
            if image.size != (224, 224):
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
        except:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        label = row["label_idx"]

        # Use test-time augmentation
        prob = test_time_augmentation_prediction(model, image, device)
        prob_numpy = prob.squeeze().numpy()

        all_labels.append(label)
        all_probs.append(prob_numpy)

    # Apply temperature scaling
    print("Applying temperature scaling...")
    all_probs = np.array(all_probs)
    calibrated_probs = calibrate_predictions(all_probs, temperature=1.3)
    all_preds = np.argmax(calibrated_probs, axis=1)

    return all_preds, np.array(all_labels), calibrated_probs


def print_metrics(y_true, y_pred, class_names, method_name=""):
    """Print detailed classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 60}")
    print(f"{method_name} ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 60}")

    # Per-class accuracy
    print(f"\nPER-CLASS ACCURACY:")
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).mean()
            print(f"   {class_name:>10}: {class_acc:.4f} ({class_acc * 100:.1f}%)")

    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def print_improvement_analysis(y_true, y_pred_baseline, y_pred_improved, class_names):
    """Compare baseline vs improved results"""
    baseline_acc = accuracy_score(y_true, y_pred_baseline)
    improved_acc = accuracy_score(y_true, y_pred_improved)
    improvement = improved_acc - baseline_acc

    print(f"\n{'=' * 60}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Baseline accuracy:  {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")
    print(f"Improved accuracy:  {improved_acc:.4f} ({improved_acc * 100:.2f}%)")
    print(f"Improvement:        {improvement:+.4f} ({improvement * 100:+.2f} percentage points)")

    # Per-class improvement
    print(f"\nPER-CLASS IMPROVEMENTS:")
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            baseline_class_acc = (y_pred_baseline[class_mask] == i).mean()
            improved_class_acc = (y_pred_improved[class_mask] == i).mean()
            class_improvement = improved_class_acc - baseline_class_acc
            print(f"   {class_name:>10}: {class_improvement:+.4f} ({class_improvement * 100:+.1f}%)")

    return improvement


def plot_comparison_confusion_matrix(y_true, y_pred_baseline, y_pred_improved, class_names,
                                     save_path='results/comparison_confusion_matrix.png'):
    """Plot side-by-side confusion matrices for comparison"""
    cm_baseline = confusion_matrix(y_true, y_pred_baseline)
    cm_improved = confusion_matrix(y_true, y_pred_improved)

    # Normalize
    cm_baseline_norm = cm_baseline.astype('float') / cm_baseline.sum(axis=1)[:, np.newaxis]
    cm_improved_norm = cm_improved.astype('float') / cm_improved.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    # Baseline
    sns.heatmap(cm_baseline_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Baseline Model', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # Improved
    sns.heatmap(cm_improved_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Improved Model (TTA + Calibration)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    # Difference (improvement)
    cm_diff = cm_improved_norm - cm_baseline_norm
    sns.heatmap(cm_diff, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Improvement (Green - Blue)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison confusion matrix saved to: {save_path}")
    plt.show()


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
    print(f"Confusion matrix saved to: {save_path}")
    plt.show()


def analyze_predictions(y_true, y_pred, y_probs, class_names):
    """Analyze prediction patterns and confidence"""
    correct_mask = (y_true == y_pred)
    max_probs = np.max(y_probs, axis=1)

    print(f"\n{'=' * 60}")
    print("PREDICTION ANALYSIS:")
    print(f"{'=' * 60}")
    print(f"Correct predictions: {correct_mask.sum()}/{len(y_true)} ({correct_mask.mean():.4f})")
    print(f"Average confidence (correct): {max_probs[correct_mask].mean():.4f}")
    print(f"Average confidence (incorrect): {max_probs[~correct_mask].mean():.4f}")

    # Confidence distribution
    print(f"\nCONFIDENCE DISTRIBUTION:")
    confidence_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in confidence_ranges:
        mask = (max_probs >= low) & (max_probs < high)
        accuracy_in_range = correct_mask[mask].mean() if mask.sum() > 0 else 0
        print(f"   {low:.1f}-{high:.1f}: {mask.sum():4d} samples, accuracy: {accuracy_in_range:.3f}")

    # Most confused classes
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # Remove correct predictions
    most_confused = np.unravel_index(cm.argmax(), cm.shape)
    print(f"\nMost confused classes:")
    print(f"   {class_names[most_confused[0]]} -> {class_names[most_confused[1]]}: {cm[most_confused]} times")

    # Find other highly confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((cm[i, j], class_names[i], class_names[j]))

    confused_pairs.sort(reverse=True)
    print(f"   Top confusion pairs:")
    for count, true_class, pred_class in confused_pairs[:3]:
        print(f"     {true_class} -> {pred_class}: {count} times")


def plot_training_history(history_path='models/training_history_efficientnet.pkl',
                          save_path='results/training_curves.png'):
    """Plot comprehensive training history - FIXED PATH"""
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)

        print(f"\nTRAINING HISTORY ANALYSIS:")

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

            print(f"Training Summary:")
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

            print(f"Training Summary (Loss only):")
            print(f"   Total epochs: {len(val_losses)}")
            print(f"   Best validation loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")
            print(f"   Final validation loss: {val_losses[-1]:.4f}")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        plt.show()

    except FileNotFoundError:
        print(f"Training history not found at {history_path}")
    except Exception as e:
        print(f"Error loading training history: {e}")


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
    print(f"Prediction samples saved to: {save_path}")
    plt.show()


def setup_test_data(preprocessed_path="archive/preprocessed_faces"):
    """Setup test dataset from preprocessed images"""
    base_path = Path(preprocessed_path)

    # Use the same validation split as training for testing
    test_csv = base_path / "val_split.csv"  # FIXED: Use val_split as test set
    test_dir = base_path / "val_split"  # FIXED: Use val_split as test set

    # Use the same transform as training validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_set = FastEmotionDataset(test_csv, test_dir, transform=val_transform)
    test_set_raw = FastEmotionDataset(test_csv, test_dir, transform=None)  # For TTA
    return test_set, test_set_raw


def main():
    print("COMPREHENSIVE EMOTION MODEL EVALUATION")
    print("Includes baseline + improved (TTA + calibration) methods")
    print("=" * 60)

    # Create results directory first
    os.makedirs('results', exist_ok=True)
    print("Created results directory")

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for preprocessed data
    preprocessed_path = "archive/preprocessed_faces"
    if not Path(preprocessed_path).exists():
        print(f"Error: Preprocessed data not found at {preprocessed_path}")
        print("Please run the training script first to create preprocessed data.")
        return

    # Setup test data
    print(f"\nLoading test data from: {preprocessed_path}")
    test_set, test_set_raw = setup_test_data(preprocessed_path)

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Test set: {len(test_set)} samples, {test_set.num_classes} classes")
    print(f"Classes: {test_set.class_names}")

    # Load model - FIXED PATH
    model_path = 'models/emotion_model_efficientnet.pth'
    model = load_model_smart(model_path, test_set.num_classes, device)

    # Run evaluation
    print(f"\n{'=' * 60}")
    print("STARTING COMPREHENSIVE EVALUATION")
    print(f"{'=' * 60}")

    # 1. Baseline evaluation
    print("\n1. BASELINE EVALUATION")
    print("-" * 30)
    y_pred_baseline, y_true, y_probs_baseline = evaluate_model_baseline(model, test_loader, device,
                                                                        test_set.class_names)
    print_metrics(y_true, y_pred_baseline, test_set.class_names, "BASELINE")

    # 2. Improved evaluation
    print("\n2. IMPROVED EVALUATION (TTA + CALIBRATION)")
    print("-" * 50)
    y_pred_improved, _, y_probs_improved = evaluate_model_improved(model, test_set_raw, device, test_set.class_names)
    print_metrics(y_true, y_pred_improved, test_set.class_names, "IMPROVED")

    # 3. Comparison analysis
    improvement = print_improvement_analysis(y_true, y_pred_baseline, y_pred_improved, test_set.class_names)

    # 4. Generate visualizations
    print(f"\n3. GENERATING VISUALIZATIONS")
    print("-" * 30)

    # Comparison confusion matrices
    plot_comparison_confusion_matrix(y_true, y_pred_baseline, y_pred_improved, test_set.class_names)

    # Detailed analysis for improved model
    analyze_predictions(y_true, y_pred_improved, y_probs_improved, test_set.class_names)

    # Training history
    plot_training_history()

    # Sample predictions
    generate_prediction_samples(model, test_loader, device, test_set.class_names)

    # 5. Save comprehensive results
    results = {
        'baseline_accuracy': accuracy_score(y_true, y_pred_baseline),
        'improved_accuracy': accuracy_score(y_true, y_pred_improved),
        'improvement': improvement,
        'baseline_predictions': y_pred_baseline,
        'improved_predictions': y_pred_improved,
        'true_labels': y_true,
        'baseline_probabilities': y_probs_baseline,
        'improved_probabilities': y_probs_improved,
        'class_names': test_set.class_names,
        'confusion_matrix_baseline': confusion_matrix(y_true, y_pred_baseline),
        'confusion_matrix_improved': confusion_matrix(y_true, y_pred_improved),
        'classification_report_baseline': classification_report(y_true, y_pred_baseline,
                                                                target_names=test_set.class_names, output_dict=True),
        'classification_report_improved': classification_report(y_true, y_pred_improved,
                                                                target_names=test_set.class_names, output_dict=True)
    }

    np.save('results/comprehensive_evaluation_results.npy', results)

    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"{'=' * 60}")
    print("Files generated:")
    print("   - results/comparison_confusion_matrix.png (baseline vs improved)")
    print("   - results/training_curves.png (learning progression)")
    print("   - results/prediction_samples.png (visual predictions)")
    print("   - results/comprehensive_evaluation_results.npy (all results)")

    print(f"\nFINAL RESULTS:")
    print(
        f"Baseline Test Accuracy:  {accuracy_score(y_true, y_pred_baseline):.4f} ({accuracy_score(y_true, y_pred_baseline) * 100:.2f}%)")
    print(
        f"Improved Test Accuracy:  {accuracy_score(y_true, y_pred_improved):.4f} ({accuracy_score(y_true, y_pred_improved) * 100:.2f}%)")
    print(f"Improvement:             {improvement:+.4f} ({improvement * 100:+.2f} percentage points)")

    if improvement > 0.02:
        print("SUCCESS: Significant improvement achieved!")
    elif improvement > 0.01:
        print("GOOD: Moderate improvement achieved!")
    else:
        print("LIMITED: Small improvement. Consider retraining with stronger regularization.")


if __name__ == '__main__':
    main()
