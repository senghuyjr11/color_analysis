import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights  # FIXED: Use ResNet18
from facenet_pytorch import MTCNN
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


# ------------------ FIXED: Use EXACT same classes as training ------------------
class RobustEmotionDataset(Dataset):
    """EXACT COPY from training code"""

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
        self.class_names = unique_labels  # Added for evaluation
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


class RobustEmotionClassifier(nn.Module):
    """EXACT COPY from training code"""

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
    """EXACT COPY from training code"""
    print("🔧 Creating robust emotion model...")

    # FIXED: Use ResNet18 to match training
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


def load_model_correct(model_path, num_classes, device):
    """Load model with correct architecture matching training"""
    print(f"🔍 Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create the exact same model as training
    model = create_robust_model(num_classes, device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    # Check if keys match
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    if model_keys != loaded_keys:
        print("⚠️  Key mismatch detected!")
        print(f"Missing in loaded: {model_keys - loaded_keys}")
        print(f"Extra in loaded: {loaded_keys - model_keys}")

        # Try to handle some common naming differences
        if 'fc.classifier.0.weight' in loaded_keys and 'fc.classifier.1.weight' in loaded_keys:
            print("✅ Detected: RobustEmotionClassifier architecture")
        else:
            raise ValueError("Cannot match model architecture!")

    model.load_state_dict(state_dict)
    print(f"✅ Model loaded successfully!")

    return model


# ------------------ FIXED: Use same transforms as training ------------------
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


def plot_training_history(history_path='models/training_history_v2.pkl', save_path='results/training_curves.png'):
    """Plot training history - FIXED to use correct path"""
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)

        print(f"\n📈 TRAINING HISTORY ANALYSIS:")

        if isinstance(history, dict) and 'train_loss' in history:
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

            # Learning curve
            ax3.plot(history['train_acc'], history['val_acc'], 'go-', alpha=0.7)
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax3.set_title('Learning Curve (Train vs Val Accuracy)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Train Accuracy')
            ax3.set_ylabel('Validation Accuracy')
            ax3.grid(True, alpha=0.3)

            # Loss difference
            loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
            ax4.plot(epochs, loss_diff, 'purple', linewidth=2)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax4.set_title('Overfitting Indicator (Train - Val Loss)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Difference')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Training curves saved to: {save_path}")
            plt.show()

            # Print summary
            best_val_acc = max(history['val_acc'])
            best_val_epoch = history['val_acc'].index(best_val_acc) + 1
            print(f"📊 Training Summary:")
            print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_epoch})")

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


# ------------------ Main Evaluation Function ------------------
def main():
    print("🎯 FIXED EMOTION MODEL EVALUATION")
    print("=" * 60)

    # Create results directory
    os.makedirs('results', exist_ok=True)
    print("📁 Created results directory")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # FIXED: Match training setup exactly
    base_path = "archive/data_balanced_1x"
    test_csv = os.path.join(base_path, "test.csv")
    test_dir = os.path.join(base_path, "test")

    # STEP 1: Determine MTCNN usage (same logic as training)
    print("\n🧪 Determining MTCNN usage...")

    # Try to match training's decision logic
    use_mtcnn = False  # Default to False like training often does

    # You can override this based on your training logs:
    # If training used MTCNN successfully, set use_mtcnn = True

    print(f"🔧 Using MTCNN: {use_mtcnn} (matching training)")

    # FIXED: Use exact same dataset class and transforms
    test_set = RobustEmotionDataset(
        test_csv,
        test_dir,
        transform=val_transform,
        use_mtcnn=use_mtcnn,
        debug=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=32,  # Match training batch size
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"📋 Test set: {len(test_set)} samples, {test_set.num_classes} classes")
    print(f"📋 Classes: {test_set.class_names}")

    # FIXED: Load model with correct architecture
    model_path = 'models/emotion_model_v2.pth'  # Match training save path
    model = load_model_correct(model_path, test_set.num_classes, device)

    # Run evaluation
    print(f"\n{'=' * 60}")
    print("🚀 STARTING EVALUATION")
    print(f"{'=' * 60}")

    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device, test_set.class_names)

    # Generate all analysis and plots
    print_metrics(y_true, y_pred, test_set.class_names)
    plot_confusion_matrix(y_true, y_pred, test_set.class_names)
    analyze_predictions(y_true, y_pred, y_probs, test_set.class_names)
    plot_training_history()  # Uses correct path
    generate_prediction_samples(model, test_loader, device, test_set.class_names)

    # Save results
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
    print(f"🏆 Final Test Accuracy: {accuracy_score(y_true, y_pred):.4f} ({accuracy_score(y_true, y_pred) * 100:.2f}%)")


if __name__ == '__main__':
    main()
