import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score, roc_curve, auc)
import json
import os
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SeasonalColorDataset(Dataset):
    def __init__(self, csv_file, transform):
        """Dataset class for evaluation"""
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Create label mapping
        self.unique_labels = sorted(self.data['combined_label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        label = self.label_to_idx[row['combined_label']]

        return image, label, row['combined_label'], img_path


def create_improved_model_for_eval(num_classes, model_name, dropout_rate):
    """Recreate the improved model architecture for evaluation"""

    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier[1].in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier[1].in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


def load_improved_model(model_path, device):
    """Load the improved trained model"""
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Recreate improved model architecture
    model = create_improved_model_for_eval(
        num_classes=config['num_classes'],
        model_name=config['model_name'],
        dropout_rate=0.4
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Architecture: {config['model_name']}")
    print(f"Classes: {config['num_classes']}")
    print(f"Training validation accuracy: {checkpoint['val_acc']:.2f}%")

    return model, config


def get_test_transform():
    """Transform for test data"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def evaluate_model(model, test_loader, device, class_names):
    """Run inference and collect predictions"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_label_names = []
    all_image_paths = []
    all_probabilities = []

    print("Running inference on test dataset...")

    with torch.no_grad():
        for images, labels, label_names, img_paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_label_names.extend(label_names)
            all_image_paths.extend(img_paths)
            all_probabilities.extend(probabilities.cpu().numpy())

    return (np.array(all_predictions), np.array(all_labels),
            all_label_names, all_image_paths, np.array(all_probabilities))


def create_comparison_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Create comparison confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison confusion matrix saved to: {save_path}")


def create_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Create per-class metrics visualization"""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    results_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    # Sort by F1-score
    results_df = results_df.sort_values('F1-Score', ascending=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    # F1-Score horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    bars = ax1.barh(range(len(results_df)), results_df['F1-Score'], color=colors)
    ax1.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    ax1.set_xlabel('F1-Score')
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['Class'])
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center', fontsize=9)

    # Precision vs Recall scatter
    scatter = ax2.scatter(results_df['Precision'], results_df['Recall'],
                          c=results_df['F1-Score'], cmap='viridis', s=100, alpha=0.7)
    ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1-Score')

    # Support (sample counts)
    ax3.bar(range(len(results_df)), results_df['Support'], color='lightcoral')
    ax3.set_title('Support (Number of Samples)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xticks(range(len(results_df)))
    ax3.set_xticklabels(results_df['Class'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # Metrics comparison
    x_pos = np.arange(len(results_df))
    width = 0.25
    ax4.bar(x_pos - width, results_df['Precision'], width, label='Precision', alpha=0.8)
    ax4.bar(x_pos, results_df['Recall'], width, label='Recall', alpha=0.8)
    ax4.bar(x_pos + width, results_df['F1-Score'], width, label='F1-Score', alpha=0.8)
    ax4.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Score')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(results_df['Class'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-class metrics saved to: {save_path}")


def create_prediction_samples(y_true, y_pred, y_proba, class_names, image_paths, save_path):
    """Create prediction samples visualization"""
    fig, axes = plt.subplots(4, 6, figsize=(20, 14))
    fig.suptitle('Prediction Samples (Correct and Incorrect)', fontsize=16, fontweight='bold')

    # Get correct and incorrect predictions
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]

    # Sample predictions
    sample_indices = []

    # Add some correct predictions
    if len(correct_indices) > 12:
        sample_indices.extend(np.random.choice(correct_indices, 12, replace=False))
    else:
        sample_indices.extend(correct_indices)

    # Add some incorrect predictions
    if len(incorrect_indices) > 12:
        sample_indices.extend(np.random.choice(incorrect_indices, 12, replace=False))
    else:
        sample_indices.extend(incorrect_indices)

    # Shuffle and take first 24
    np.random.shuffle(sample_indices)
    sample_indices = sample_indices[:24]

    for i, idx in enumerate(sample_indices):
        row = i // 6
        col = i % 6
        ax = axes[row, col]

        # Load and display image
        try:
            img = Image.open(image_paths[idx]).convert('RGB')
            img_resized = img.resize((128, 128))
            ax.imshow(img_resized)
        except:
            # Create placeholder if image can't be loaded
            ax.imshow(np.random.rand(128, 128, 3))

        # Get prediction info
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        confidence = y_proba[idx][y_pred[idx]]
        is_correct = y_true[idx] == y_pred[idx]

        # Set title with color coding
        title_color = 'green' if is_correct else 'red'
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}"

        ax.set_title(title, fontsize=8, color=title_color, fontweight='bold')
        ax.axis('off')

    # Hide any unused subplots
    for i in range(len(sample_indices), 24):
        row = i // 6
        col = i % 6
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Prediction samples saved to: {save_path}")


def create_roc_curves_per_class(y_true, y_proba, class_names, save_path):
    """Create ROC curves for each class"""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle

    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('ROC Curves per Class', fontsize=16, fontweight='bold')

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple',
                    'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'])

    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve (AUC = {roc_auc[i]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{class_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved to: {save_path}")


def load_training_history(model_dir):
    """Load training history from the model directory"""
    history_path = os.path.join(model_dir, 'training_history.json')
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        return history
    except:
        return None


def create_training_curves(model_dir, save_path):
    """Create training curves plot"""
    history = load_training_history(model_dir)

    if history is None:
        # Create dummy plot if no history
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Training history not available',
                transform=ax.transAxes, ha='center', va='center', fontsize=16)
        ax.set_title('Training Curves', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"⚠ Training curves placeholder saved to: {save_path}")
        return

    train_losses = history['train_losses']
    train_accs = history['train_accs']
    val_losses = history['val_losses']
    val_accs = history['val_accs']
    epochs = range(1, len(train_losses) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate (if available, otherwise show loss difference)
    loss_diff = [v - t for v, t in zip(val_losses, train_losses)]
    ax3.plot(epochs, loss_diff, 'purple', linewidth=2, alpha=0.8)
    ax3.set_title('Validation - Training Loss (Overfitting Monitor)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    # Combined plot
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(epochs, train_losses, 'b-', alpha=0.7, label='Train Loss')
    line2 = ax4.plot(epochs, val_losses, 'r-', alpha=0.7, label='Val Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')

    line3 = ax4_twin.plot(epochs, train_accs, 'b--', alpha=0.7, label='Train Acc')
    line4 = ax4_twin.plot(epochs, val_accs, 'r--', alpha=0.7, label='Val Acc')
    ax4_twin.set_ylabel('Accuracy (%)', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.set_title('Combined Learning Curves')

    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to: {save_path}")


def generate_five_plot_evaluation(model_path, test_csv, save_dir):
    """
    Generate exactly 5 plots for model evaluation

    Args:
        model_path: Path to saved model (.pth file)
        test_csv: Path to test CSV file
        save_dir: Directory to save the 5 plots
    """

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_improved_model(model_path, device)
    class_names = config['class_names']

    # Load test dataset
    print(f"Loading test dataset from {test_csv}...")
    test_transform = get_test_transform()
    test_dataset = SeasonalColorDataset(csv_file=test_csv, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Test samples: {len(test_dataset)}")

    # Run evaluation
    predictions, true_labels, label_names, image_paths, probabilities = evaluate_model(
        model, test_loader, device, class_names)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    # Generate the 5 plots
    print(f"\n📊 Generating 5 evaluation plots...")

    # 1. Comparison Confusion Matrix
    create_comparison_confusion_matrix(
        y_true=true_labels,
        y_pred=predictions,
        class_names=class_names,
        save_path=os.path.join(save_dir, 'comparison_confusion_matrix.png')
    )

    # 2. Per-Class Metrics
    create_per_class_metrics(
        y_true=true_labels,
        y_pred=predictions,
        class_names=class_names,
        save_path=os.path.join(save_dir, 'per_class_metrics.png')
    )

    # 3. Prediction Samples
    create_prediction_samples(
        y_true=true_labels,
        y_pred=predictions,
        y_proba=probabilities,
        class_names=class_names,
        image_paths=image_paths,
        save_path=os.path.join(save_dir, 'prediction_samples.png')
    )

    # 4. ROC Curves per Class
    create_roc_curves_per_class(
        y_true=true_labels,
        y_proba=probabilities,
        class_names=class_names,
        save_path=os.path.join(save_dir, 'roc_curves_per_class.png')
    )

    # 5. Training Curves
    model_dir = os.path.dirname(model_path)
    create_training_curves(
        model_dir=model_dir,
        save_path=os.path.join(save_dir, 'training_curves.png')
    )

    print(f"\n✅ All 5 plots generated successfully!")
    print(f"📁 Saved to: {save_dir}")
    print(f"📊 Test Accuracy: {overall_accuracy * 100:.2f}%")

    return overall_accuracy


# Example usage
if __name__ == "__main__":
    # Update these paths
    model_path = "./improved_seasonal_model/best_model.pth"
    test_csv = "./dataset_splits/test.csv"

    # Generate the 5 evaluation plots
    accuracy = generate_five_plot_evaluation(
        model_path=model_path,
        test_csv=test_csv,
        save_dir='./results'
    )

    print(f"\n🎉 Evaluation completed!")
    print(f"📈 Final accuracy: {accuracy * 100:.2f}%")
