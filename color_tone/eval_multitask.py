import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pickle
from PIL import Image
import random

from color_tone.train_multitask import MultiTaskColorModel, MultiTaskColorDataset


class ModelEvaluator:
    def __init__(self, model_path, history_path, test_csv, data_root, results_dir='results'):
        """
        Initialize evaluator with model and test data

        Args:
            model_path (str): Path to saved model
            history_path (str): Path to training history
            test_csv (str): Path to test CSV
            data_root (str): Root directory for images
            results_dir (str): Directory to save evaluation results
        """
        self.model_path = model_path
        self.history_path = history_path
        self.test_csv = test_csv
        self.data_root = data_root
        self.results_dir = results_dir

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Load model and history
        self.load_model()
        self.load_history()
        self.prepare_test_data()

    def load_model(self):
        """Load the trained model"""
        print("Loading model...")
        checkpoint = torch.load(self.model_path, map_location='cpu')

        self.season_classes = checkpoint['season_classes']
        self.tone_classes = checkpoint['tone_classes']
        self.season_to_idx = checkpoint['season_to_idx']
        self.tone_to_idx = checkpoint['tone_to_idx']
        self.config = checkpoint['config']

        # Initialize model
        self.model = MultiTaskColorModel(
            num_season_classes=len(self.season_classes),
            num_tone_classes=len(self.tone_classes),
            model_name=self.config['model_name'],
            pretrained=False
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        print(f"Model loaded with {len(self.season_classes)} season classes and {len(self.tone_classes)} tone classes")

    def load_history(self):
        """Load training history"""
        print("Loading training history...")
        with open(self.history_path, 'rb') as f:
            self.history = pickle.load(f)
        print(f"History loaded: {self.history['epochs_completed']} epochs completed")

    def prepare_test_data(self):
        """Prepare test dataset and dataloader"""
        print("Preparing test data...")

        val_transform = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.test_dataset = MultiTaskColorDataset(
            csv_file=self.test_csv,
            root_dir=self.data_root,
            transform=val_transform
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def get_predictions(self):
        """Get model predictions on test set"""
        print("Getting model predictions...")

        all_season_preds = []
        all_tone_preds = []
        all_season_labels = []
        all_tone_labels = []
        all_season_probs = []
        all_tone_probs = []
        all_images = []
        all_paths = []

        with torch.no_grad():
            for i, (images, season_labels, tone_labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                season_labels = season_labels.to(self.device)
                tone_labels = tone_labels.to(self.device)

                season_logits, tone_logits = self.model(images)

                season_probs = torch.softmax(season_logits, dim=1)
                tone_probs = torch.softmax(tone_logits, dim=1)

                _, season_predicted = torch.max(season_logits, 1)
                _, tone_predicted = torch.max(tone_logits, 1)

                all_season_preds.extend(season_predicted.cpu().numpy())
                all_tone_preds.extend(tone_predicted.cpu().numpy())
                all_season_labels.extend(season_labels.cpu().numpy())
                all_tone_labels.extend(tone_labels.cpu().numpy())
                all_season_probs.extend(season_probs.cpu().numpy())
                all_tone_probs.extend(tone_probs.cpu().numpy())

                # Store some images for visualization
                if i < 5:  # Store first few batches
                    all_images.extend(images.cpu())
                    start_idx = i * self.test_loader.batch_size
                    end_idx = start_idx + images.size(0)
                    batch_paths = self.test_dataset.annotations.iloc[start_idx:end_idx]['path'].tolist()
                    all_paths.extend(batch_paths)

        return {
            'season_preds': np.array(all_season_preds),
            'tone_preds': np.array(all_tone_preds),
            'season_labels': np.array(all_season_labels),
            'tone_labels': np.array(all_tone_labels),
            'season_probs': np.array(all_season_probs),
            'tone_probs': np.array(all_tone_probs),
            'images': all_images[:50],  # Keep first 50 images
            'paths': all_paths[:50]
        }

    def plot_training_curves(self):
        """Plot training and validation curves"""
        print("Plotting training curves...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        epochs = range(1, len(self.history['train_losses']) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_losses'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_losses'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Combined accuracy curves
        axes[0, 1].plot(epochs, self.history['train_combined_accs'], 'b-', label='Train Combined Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_combined_accs'], 'r-', label='Val Combined Acc', linewidth=2)
        axes[0, 1].set_title('Combined Accuracy (Season + Tone)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Season accuracy curves
        axes[1, 0].plot(epochs, self.history['train_season_accs'], 'b-', label='Train Season Acc', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_season_accs'], 'r-', label='Val Season Acc', linewidth=2)
        axes[1, 0].set_title('Season Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Tone accuracy curves
        axes[1, 1].plot(epochs, self.history['train_tone_accs'], 'b-', label='Train Tone Acc', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_tone_accs'], 'r-', label='Val Tone Acc', linewidth=2)
        axes[1, 1].set_title('Tone Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, predictions):
        """Plot confusion matrices for both tasks"""
        print("Plotting confusion matrices...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Season confusion matrix
        season_cm = confusion_matrix(predictions['season_labels'], predictions['season_preds'])
        season_cm_norm = season_cm.astype('float') / season_cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(season_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.season_classes, yticklabels=self.season_classes,
                    ax=axes[0])
        axes[0].set_title('Season Classification\nConfusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # Tone confusion matrix
        tone_cm = confusion_matrix(predictions['tone_labels'], predictions['tone_preds'])
        tone_cm_norm = tone_cm.astype('float') / tone_cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(tone_cm_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=self.tone_classes, yticklabels=self.tone_classes,
                    ax=axes[1])
        axes[1].set_title('Tone Classification\nConfusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        # Combined confusion matrix (reconstruct original labels)
        combined_labels_true = []
        combined_labels_pred = []

        for i in range(len(predictions['season_labels'])):
            season_true = self.season_classes[predictions['season_labels'][i]]
            tone_true = self.tone_classes[predictions['tone_labels'][i]]
            season_pred = self.season_classes[predictions['season_preds'][i]]
            tone_pred = self.tone_classes[predictions['tone_preds'][i]]

            combined_labels_true.append(f"{season_true}_{tone_true}")
            combined_labels_pred.append(f"{season_pred}_{tone_pred}")

        unique_combined = sorted(list(set(combined_labels_true + combined_labels_pred)))
        combined_cm = confusion_matrix(combined_labels_true, combined_labels_pred, labels=unique_combined)
        combined_cm_norm = combined_cm.astype('float') / (combined_cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        sns.heatmap(combined_cm_norm, annot=False, cmap='Reds', ax=axes[2])
        axes[2].set_title('Combined (Season + Tone)\nConfusion Matrix', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'comparison_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_per_class_metrics(self, predictions):
        """Plot per-class precision, recall, and F1-score"""
        print("Plotting per-class metrics...")

        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # Season metrics
        season_report = classification_report(
            predictions['season_labels'],
            predictions['season_preds'],
            target_names=self.season_classes,
            output_dict=True
        )

        season_metrics = pd.DataFrame(season_report).T.iloc[:-3]  # Exclude avg rows
        season_metrics = season_metrics[['precision', 'recall', 'f1-score']].astype(float)

        x = np.arange(len(season_metrics.index))
        width = 0.25

        axes[0].bar(x - width, season_metrics['precision'], width, label='Precision', alpha=0.8)
        axes[0].bar(x, season_metrics['recall'], width, label='Recall', alpha=0.8)
        axes[0].bar(x + width, season_metrics['f1-score'], width, label='F1-Score', alpha=0.8)

        axes[0].set_xlabel('Season Classes')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Season Classification - Per Class Metrics', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(season_metrics.index, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        # Tone metrics
        tone_report = classification_report(
            predictions['tone_labels'],
            predictions['tone_preds'],
            target_names=self.tone_classes,
            output_dict=True
        )

        tone_metrics = pd.DataFrame(tone_report).T.iloc[:-3]  # Exclude avg rows
        tone_metrics = tone_metrics[['precision', 'recall', 'f1-score']].astype(float)

        x = np.arange(len(tone_metrics.index))

        axes[1].bar(x - width, tone_metrics['precision'], width, label='Precision', alpha=0.8)
        axes[1].bar(x, tone_metrics['recall'], width, label='Recall', alpha=0.8)
        axes[1].bar(x + width, tone_metrics['f1-score'], width, label='F1-Score', alpha=0.8)

        axes[1].set_xlabel('Tone Classes')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Tone Classification - Per Class Metrics', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tone_metrics.index, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, predictions):
        """Plot ROC curves for each class"""
        print("Plotting ROC curves...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Season ROC curves
        season_labels_bin = label_binarize(predictions['season_labels'], classes=range(len(self.season_classes)))

        for i, class_name in enumerate(self.season_classes):
            fpr, tpr, _ = roc_curve(season_labels_bin[:, i], predictions['season_probs'][:, i])
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Season Classification - ROC Curves', fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Tone ROC curves
        tone_labels_bin = label_binarize(predictions['tone_labels'], classes=range(len(self.tone_classes)))

        for i, class_name in enumerate(self.tone_classes):
            fpr, tpr, _ = roc_curve(tone_labels_bin[:, i], predictions['tone_probs'][:, i])
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2)
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Tone Classification - ROC Curves', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'roc_curves_per_class.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_samples(self, predictions):
        """Plot sample predictions with images"""
        print("Plotting prediction samples...")

        # Denormalization function
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return tensor * std + mean

        # Select samples: correct and incorrect predictions
        correct_indices = []
        incorrect_indices = []

        for i in range(min(len(predictions['season_preds']), len(predictions['images']))):
            season_correct = predictions['season_preds'][i] == predictions['season_labels'][i]
            tone_correct = predictions['tone_preds'][i] == predictions['tone_labels'][i]

            if season_correct and tone_correct:
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)

        # Select subset for visualization
        num_samples = 16
        selected_indices = (correct_indices[:num_samples // 2] +
                            incorrect_indices[:num_samples // 2])[:num_samples]

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for idx, img_idx in enumerate(selected_indices):
            if img_idx >= len(predictions['images']):
                continue

            # Get image and predictions
            image = predictions['images'][img_idx]
            season_pred = self.season_classes[predictions['season_preds'][img_idx]]
            tone_pred = self.tone_classes[predictions['tone_preds'][img_idx]]
            season_true = self.season_classes[predictions['season_labels'][img_idx]]
            tone_true = self.tone_classes[predictions['tone_labels'][img_idx]]

            # Denormalize and convert to displayable format
            image = denormalize(image)
            image = torch.clamp(image, 0, 1)
            image = image.permute(1, 2, 0).numpy()

            # Display image
            axes[idx].imshow(image)
            axes[idx].axis('off')

            # Create title with predictions
            season_color = 'green' if season_pred == season_true else 'red'
            tone_color = 'green' if tone_pred == tone_true else 'red'

            title = f"True: {season_true}_{tone_true}\n"
            title += f"Pred: {season_pred}_{tone_pred}"

            # Color code the title based on correctness
            if season_pred == season_true and tone_pred == tone_true:
                axes[idx].set_title(title, fontsize=10, color='green', fontweight='bold')
            else:
                axes[idx].set_title(title, fontsize=10, color='red', fontweight='bold')

        # Hide empty subplots
        for idx in range(len(selected_indices), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Prediction Samples (Green=Correct, Red=Incorrect)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'prediction_samples.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def run_evaluation(self):
        """Run complete evaluation and generate all plots"""
        print("Starting comprehensive evaluation...")

        # Get predictions
        predictions = self.get_predictions()

        # Calculate accuracies
        season_acc = (predictions['season_preds'] == predictions['season_labels']).mean() * 100
        tone_acc = (predictions['tone_preds'] == predictions['tone_labels']).mean() * 100
        combined_acc = ((predictions['season_preds'] == predictions['season_labels']) &
                        (predictions['tone_preds'] == predictions['tone_labels'])).mean() * 100

        print(f"\nTest Set Results:")
        print(f"Season Accuracy: {season_acc:.2f}%")
        print(f"Tone Accuracy: {tone_acc:.2f}%")
        print(f"Combined Accuracy: {combined_acc:.2f}%")

        # Generate all plots
        self.plot_training_curves()
        self.plot_confusion_matrices(predictions)
        self.plot_per_class_metrics(predictions)
        self.plot_roc_curves(predictions)
        self.plot_prediction_samples(predictions)

        # Save detailed classification reports
        season_report = classification_report(
            predictions['season_labels'],
            predictions['season_preds'],
            target_names=self.season_classes,
            digits=4
        )

        tone_report = classification_report(
            predictions['tone_labels'],
            predictions['tone_preds'],
            target_names=self.tone_classes,
            digits=4
        )

        with open(os.path.join(self.results_dir, 'classification_reports.txt'), 'w') as f:
            f.write("SEASON CLASSIFICATION REPORT:\n")
            f.write("=" * 50 + "\n")
            f.write(season_report)
            f.write("\n\nTONE CLASSIFICATION REPORT:\n")
            f.write("=" * 50 + "\n")
            f.write(tone_report)
            f.write(f"\n\nSUMMARY:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Season Accuracy: {season_acc:.2f}%\n")
            f.write(f"Tone Accuracy: {tone_acc:.2f}%\n")
            f.write(f"Combined Accuracy: {combined_acc:.2f}%\n")

        print(f"\nEvaluation complete! Results saved to '{self.results_dir}' directory:")
        print(f"  - training_curves.png")
        print(f"  - comparison_confusion_matrix.png")
        print(f"  - per_class_metrics.png")
        print(f"  - roc_curves_per_class.png")
        print(f"  - prediction_samples.png")
        print(f"  - classification_reports.txt")


def main():
    """Main evaluation function"""

    config = {
        'model_path': 'model_checkpoints/best_multitask_model.pth',  # Fixed path
        'history_path': 'model_checkpoints/training_history.pkl',  # Fixed path
        'test_csv': 'ORIGINAL_RGB_NOT_PROCESSED/test.csv',
        'data_root': '',
        'results_dir': 'results'
    }

    # Check if files exist
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file '{config['model_path']}' not found!")
        return

    if not os.path.exists(config['history_path']):
        print(f"Error: History file '{config['history_path']}' not found!")
        return

    if not os.path.exists(config['test_csv']):
        print(f"Error: Test CSV file '{config['test_csv']}' not found!")
        return

    # Run evaluation
    evaluator = ModelEvaluator(
        model_path=config['model_path'],
        history_path=config['history_path'],
        test_csv=config['test_csv'],
        data_root=config['data_root'],
        results_dir=config['results_dir']
    )

    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
