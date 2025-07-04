import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Same model as training
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


def load_model_and_classes():
    """Load the trained model and get emotion classes"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get emotion classes from training data
    train_csv = "archive/preprocessed_faces/train_split.csv"
    if Path(train_csv).exists():
        data = pd.read_csv(train_csv)
        emotion_classes = sorted(data["label"].dropna().unique())
        print(f"Found {len(emotion_classes)} emotion classes: {emotion_classes}")

        # Show class distribution from training data
        class_counts = data["label"].value_counts()
        print(f"\nTraining data distribution:")
        for emotion in emotion_classes:
            count = class_counts.get(emotion, 0)
            print(f"  {emotion}: {count} samples")

    else:
        # Default classes if CSV not found
        emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'contempt']
        print(f"Using default classes: {emotion_classes}")

    # Load model
    model = CustomEfficientNet(len(emotion_classes))
    model.load_state_dict(torch.load('models/emotion_model_efficientnet.pth', map_location=device))
    model.to(device)
    model.eval()

    return model, emotion_classes, device


def visualize_prediction(image_path, result):
    """Visualize the image and prediction results"""
    try:
        # Load and show original image
        image = Image.open(image_path).convert('RGB')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Show original image
        ax1.imshow(image)
        ax1.set_title(f"Input Image: {Path(image_path).name}")
        ax1.axis('off')

        # Show prediction bar chart
        emotions = list(result['all_predictions'].keys())
        confidences = list(result['all_predictions'].values())

        # Color bars - red for predicted, blue for others
        colors = ['red' if emotion == result['emotion'] else 'lightblue' for emotion in emotions]

        bars = ax2.barh(emotions, confidences, color=colors)
        ax2.set_xlabel('Confidence')
        ax2.set_title(f'Predictions\nPredicted: {result["emotion"]} ({result["confidence"] * 100:.1f}%)')
        ax2.set_xlim(0, 1)

        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{conf:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in visualization: {e}")


def predict_emotion(image_path, debug=True):
    """Predict emotion from image with debugging info"""

    # Load model and classes
    model, emotion_classes, device = load_model_and_classes()

    # Same transform as training validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"\nOriginal image size: {image.size}")

        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Preprocessed tensor shape: {image_tensor.shape}")

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence_scores = probabilities.cpu().numpy()[0]
            predicted_class = confidence_scores.argmax()
            max_confidence = confidence_scores[predicted_class]

        # Print results
        print(f"\n{'=' * 50}")
        print(f"PREDICTION RESULTS")
        print(f"{'=' * 50}")
        print(f"Image: {image_path}")
        print(f"Predicted Emotion: {emotion_classes[predicted_class]}")
        print(f"Confidence: {max_confidence:.4f} ({max_confidence * 100:.2f}%)")

        print(f"\nAll predictions (sorted by confidence):")
        emotion_conf_pairs = list(zip(emotion_classes, confidence_scores))
        emotion_conf_pairs.sort(key=lambda x: x[1], reverse=True)

        for emotion, conf in emotion_conf_pairs:
            marker = "👉" if emotion == emotion_classes[predicted_class] else "  "
            print(f"{marker} {emotion}: {conf:.4f} ({conf * 100:.2f}%)")

        # Debug info
        if debug:
            print(f"\n{'=' * 50}")
            print(f"DEBUG INFO")
            print(f"{'=' * 50}")
            print(f"Top 3 predictions:")
            for i, (emotion, conf) in enumerate(emotion_conf_pairs[:3]):
                print(f"  {i + 1}. {emotion}: {conf:.4f} ({conf * 100:.2f}%)")

            print(f"\nModel output (logits): {outputs.cpu().numpy()[0]}")
            print(f"Difference between top 2: {emotion_conf_pairs[0][1] - emotion_conf_pairs[1][1]:.4f}")

            # Check if prediction is confident
            if max_confidence < 0.5:
                print(f"⚠️  WARNING: Low confidence prediction (< 50%)")
            elif max_confidence > 0.8:
                print(f"✅ High confidence prediction (> 80%)")
            else:
                print(f"ℹ️  Moderate confidence prediction (50-80%)")

        result = {
            'emotion': emotion_classes[predicted_class],
            'confidence': max_confidence,
            'all_predictions': dict(zip(emotion_classes, confidence_scores))
        }

        # Show visualization
        if debug:
            visualize_prediction(image_path, result)

        return result

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def test_multiple_images():
    """Test on multiple images to see model behavior"""
    test_images = [
        "sad.jpg",
        "happy.jpg",
        "angry.jpg",
        "neutral.jpg"
    ]

    print("Testing multiple images:")
    print("=" * 50)

    for img_path in test_images:
        if Path(img_path).exists():
            result = predict_emotion(img_path, debug=False)
            if result:
                print(f"{img_path} -> {result['emotion']} ({result['confidence'] * 100:.1f}%)")
        else:
            print(f"{img_path} -> File not found")


if __name__ == "__main__":
    # Change this to your image path
    image_path = "sad1.jpg"

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with a valid image path")
    else:
        result = predict_emotion(image_path, debug=True)

        # Optional: Test multiple images
        print(f"\n{'=' * 50}")
        print("Want to test multiple images? Uncomment the line below:")
        print("# test_multiple_images()")
        # test_multiple_images()
