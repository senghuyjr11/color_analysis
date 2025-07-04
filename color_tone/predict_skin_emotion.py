import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import mediapipe as mp

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Emotion classes ===
emotion_classes = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

# === Rebuild label encoder from training CSV ===
train_df = pd.read_csv("csv/train.csv")
label_encoder = LabelEncoder()
label_encoder.fit(train_df["label"].tolist())

# === Shared transform ===
shared_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# === Load skin tone model ===
color_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
color_model.classifier = nn.Sequential(
    nn.Flatten(1),
    nn.LayerNorm(768, eps=1e-06),
    nn.Dropout(0.5),
    nn.Linear(768, len(label_encoder.classes_))
)
color_model.load_state_dict(torch.load("models/seasonal_model_best_convnexttiny.pth", map_location=device))
color_model.to(device).eval()

# === Load emotion model ===
emotion_model = resnet50(weights=ResNet50_Weights.DEFAULT)
emotion_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(emotion_model.fc.in_features, 7)
)
emotion_model.load_state_dict(torch.load("models/best_resnet50_rafdb.pth", map_location=device))
emotion_model.to(device).eval()

# === Background removal using MediaPipe FaceMesh ===
def remove_background(image: Image.Image) -> Image.Image:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    image_np = np.array(image)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return image  # return original image if no face is detected

        h, w, _ = image_np.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get landmarks and draw a convex hull
        face_points = []
        for lm in results.multi_face_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            face_points.append([x, y])

        face_points = np.array(face_points)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(face_points)
        cv2 = __import__('cv2')
        cv2.fillConvexPoly(mask, face_points[hull.vertices], 255)

        # Apply mask
        result = image_np.copy()
        result[mask == 0] = 255  # white background
        result_img = Image.fromarray(result)

        return result_img

# === Main prediction logic ===
def predict_skin_and_emotion(image_path):
    image = Image.open(image_path).convert("RGB")

    # Remove background first
    face_only_image = remove_background(image)

    # Apply transform
    tensor = shared_transform(face_only_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Skin tone prediction
        skin_output = color_model(tensor)
        skin_idx = skin_output.argmax(dim=1).item()
        skin_label = label_encoder.inverse_transform([skin_idx])[0]

        # Emotion prediction
        emotion_output = emotion_model(tensor)
        emotion_idx = emotion_output.argmax(dim=1).item()
        emotion_label = emotion_classes[emotion_idx]

    return skin_label, emotion_label


# === Main Entry Point ===
if __name__ == "__main__":
    image_path = "happy.jpg"
    skin_tone, emotion = predict_skin_and_emotion(image_path)

    print(f"\nDetected Emotion: {emotion}")
    print(f"Detected Skin Tone: {skin_tone}")
