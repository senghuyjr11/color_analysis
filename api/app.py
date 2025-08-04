import io
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from helper import (
    is_human_face,
    season_tone_palettes,
    palette_to_vector,
    save_palette_image
)

# === Config ===
client_initialized = False
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../color_tone/model/color_convnext_fafl.pth"

# === Use 12-class labels ===
label_classes = [
    "Autumn_Warm", "Autumn_Soft", "Autumn_Deep",
    "Spring_Light", "Spring_Clear", "Spring_Warm",
    "Summer_Cool", "Summer_Soft", "Summer_Light",
    "Winter_Clear", "Winter_Cool", "Winter_Deep"
]

label_encoder = LabelEncoder()
label_encoder.fit(label_classes)

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load Model ===
def load_model():
    model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(1),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features, len(label_encoder.classes_))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# === FastAPI App ===
app = FastAPI(title="Skin Tone Color Analysis API")

# === CORS for frontend access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["http://<frontend-ip>:<port>"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    global client_initialized
    client_initialized = True
    return {"message": "Client initialized. Ready to use API."}


@app.post("/predict_skin_tone")
async def predict(file: UploadFile = File(...)):
    if not client_initialized:
        return {"error": "Please call the root endpoint (/) before using this API."}
    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("Checking for face...")
        if not is_human_face(image):
            print(" No face detected.")
            return {
                "error": "No human face detected in the image. Please upload a clear face photo."
            }
        print("Face detected.")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()
            pred_idx = np.argmax(probs)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

        season, subtype = pred_label.split("_")
        palette_hex = season_tone_palettes.get(pred_label, [])
        palette_vec = palette_to_vector(palette_hex)
        palette_path = save_palette_image(palette_hex, pred_label, file.filename)

        return {
            "predicted_label": pred_label,
            "hex_palette": palette_hex,
            "rgb_vector_15d": palette_vec.tolist(),
            "palette_image_path": palette_path,
            "message_lines": [
                f"Predicted Label: {pred_label}",
                f"HEX Palette: {palette_hex}",
                f"RGB Vector (15D): {palette_vec}",
                f"Palette image saved to: {palette_path}"
            ],
            "season": season,
            "subtype": subtype,
            "confidence": float(probs[pred_idx]),
            "class_probabilities": dict(zip(label_encoder.classes_, map(float, probs)))
        }
    except Exception as e:
        return {"error": str(e)}

# Emotion Analysis Setup
from torchvision.models import densenet121, DenseNet121_Weights

emotion_model_path = "../color_tone/model/best_densenet121_rafdb.pth"
emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# Load emotion model once at startup
emotion_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
emotion_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(emotion_model.classifier.in_features, 7)
)
emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=device))
emotion_model.eval()
emotion_model = emotion_model.to(device)

# Emotion transform (same as test_tf)
emotion_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    if not client_initialized:
        return {"error": "Please call the root endpoint (/) before using this API."}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Face check (reuse from color analysis)
        if not is_human_face(image):
            return {"error": "No human face detected in the image."}

        # Transform and predict
        input_tensor = emotion_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = emotion_model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_emotion = emotion_labels[pred_idx]

        return {
            "emotion": pred_emotion,
            "confidence": float(probs[pred_idx]),
            "class_probabilities": dict(zip(emotion_labels, map(float, probs)))
        }

    except Exception as e:
        return {"error": f"Emotion prediction failed: {str(e)}"}
