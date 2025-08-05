import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

from helper import (
    is_human_face,
    season_tone_palettes,
    palette_to_vector,
    save_palette_image
)
from predictor import predict_skin_tone

# === Config ===
client_initialized = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === FastAPI App ===
app = FastAPI(title="Skin Tone & Emotion Analysis API")

# === CORS for frontend access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
async def predict_skin_tone_endpoint(file: UploadFile = File(...)):
    if not client_initialized:
        return {"error": "Please call the root endpoint (/) before using this API."}
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print("Checking for face...")
        if not is_human_face(image):
            print("No face detected.")
            return {
                "error": "No human face detected in the image. Please upload a clear face photo."
            }
        print("Face detected.")

        # Call the updated prediction function (next step below)
        main_season, confidence, subtones, season_confidences, top_2_seasons = predict_skin_tone(image)
        predicted_label = f"{main_season.capitalize()}_{subtones[0][0].capitalize()}"

        palette_hex = season_tone_palettes.get(predicted_label, [])
        palette_vec = palette_to_vector(palette_hex)
        palette_path = save_palette_image(palette_hex, predicted_label, file.filename)

        return {
            "predicted_label": predicted_label,
            "season": main_season,
            "subtype": subtones[0][0],
            "confidence": confidence,
            "top_2_subtones": subtones,
            "season_confidences": season_confidences,
            "top_2_seasons": top_2_seasons,
            "hex_palette": palette_hex,
            "rgb_vector_15d": palette_vec.tolist(),
            "palette_image_path": palette_path,
        }

    except Exception as e:
        return {"error": str(e)}


# === Emotion Analysis ===
emotion_model_path = "../color_tone/model/best_densenet121_rafdb.pth"
emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

emotion_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
emotion_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(emotion_model.classifier.in_features, 7)
)
emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=device))
emotion_model.eval()
emotion_model = emotion_model.to(device)

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

        if not is_human_face(image):
            return {"error": "No human face detected in the image."}

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
