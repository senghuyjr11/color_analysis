import os
import io
import torch
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.preprocessing import LabelEncoder
from helper import is_human_face


# === Config ===
client_initialized = False
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../models/color_convnext_fafl.pth"

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
        return {
            "prediction": pred_label,
            "season": season,
            "subtype": subtype,
            "confidence": float(probs[pred_idx])
        }

    except Exception as e:
        return {"error": str(e)}
