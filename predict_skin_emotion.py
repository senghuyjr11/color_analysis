import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Emotion classes ===
emotion_classes = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

# === Expanded Color/Emotion Recommendations ===
recommendations = {
    "Winter/Deep": {
        "happy": ["true red", "royal blue"],
        "sad": ["plum", "charcoal"],
        "angry": ["midnight navy", "steel"],
        "neutral": ["black", "crimson"]
    },
    "Winter/Cool": {
        "happy": ["icy blue", "pure white"],
        "sad": ["gray", "cool lavender"],
        "angry": ["teal", "steel gray"],
        "neutral": ["navy", "frost"]
    },
    "Winter/Clear": {
        "happy": ["electric blue", "crystal pink"],
        "sad": ["deep magenta", "silver gray"],
        "angry": ["jet black", "royal purple"],
        "neutral": ["white", "cherry red"]
    },
    "Spring/Warm": {
        "happy": ["coral", "sunny yellow"],
        "sad": ["peach", "light apricot"],
        "angry": ["mint", "cool beige"],
        "neutral": ["ivory", "aqua"]
    },
    "Spring/Clear": {
        "happy": ["buttercup", "sky blue"],
        "sad": ["light orange", "blush"],
        "angry": ["bright teal", "soft green"],
        "neutral": ["light coral", "white"]
    },
    "Spring/Light": {
        "happy": ["apricot", "baby blue"],
        "sad": ["powder pink", "pale yellow"],
        "angry": ["cool mint", "light aqua"],
        "neutral": ["pearl", "light gray"]
    },
    "Summer/Cool": {
        "happy": ["lavender", "soft pink"],
        "sad": ["mauve", "dusty lilac"],
        "angry": ["soft teal", "cool taupe"],
        "neutral": ["light navy", "ash gray"]
    },
    "Summer/Light": {
        "happy": ["sky blue", "soft lavender"],
        "sad": ["dusty pink", "pale mauve"],
        "angry": ["sage green", "taupe"],
        "neutral": ["light gray", "powder blue"]
    },
    "Summer/Soft": {
        "happy": ["blue-gray", "rosy beige"],
        "sad": ["pale plum", "smoky mauve"],
        "angry": ["cool stone", "muted teal"],
        "neutral": ["soft white", "warm gray"]
    },
    "Autumn/Warm": {
        "happy": ["terracotta", "pumpkin"],
        "sad": ["mustard", "olive"],
        "angry": ["deep orange", "earth brown"],
        "neutral": ["camel", "caramel"]
    },
    "Autumn/Soft": {
        "happy": ["rust", "warm peach"],
        "sad": ["mustard", "olive drab"],
        "angry": ["brown", "deep green"],
        "neutral": ["beige", "earth gray"]
    },
    "Autumn/Deep": {
        "happy": ["burnt sienna", "bronze"],
        "sad": ["dark mustard", "wine red"],
        "angry": ["forest green", "espresso"],
        "neutral": ["dark olive", "chocolate"]
    }
}

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

# === Main prediction logic ===
def predict_skin_and_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = shared_transform(image).unsqueeze(0).to(device)

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

# === Recommendation logic ===
def recommend_colors(skin_tone, emotion):
    # Normalize to match dictionary key format
    skin_tone = skin_tone.replace("_", "/").strip().title()  # "Spring_Clear" → "Spring/Clear"

    tone_dict = recommendations.get(skin_tone)
    if tone_dict:
        return tone_dict.get(emotion, tone_dict.get("neutral", []))
    return ["gray"]  # fallback


# === Main Entry Point ===
if __name__ == "__main__":
    image_path = "clustered_seasons_kmeans/Spring/Clear/543.png"
    skin_tone, emotion = predict_skin_and_emotion(image_path)

    print(f"\nDetected Emotion: {emotion}")
    print(f"Detected Skin Tone: {skin_tone}")

    suggestions = recommend_colors(skin_tone, emotion)
    print("\nSuggested Colors to Boost Mood & Match Tone:")
    for color in suggestions:
        print(" -", color)
