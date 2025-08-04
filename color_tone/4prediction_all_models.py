import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_large
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import heapq

# === Settings ===
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "merge_dataset_cropped_segmented"
MAIN_MODEL_PATH = os.path.join(DATASET_DIR, "convnext_season_large.pth")

# === Image Transform ===
predict_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load main model and label encoder
main_train_csv = os.path.join(DATASET_DIR, "train.csv")
main_df = pd.read_csv(main_train_csv)
main_df['season'] = main_df['label'].apply(lambda x: x.split('_')[0])
main_label_encoder = LabelEncoder()
main_label_encoder.fit(main_df['season'])
num_seasons = len(main_label_encoder.classes_)

main_model = convnext_large(weights=None)
main_model.classifier[2] = nn.Linear(main_model.classifier[2].in_features, num_seasons)
main_model.load_state_dict(torch.load(MAIN_MODEL_PATH, map_location=DEVICE))
main_model = main_model.to(DEVICE)
main_model.eval()


# === Predict Function ===
def predict_and_print(image_path, topk=2):
    img = Image.open(image_path).convert('RGB')
    img_tensor = predict_transform(img).unsqueeze(0).to(DEVICE)

    # === Main Season Prediction
    with torch.no_grad():
        season_logits = main_model(img_tensor)
        season_probs = torch.softmax(season_logits, dim=1).squeeze().cpu().numpy()

    all_season_probs = {main_label_encoder.inverse_transform([i])[0]: float(season_probs[i])
                        for i in range(len(season_probs))}

    print("📊 All Season Probabilities:")
    for label, prob in all_season_probs.items():
        print(f"  - {label}: {prob:.4f}")

    top_season_idx = heapq.nlargest(topk, range(len(season_probs)), season_probs.__getitem__)
    top_seasons = [(main_label_encoder.inverse_transform([i])[0], float(season_probs[i]))
                   for i in top_season_idx]

    # === Subtone Predictions per Season
    final_results = {}
    for season, season_prob in top_seasons:
        model_path = os.path.join(DATASET_DIR, f"convnext_subtone_{season}.pth")
        if not os.path.exists(model_path):
            print(f"⛔ Skipping {season} — model not found.")
            continue

        subtone_df = main_df[main_df['season'] == season]
        subtone_le = LabelEncoder()
        subtone_le.fit(subtone_df['label'].apply(lambda x: x.split('_')[1]))
        num_subtones = len(subtone_le.classes_)

        # Load subtone model
        subtone_model = convnext_large(weights=None)
        subtone_model.classifier[2] = nn.Linear(subtone_model.classifier[2].in_features, num_subtones)
        subtone_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        subtone_model = subtone_model.to(DEVICE)
        subtone_model.eval()

        with torch.no_grad():
            subtone_logits = subtone_model(img_tensor)
            subtone_probs = torch.softmax(subtone_logits, dim=1).squeeze().cpu().numpy()

        all_subtone_probs = {subtone_le.inverse_transform([i])[0]: float(subtone_probs[i])
                             for i in range(len(subtone_probs))}

        print(f"\n📊 All Subtone Probabilities for {season}:")
        for label, prob in all_subtone_probs.items():
            print(f"  - {label}: {prob:.4f}")

        top_subtone_idx = heapq.nlargest(topk, range(len(subtone_probs)), subtone_probs.__getitem__)
        top_subtones = [(subtone_le.inverse_transform([i])[0], float(subtone_probs[i]))
                        for i in top_subtone_idx]

        final_results[season] = top_subtones

    # === Final Decision
    print("\n✅ Final Top Predictions:")
    for season, prob in top_seasons:
        print(f"  🌸 {season.upper()} ({prob:.4f})")
        if season in final_results:
            for subtone, p in final_results[season]:
                print(f"     ↳ {subtone} ({p:.4f})")

if __name__ == "__main__":
    predict_and_print("merge_dataset_cropped_segmented/autumn/deep/981.png", topk=2)
