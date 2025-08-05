import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import cv2
from face_parsing.model import BiSeNet


# ==== Paths ====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "merge_dataset_cropped_segmented")
WEIGHTS = os.path.join(PROJECT_ROOT, "face_parsing", "res", "cp", "79999_iter.pth")
IMG_PATH = "ORIGINAL_RGB_NOT_PROCESSED/test/spring/warm/52.png"

# ==== Device ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== BiSeNet for Segmentation ====
to_tensor = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def load_face_segmenter():
    model = BiSeNet(n_classes=19)
    model.to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    return model

def segment_face(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((512, 512), Image.BILINEAR)
    tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    skin_mask = (parsing == 1).astype(np.uint8) * 255
    skin_mask = cv2.merge([skin_mask]*3)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = cv2.bitwise_and(img_cv, skin_mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

from torchvision.models import convnext_large
import torch.nn as nn

def load_convnext_model(model_path, num_classes):
    model = convnext_large(weights=None)  # No pretrained weights
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ==== Prediction Function ====
def predict_top_k(image_pil, model_path, top_k=2, labels=None):
    model = load_convnext_model(model_path, num_classes=len(labels))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    sorted_indices = np.argsort(probs)[::-1]
    if labels is None:
        labels = [f"Class {i}" for i in range(len(probs))]

    print("Full Prediction Confidence:")
    for i in range(len(labels)):
        print(f"{labels[sorted_indices[i]]}: {probs[sorted_indices[i]]:.4f}")

    top_preds = []
    if labels == MAIN_SEASON_LABELS:
        print("Top Predictions:")
        for i in range(top_k):
            label = labels[sorted_indices[i]]
            prob = probs[sorted_indices[i]]
            print(f"Top {i + 1}: {label} ({prob:.2%})")
            top_preds.append((label, prob))
    else:
        for i in range(top_k):
            label = labels[sorted_indices[i]]
            prob = probs[sorted_indices[i]]
            top_preds.append((label, prob))
    print("-" * 50)

    return top_preds

# ==== Label Sets ====
MAIN_SEASON_LABELS = ['spring', 'summer', 'autumn', 'winter']

# ==== MODEL PATHS ====
MODELS = {
    "main_season": {
        "path": os.path.join(MODEL_DIR, "convnext_season_large.pth"),
        "labels": ['spring', 'summer', 'autumn', 'winter']
    },
    "subtone_spring": {
        "path": os.path.join(MODEL_DIR, "convnext_subtone_spring.pth"),
        "labels": ['bright', 'light', 'warm']
    },
    "subtone_summer": {
        "path": os.path.join(MODEL_DIR, "convnext_subtone_summer.pth"),
        "labels": ['cool', 'light', 'soft']
    },
    "subtone_autumn": {
        "path": os.path.join(MODEL_DIR, "convnext_subtone_autumn.pth"),
        "labels": ['deep', 'soft', 'warm']
    },
    "subtone_winter": {
        "path": os.path.join(MODEL_DIR, "convnext_subtone_winter.pth"),
        "labels": ['bright', 'cool', 'deep']
    }
}

# ==== Main Execution ====
if __name__ == "__main__":
    print("Segmenting input image...")
    face_parser = load_face_segmenter()
    segmented = segment_face(IMG_PATH, face_parser)

    # Optional: Show segmented image
    # segmented.show(title="Segmented Image")

    print("Running predictions...\n")

    all_results = {}
    main_top2 = []

    # Run all models
    top_main_season = ""
    subtone_results = {}

    for name, config in MODELS.items():
        print(f"MODEL: {name}")
        top2 = predict_top_k(segmented, config["path"], top_k=2, labels=config["labels"])

        if name == "main_season":
            main_top2 = top2
            top_main_season = top2[0][0]  # Get top-1 main season

        if "subtone" in name:
            subtone_results[name] = top2

    # Final summary based on top-1 main season
    top_main_season = main_top2[0][0]
    subtone_key = f"subtone_{top_main_season}"

    top_main_season = main_top2[0][0]
    top_main_prob = main_top2[0][1]

    print("\nFinal Prediction Results")
    print(f"Main Season: {top_main_season} ({top_main_prob:.2%})")

    matching_subtone_key = f"subtone_{top_main_season}"
    if matching_subtone_key in subtone_results:
        print("Top 2 Subtones:")
        for i, (label, prob) in enumerate(subtone_results[matching_subtone_key], 1):
            print(f"  {i}. {label} ({prob:.2%})")
