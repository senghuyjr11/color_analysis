import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_large
import torch.nn as nn
import cv2

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from color_tone.face_parsing.model import BiSeNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "color_tone", "merge_dataset_cropped_segmented")
WEIGHTS = os.path.join(PROJECT_ROOT, "..", "color_tone", "face_parsing", "res", "cp", "79999_iter.pth")


MAIN_SEASON_LABELS = ['spring', 'summer', 'autumn', 'winter']

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

def segment_face(image_pil, model):
    print("[INFO] Segmenting face before prediction...")
    img_resized = image_pil.resize((512, 512), Image.BILINEAR)
    tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    skin_mask = (parsing == 1).astype(np.uint8) * 255
    skin_mask = cv2.merge([skin_mask]*3)
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    result = cv2.bitwise_and(img_cv, skin_mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

def load_convnext_model(model_path, num_classes):
    model = convnext_large(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

def predict_skin_tone(image_pil):
    face_segmenter = load_face_segmenter()
    segmented = segment_face(image_pil, face_segmenter)
    # segmented.save("debug_segmented_face.png") for debug make sure that it segmented before prediction

    # Main season prediction
    main_model = load_convnext_model(MODELS["main_season"]["path"], 4)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(segmented).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = main_model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    labels = MODELS["main_season"]["labels"]
    season_confidences = dict(zip(labels, map(float, probs)))

    # Top-2 seasons
    sorted_indices = np.argsort(probs)[::-1]
    top_2_seasons = [
        (labels[sorted_indices[0]], float(probs[sorted_indices[0]])),
        (labels[sorted_indices[1]], float(probs[sorted_indices[1]]))
    ]

    main_season = labels[sorted_indices[0]]
    confidence = float(probs[sorted_indices[0]])

    # Subtone prediction
    subtone_config = MODELS[f"subtone_{main_season}"]
    subtone_model = load_convnext_model(subtone_config["path"], len(subtone_config["labels"]))
    with torch.no_grad():
        logits = subtone_model(input_tensor)
        subtone_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_subtone_indices = np.argsort(subtone_probs)[::-1][:2]
    subtones = [(subtone_config["labels"][i], float(subtone_probs[i])) for i in top_subtone_indices]

    return main_season, confidence, subtones, season_confidences, top_2_seasons

