# predictor.py
import os
import io
import numpy as np
import torch
from PIL import Image
import cv2

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_parsing.model import BiSeNet
from gemini_agent import analyze_with_gemini

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(PROJECT_ROOT, "..", "face_parsing", "res", "cp", "79999_iter.pth")


# ====================================================
# FACE SEGMENTATION (SKIN REGION)
# ====================================================
def load_face_segmenter():
    model = BiSeNet(n_classes=19)
    model.to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    return model


def segment_face(image_pil, model):
    """Extract only facial skin regions using BiSeNet segmentation."""
    print("[INFO] Segmenting face before prediction...")
    original_size = image_pil.size
    img_resized = image_pil.resize((512, 512), Image.BILINEAR)

    # Convert to tensor
    import torchvision.transforms as transforms
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    skin_mask = (parsing == 1).astype(np.uint8) * 255
    skin_mask_resized = cv2.resize(skin_mask, original_size, interpolation=cv2.INTER_NEAREST)
    skin_mask_3ch = cv2.merge([skin_mask_resized] * 3)

    # Convert original image to CV2 format
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # --- NEW: fill non-skin with median skin color (avoid black bias) ---
    skin_pixels = img_cv[skin_mask_resized > 0]  # (N,3) BGR
    if skin_pixels.size > 0:
        median_bgr = np.median(skin_pixels, axis=0).astype(np.uint8)
    else:
        median_bgr = np.array([128, 128, 128], dtype=np.uint8)  # neutral fallback
    bg = np.full_like(img_cv, median_bgr)
    filled = np.where(skin_mask_3ch > 0, img_cv, bg)

    # Back to RGB PIL
    result = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


# ====================================================
# MAIN PREDICT FUNCTION
# ====================================================
def predict_skin_tone(image_pil):
    """Segment → Gemini → Return structured tone + emotion output"""
    face_segmenter = load_face_segmenter()
    segmented = segment_face(image_pil, face_segmenter)

    # Convert segmented image to bytes
    buffer = io.BytesIO()
    segmented.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    # Send to Gemini
    gemini_out = analyze_with_gemini(img_bytes)

    # Extract structured fields
    main_season = gemini_out.get("season")
    confidence = gemini_out.get("confidence")
    subtones = gemini_out.get("top_2_subtones")
    season_confidences = gemini_out.get("season_confidences")
    top_2_seasons = gemini_out.get("top_2_seasons")

    return main_season, confidence, subtones, season_confidences, top_2_seasons, gemini_out
