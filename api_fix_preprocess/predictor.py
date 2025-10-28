import io
import os
import random
import sys

from helper import save_pil, save_np_mask, fix_orientation_to_portrait, crop_face_for_emotion
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import convnext_large, densenet121, DenseNet121_Weights

from helper import gentle_brighten_underexposed, pil_from_rgb

# ---------------- Determinism ----------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# Face parsing import & paths
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from face_parsing.model import BiSeNet  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "models")
WEIGHTS = os.path.join(PROJECT_ROOT, "..", "face_parsing", "res", "cp", "79999_iter.pth")

# =========================
# ConvNeXt Model Configurations
# =========================
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

# =========================
# Emotion Model Configuration
# =========================
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "best_densenet121_rafdb.pth")
EMOTION_LABELS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# segmentation transform
to_tensor = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ConvNeXt transform
convnext_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Emotion transform
emotion_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------- load face segmenter once (global) ----------
_FACE_SEGMENTER = None


def _get_face_segmenter():
    global _FACE_SEGMENTER
    if _FACE_SEGMENTER is None:
        if not os.path.exists(WEIGHTS):
            raise FileNotFoundError(f"BiSeNet weights not found: {os.path.abspath(WEIGHTS)}")
        model = BiSeNet(n_classes=19)
        model.to(DEVICE)
        model.load_state_dict(torch.load(os.path.abspath(WEIGHTS), map_location=DEVICE))
        model.eval()
        _FACE_SEGMENTER = model
    return _FACE_SEGMENTER


# --------- load emotion model once (global) ----------
_EMOTION_MODEL = None


def _get_emotion_model():
    global _EMOTION_MODEL
    if _EMOTION_MODEL is None:
        if not os.path.exists(EMOTION_MODEL_PATH):
            raise FileNotFoundError(f"Emotion model not found: {os.path.abspath(EMOTION_MODEL_PATH)}")
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier.in_features, 7)
        )
        model.load_state_dict(torch.load(os.path.abspath(EMOTION_MODEL_PATH), map_location=DEVICE))
        model.eval()
        _EMOTION_MODEL = model.to(DEVICE)
    return _EMOTION_MODEL


def _morph_close(mask, k=9, iters=1):
    """Morphological closing to fill holes in mask."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=iters)


def _ellipse_fallback_mask(image_pil: Image.Image) -> np.ndarray:
    """
    If parsing fails, build a soft elliptical mask from the detected face box.
    Uses DeepFace RetinaFace for robust face detection.
    """
    from deepface import DeepFace

    rgb = np.array(image_pil)
    H, W = rgb.shape[:2]
    dets = []
    try:
        print("[INFO] Parsing failed, attempting DeepFace fallback mask...")
        dets = DeepFace.extract_faces(rgb, enforce_detection=False, detector_backend="retinaface")
    except Exception as e:
        print(f"[WARNING] DeepFace fallback failed: {e}")
        pass

    mask = np.zeros((H, W), np.uint8)
    if not dets:
        return mask  # no face → return empty (caller will handle)

    fa = dets[0].get("facial_area") or {}
    x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
    # Expand bounding box slightly
    x = max(0, x - int(0.08 * w));
    y = max(0, y - int(0.08 * h))
    w = int(w * 1.16);
    h = int(h * 1.12)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x or y2 <= y:
        return mask

    # draw ellipse inside the face bbox
    cx, cy = (x + x2) // 2, (y + y2) // 2
    ax, ay = max(1, (x2 - x) // 2), max(1, (y2 - y) // 2)
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    print(f"[INFO] Created fallback ellipse mask: center=({cx},{cy}), axes=({ax},{ay})")
    return ellipse


def segment_face(image_pil: Image.Image):
    """
    Robust skin segmentation with comprehensive preprocessing:
      1. Fix image orientation (portrait mode)
      2. BiSeNet parsing for skin (1) + neck (14) regions
      3. Morphological closing to fill holes
      4. Area sanity check with DeepFace fallback if parsing fails
      5. Fill non-skin regions with median skin color for consistency

    Returns:
      filled_rgb_pil: Image with non-skin filled with median skin color
      skin_mask: Binary mask of skin regions (uint8, 0 or 255)
    """
    print("[INFO] Starting face segmentation...")

    # 1) Fix orientation to portrait
    image_pil = fix_orientation_to_portrait(image_pil)
    print(f"[INFO] Image size after orientation fix: {image_pil.size}")

    model = _get_face_segmenter()

    # 2) Run BiSeNet parsing at 512x512
    original_size = image_pil.size  # (w, h)
    img_resized = image_pil.resize((512, 512), Image.BILINEAR)
    tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)  # 512x512

    # 3) Extract skin classes: skin(1) + neck(14) for broader, more stable mask
    skin_512 = np.isin(parsing, [1, 14]).astype(np.uint8) * 255
    print(f"[INFO] BiSeNet parsing - skin pixels at 512x512: {(skin_512 > 0).sum()}")

    # 4) Upsample to original size and apply morphological closing
    skin_mask = cv2.resize(skin_512, original_size, interpolation=cv2.INTER_NEAREST)
    skin_mask = _morph_close(skin_mask, k=11, iters=1)
    skin_mask_3ch = cv2.merge([skin_mask] * 3)

    print(f"[INFO] Skin mask after morphology - skin pixels: {(skin_mask > 0).sum()}")

    # 5) Sanity check: if mask is too small (<2% of image), use DeepFace fallback
    area_ratio = float((skin_mask > 0).sum()) / (skin_mask.size + 1e-6)
    print(f"[INFO] Skin area ratio: {area_ratio:.4f} ({area_ratio * 100:.2f}%)")

    if area_ratio < 0.02:  # <2% of pixels → almost certainly a failure
        print("[WARNING] Skin mask too small, using DeepFace ellipse fallback...")
        fb = _ellipse_fallback_mask(image_pil)
        if fb.sum() > skin_mask.sum():  # only replace if better
            skin_mask = fb
            skin_mask_3ch = cv2.merge([skin_mask] * 3)
            print(f"[INFO] Using fallback mask - skin pixels: {(skin_mask > 0).sum()}")

    # 6) Fill non-skin regions with median skin color
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    skin_pixels = img_cv[skin_mask > 0]

    if skin_pixels.size > 0:
        median_bgr = np.median(skin_pixels, axis=0).astype(np.uint8)
        print(f"[INFO] Median skin color (BGR): {median_bgr}")
    else:
        median_bgr = np.array([128, 128, 128], dtype=np.uint8)
        print("[WARNING] No skin pixels found, using gray as median")

    bg = np.full_like(img_cv, median_bgr)
    filled = np.where(skin_mask_3ch > 0, img_cv, bg)

    filled_rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    print("[INFO] Face segmentation complete.\n")
    return Image.fromarray(filled_rgb), skin_mask


def load_convnext_model(model_path, num_classes):
    """Load a ConvNeXt Large model with custom number of classes."""
    model = convnext_large(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def predict_skin_tone(image_pil: Image.Image, run_dir: str | None = None):
    """
    Complete skin tone analysis pipeline with comprehensive preprocessing:

    1. Face segmentation (BiSeNet + DeepFace fallback)
    2. AWB (Auto White Balance) + exposure normalization on skin midtones
    3. Main season classification (spring/summer/autumn/winter)
    4. Subtone classification based on predicted season

    Args:
        image_pil: Input PIL Image
        run_dir: Optional directory to save intermediate images for debugging

    Returns:
        Tuple of (main_season, confidence, top_2_subtones, season_confidences,
                  top_2_seasons, gemini_out, filled_pil, stabilized_pil, skin_mask)
    """
    print("\n" + "=" * 60)
    print("SKIN TONE ANALYSIS PIPELINE")
    print("=" * 60 + "\n")

    # STEP 1: Face segmentation with robust preprocessing
    print("STEP 1: Face Segmentation")
    print("-" * 40)
    filled_pil, skin_mask = segment_face(image_pil)

    if run_dir:
        save_pil(filled_pil, os.path.join(run_dir, "02_segmented_filled.jpg"))
        save_np_mask(skin_mask, os.path.join(run_dir, "02a_skin_mask.png"))
        print(f"[SAVED] Segmented image: 02_segmented_filled.jpg")
        print(f"[SAVED] Skin mask: 02a_skin_mask.png\n")

    # STEP 2: Gentle Brighten (underexposed-only)
    print("STEP 2: Gentle Brighten (underexposed-only)")
    print("-" * 40)

    filled_np = np.array(filled_pil)  # RGB uint8

    bright_np = gentle_brighten_underexposed(
        filled_np,
        skin_mask,
        low_thresh=0.46,  # consider “dark” if skin midtone V < 0.46 (was 0.42)
        target_V=0.52,  # nudge a bit higher so change is noticeable
        gamma_floor=0.70,  # allow real brightening (0.65–0.80 works well)
        exp_strength=0.40,  # a touch stronger blend on dark shots
        shadow_power=1.40,  # protect highlights more; push lift into shadows
        min_pixels=150  # don’t bail too early when the face is small/dim
    )

    stabilized_pil = pil_from_rgb(bright_np)

    if run_dir:
        save_pil(stabilized_pil, os.path.join(run_dir, "03_brightened_for_model.jpg"))
        print(f"[SAVED] Brightened image: 03_brightened_for_model.jpg\n")

    # STEP 3: Main season prediction
    print("STEP 3: Main Season Classification")
    print("-" * 40)
    print("[INFO] Loading ConvNeXt Large model for main season...")
    main_model = load_convnext_model(MODELS["main_season"]["path"], 4)
    input_tensor = convnext_transform(stabilized_pil).unsqueeze(0).to(DEVICE)

    print("[INFO] Running inference...")
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

    print(f"[RESULT] Predicted Season: {main_season.upper()}")
    print(f"[RESULT] Confidence: {confidence:.2%}")
    print(f"[RESULT] All confidences: {season_confidences}\n")

    # STEP 4: Subtone prediction
    print("STEP 4: Subtone Classification")
    print("-" * 40)
    print(f"[INFO] Loading subtone model for {main_season}...")
    subtone_config = MODELS[f"subtone_{main_season}"]
    subtone_model = load_convnext_model(subtone_config["path"], len(subtone_config["labels"]))

    print("[INFO] Running inference...")
    with torch.no_grad():
        logits = subtone_model(input_tensor)
        subtone_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_subtone_indices = np.argsort(subtone_probs)[::-1][:2]
    top_2_subtones = [(subtone_config["labels"][i], float(subtone_probs[i])) for i in top_subtone_indices]

    print(f"[RESULT] Top Subtones:")
    for subtone, conf in top_2_subtones:
        print(f"  - {subtone}: {conf:.2%}")

    # Output summary
    gemini_out = {
        "hex_palette": None,
        "reasoning": f"Local ConvNeXt prediction: {main_season}_{top_2_subtones[0][0]} ({confidence:.2%})"
    }

    print("\n" + "=" * 60)
    print(f"FINAL PREDICTION: {main_season.upper()}_{top_2_subtones[0][0].upper()}")
    print("=" * 60 + "\n")

    return (
        main_season, confidence, top_2_subtones, season_confidences, top_2_seasons, gemini_out,
        filled_pil, stabilized_pil, skin_mask
    )


def predict_emotion(image_pil: Image.Image, run_dir: str | None = None):
    """
    Emotion detection with face cropping preprocessing:

    1. Crop face tightly using DeepFace
    2. Run through DenseNet121 emotion classifier
    3. Return emotion with confidence scores

    Args:
        image_pil: Input PIL Image
        run_dir: Optional directory to save intermediate images

    Returns:
        Dict with emotion, confidence, reasoning, class_probabilities, warnings
    """
    print("\n" + "=" * 60)
    print("EMOTION DETECTION PIPELINE")
    print("=" * 60 + "\n")

    # STEP 1: Crop face for emotion
    print("STEP 1: Face Cropping")
    print("-" * 40)
    print("[INFO] Detecting and cropping face for emotion analysis...")
    cropped_face_pil = crop_face_for_emotion(image_pil)

    # Check if face detection succeeded
    is_full_image = cropped_face_pil.size == image_pil.size
    if is_full_image:
        print("[WARNING] Could not detect face, using full image")
    else:
        print(f"[INFO] Face cropped to size: {cropped_face_pil.size}")

    # Save cropped face
    if run_dir:
        save_pil(cropped_face_pil, os.path.join(run_dir, "04_cropped_face_for_emotion.jpg"))
        print(f"[SAVED] Cropped face: 04_cropped_face_for_emotion.jpg\n")

    # STEP 2: Emotion classification
    print("STEP 2: Emotion Classification")
    print("-" * 40)
    try:
        print("[INFO] Loading DenseNet121 emotion model...")
        emotion_model = _get_emotion_model()
        input_tensor = emotion_transform(cropped_face_pil).unsqueeze(0).to(DEVICE)

        print("[INFO] Running inference...")
        with torch.no_grad():
            logits = emotion_model(input_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_emotion = EMOTION_LABELS[pred_idx]
            confidence = float(probs[pred_idx])

        # Create class probabilities dict
        class_probs = dict(zip(EMOTION_LABELS, map(float, probs)))

        print(f"[RESULT] Predicted Emotion: {pred_emotion.upper()}")
        print(f"[RESULT] Confidence: {confidence:.2%}")
        print(f"[RESULT] All probabilities:")
        for emotion, prob in class_probs.items():
            print(f"  - {emotion}: {prob:.2%}")

        warnings = []
        if is_full_image:
            warnings.append("Could not tightly crop face; analyzing entire image.")

        print("\n" + "=" * 60)
        print(f"FINAL EMOTION: {pred_emotion.upper()}")
        print("=" * 60 + "\n")

        return {
            "emotion": pred_emotion,
            "confidence": confidence,
            "reasoning": f"DenseNet121 prediction with {confidence:.2%} confidence",
            "class_probabilities": class_probs,
            "warnings": warnings
        }
    except Exception as e:
        print(f"[ERROR] Emotion prediction failed: {str(e)}\n")
        return {
            "emotion": "error",
            "confidence": 0.0,
            "reasoning": f"Model Error: {str(e)}",
            "class_probabilities": {},
            "warnings": ["Failed to classify emotion."]
        }


def analyze_full(image_pil: Image.Image, run_dir: str | None = None):
    """
    Complete analysis pipeline: skin tone + emotion detection.

    Performs comprehensive preprocessing and saves all intermediate images:
    - 02_segmented_filled.jpg: Face with skin segmentation
    - 02a_skin_mask.png: Binary skin mask
    - 03_stabilized_for_model.jpg: Color-corrected image for skin tone model
    - 04_cropped_face_for_emotion.jpg: Cropped face for emotion model

    Args:
        image_pil: Input PIL Image
        run_dir: Directory to save intermediate debugging images

    Returns:
        Tuple of all results for both skin tone and emotion analysis
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "FULL ANALYSIS PIPELINE")
    print("=" * 70 + "\n")

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        print(f"[INFO] Saving intermediate images to: {run_dir}\n")

    # 1. Skin Tone Analysis with full preprocessing
    (
        main_season, confidence, subtones, season_confidences, top_2_seasons, gemini_tone_out,
        filled_pil, stabilized_pil, skin_mask
    ) = predict_skin_tone(image_pil, run_dir=run_dir)

    # 2. Emotion Detection with face cropping
    emotion_result = predict_emotion(image_pil, run_dir=run_dir)

    print("=" * 70)
    print(" " * 25 + "ANALYSIS COMPLETE")
    print("=" * 70 + "\n")

    # Return all results and intermediates
    return (
        main_season, confidence, subtones, season_confidences, top_2_seasons, gemini_tone_out,
        filled_pil, stabilized_pil, skin_mask,
        emotion_result
    )