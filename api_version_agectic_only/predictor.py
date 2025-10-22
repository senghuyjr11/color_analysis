import io
import os
import random
import sys

from helper import save_pil, save_np_mask, fix_orientation_to_portrait, crop_face_for_emotion
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from helper import stabilize_skin_only_rgb, pil_from_rgb

# ---------------- Determinism ----------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# Face parsing import & paths (as requested)
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from face_parsing.model import BiSeNet  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "models")
WEIGHTS = os.path.join(PROJECT_ROOT, "..", "face_parsing", "res", "cp", "79999_iter.pth")

# ---------------- Gemini tone-only API ----------------
from gemini_agent import analyze_tone_skin_only, classify_emotion_from_face  # assumes gemini_agent.py is importable

# segmentation transform
to_tensor = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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

def _morph_close(mask, k=9, iters=1):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=iters)

def _ellipse_fallback_mask(image_pil: Image.Image) -> np.ndarray:
    """If parsing fails, build a soft elliptical mask from the detected face box."""
    import numpy as np
    import cv2
    from deepface import DeepFace

    rgb = np.array(image_pil)
    H, W = rgb.shape[:2]
    dets = []
    try:
        dets = DeepFace.extract_faces(rgb, enforce_detection=False, detector_backend="retinaface")
    except Exception:
        pass

    mask = np.zeros((H, W), np.uint8)
    if not dets:
        return mask  # no face → return empty (caller will handle)

    fa = dets[0].get("facial_area") or {}
    x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
    x = max(0, x - int(0.08 * w)); y = max(0, y - int(0.08 * h))
    w = int(w * 1.16); h = int(h * 1.12)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x or y2 <= y:
        return mask

    # draw ellipse inside the face bbox
    cx, cy = (x + x2) // 2, (y + y2) // 2
    ax, ay = max(1, (x2 - x) // 2), max(1, (y2 - y) // 2)
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return ellipse

def segment_face(image_pil: Image.Image):
    """
    Robust skin mask:
      - BiSeNet parsing
      - use labels {1:skin, 14:neck}  (CelebAMask-HQ)
      - morphology to fill holes
      - sanity check on area; fallback to ellipse mask from face box if needed
    Returns:
      filled_rgb_pil, skin_mask_uint8
    """
    image_pil = fix_orientation_to_portrait(image_pil)
    model = _get_face_segmenter()

    # 1) run parsing at 512
    original_size = image_pil.size  # (w, h)
    img_resized = image_pil.resize((512, 512), Image.BILINEAR)
    tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)  # 512x512

    # 2) skin classes: skin(1) + neck(14)  (broader, more stable)
    skin_512 = np.isin(parsing, [1, 14]).astype(np.uint8) * 255

    # 3) upsample to original and clean
    skin_mask = cv2.resize(skin_512, original_size, interpolation=cv2.INTER_NEAREST)
    skin_mask = _morph_close(skin_mask, k=11, iters=1)
    skin_mask_3ch = cv2.merge([skin_mask] * 3)

    # 4) sanity check (too small → fallback ellipse)
    area_ratio = float((skin_mask > 0).sum()) / (skin_mask.size + 1e-6)
    if area_ratio < 0.02:  # <2% of pixels → almost certainly a failure
        fb = _ellipse_fallback_mask(image_pil)
        if fb.sum() > skin_mask.sum():  # only replace if better
            skin_mask = fb
            skin_mask_3ch = cv2.merge([skin_mask] * 3)

    # 5) fill non-skin with median skin color
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    skin_pixels = img_cv[skin_mask > 0]
    if skin_pixels.size > 0:
        median_bgr = np.median(skin_pixels, axis=0).astype(np.uint8)
    else:
        median_bgr = np.array([128, 128, 128], dtype=np.uint8)
    bg = np.full_like(img_cv, median_bgr)
    filled = np.where(skin_mask_3ch > 0, img_cv, bg)

    filled_rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    return Image.fromarray(filled_rgb), skin_mask



def predict_skin_tone(image_pil: Image.Image, run_dir: str | None = None):
    # 1) segmentation
    filled_pil, skin_mask = segment_face(image_pil)
    if run_dir:
        save_pil(filled_pil, os.path.join(run_dir, "02_segmented_filled.jpg"))
        save_np_mask(skin_mask, os.path.join(run_dir, "02a_skin_mask.png"))

    # 2) AWB + exposure normalization on skin midtones
    filled_np = np.array(filled_pil)
    stabilized_np = stabilize_skin_only_rgb(filled_np, skin_mask, v_range=(0.15, 0.85), s_min=0.10, target_V=0.55)
    stabilized_pil = pil_from_rgb(stabilized_np)
    if run_dir:
        save_pil(stabilized_pil, os.path.join(run_dir, "03_stabilized_for_gemini.jpg"))

    # 3) JPEG bytes for Gemini
    buf = io.BytesIO()
    stabilized_pil.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    # 4) Gemini tone-only call
    tone_out = analyze_tone_skin_only(img_bytes)

    # 5) map outputs (same as before) ...
    main_season = tone_out.get("season", "spring")
    confidence = float(tone_out.get("confidence", 0.0))
    top_2_subtones = []
    for it in tone_out.get("top_2_subtones", []) or []:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            top_2_subtones.append((str(it[0]), float(it[1])))
    sc = tone_out.get("season_confidences", {}) or {}
    season_confidences = {str(k): float(v) for k, v in sc.items()}
    top_2_seasons = []
    for it in tone_out.get("top_2_seasons", []) or []:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            top_2_seasons.append((str(it[0]), float(it[1])))

    gemini_out = {
        "hex_palette": tone_out.get("hex_palette"),
        "reasoning": tone_out.get("reasoning"),
    }

    return (
        main_season, confidence, top_2_subtones, season_confidences, top_2_seasons, gemini_out,
        filled_pil, stabilized_pil, skin_mask  # <- return intermediates for plotting
    )


def predict_emotion(image_pil: Image.Image, run_dir: str | None = None):
    """
    Detect emotion from the face in the image using a Gemini agent.
    1) Crop face tightly.
    2) Convert to JPEG bytes.
    3) Call Gemini emotion classification.
    """
    # 1) Crop face
    cropped_face_pil = crop_face_for_emotion(image_pil)

    # If the cropped image is exactly the original size, face detection failed.
    # While Gemini might still classify, we might log a warning or use a default.
    is_full_image = cropped_face_pil.size == image_pil.size

    # 2) JPEG bytes for Gemini
    buf = io.BytesIO()
    # Use a lower quality for speed and smaller payload, as minor artifacts are okay for emotion
    cropped_face_pil.save(buf, format="JPEG", quality=75)
    img_bytes = buf.getvalue()

    # Save artifact
    if run_dir:
        save_pil(cropped_face_pil, os.path.join(run_dir, "04_cropped_face_for_emotion.jpg"))

    # 3) Gemini emotion classification call
    try:
        emotion_out = classify_emotion_from_face(img_bytes)

        # 4) Map outputs
        emotion = str(emotion_out.get("emotion", "neutral"))
        confidence = float(emotion_out.get("confidence", 0.0))
        reasoning = str(emotion_out.get("reasoning", ""))

        warnings = []
        if is_full_image:
            warnings.append("Could not tightly crop face; analyzing entire image.")

        return {
            "emotion": emotion,
            "confidence": confidence,
            "reasoning": reasoning,
            "warnings": warnings
        }
    except Exception as e:
        return {
            "emotion": "error",
            "confidence": 0.0,
            "reasoning": f"Gemini/API Error: {str(e)}",
            "warnings": ["Failed to classify emotion."]
        }



def analyze_full(image_pil: Image.Image, run_dir: str | None = None):
    """
    Performs both skin tone analysis and emotion detection, saving all artifacts
    to the same run directory.
    """
    # 1. Skin Tone Analysis (Uses 02_segmented_filled.jpg and 03_stabilized_for_gemini.jpg)
    (
        main_season, confidence, subtones, season_confidences, top_2_seasons, gemini_tone_out,
        filled_pil, stabilized_pil, skin_mask
    ) = predict_skin_tone(image_pil, run_dir=run_dir)

    # 2. Emotion Detection (Uses 04_cropped_face_for_emotion.jpg)
    emotion_result = predict_emotion(image_pil, run_dir=run_dir)

    # Return all results and intermediates for the main.py formatting
    return (
        main_season, confidence, subtones, season_confidences, top_2_seasons, gemini_tone_out,
        filled_pil, stabilized_pil, skin_mask,
        emotion_result
    )