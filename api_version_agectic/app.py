from helper import make_run_dir, save_pil, save_side_by_side, write_json
from helper import fix_orientation_to_portrait
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
    save_palette_image,
    estimate_emotion_from_palette,
    crop_face_for_emotion,
    is_achromatic,
)
from predictor import predict_skin_tone, _get_face_segmenter  # Import BiSeNet loader

# ==============================
# CONFIG
# ==============================
client_initialized = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion thresholds (Option A)
T_HIGH_LOCAL = 0.80  # trust local when >= this
T_LOW_LOCAL = 0.55  # call Gemini fallback when < this
T_GEM_FINAL = 0.75  # trust Gemini fallback when >= this

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(title="Unified Color Tone + Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# LOAD LOCAL FACIAL-EMOTION MODEL ONCE
# ==============================
# Try ./models first, then ../models (robust to layout)
_model_candidates = [
    "./models/best_densenet121_rafdb.pth",
    "../models/best_densenet121_rafdb.pth",
]
import os

emotion_model_path = next((p for p in _model_candidates if os.path.exists(p)), _model_candidates[0])

emotion_labels = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# --- DenseNet121 Confirmation Print ---
if not os.path.exists(emotion_model_path):
    print(f"❌ ERROR: Emotion model weights not found at: {os.path.abspath(emotion_model_path)}")
else:
    print(f"💡 Attempting to load Emotion Model from: {os.path.basename(emotion_model_path)}")
# --------------------------------------

emotion_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
emotion_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(emotion_model.classifier.in_features, 7),
)

try:
    emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=device))
    emotion_model.eval()
    emotion_model = emotion_model.to(device)
    print("✅ Local Facial-Emotion Model (DenseNet121) loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR loading Emotion Model: {e}")
    emotion_model = None

emotion_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ==============================
# FASTAPI STARTUP EVENT
# ==============================
@app.on_event("startup")
def startup_event():
    # Trigger BiSeNet model load only (DenseNet is loaded module-wide)
    _get_face_segmenter()

    print("\n\n********************************************************")
    print("🚀 UVICORN STARTUP: ALL ACTIVE MODELS INITIALIZED SUCCESSFULLY")
    print("********************************************************\n")


# ==============================
# ROOT
# ==============================
@app.get("/")
def root():
    global client_initialized
    client_initialized = True
    return {"message": "Ready."}


# ==============================
# ONE ENDPOINT TO DO IT ALL
# ==============================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not client_initialized:
        return {"error": "Call GET / first to initialize."}

    try:
        # ---- set up run dir (use original filename stem for readability)
        stem = os.path.splitext(file.filename or "image")[0]
        run_dir = make_run_dir(base_dir="runs", stem=stem)

        # ---- read & save original
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = fix_orientation_to_portrait(image)
        save_pil(image, os.path.join(run_dir, "01_original.jpg"))

        # ---- face crop for emotion; save it
        warnings = []
        if not is_human_face(image):
            cropped = crop_face_for_emotion(image)
            if not is_human_face(cropped):
                warnings.append("Face detector was uncertain; proceeded with best effort.")
                face_for_emotion = cropped
            else:
                face_for_emotion = cropped
        else:
            face_for_emotion = crop_face_for_emotion(image)
        save_pil(face_for_emotion, os.path.join(run_dir, "01a_face_crop.jpg"))

        # ---- Tone via predictor (now includes local/Gemini fallback)
        (
            main_season, confidence, subtones, season_confidences, top_2_seasons, gemini_out,
            seg_filled_pil, stabilized_pil, skin_mask
        ) = predict_skin_tone(image, run_dir=run_dir)

        # ---- normalize tone confidences
        if season_confidences:
            total = sum(season_confidences.values()) or 1.0
            season_confidences = {k: float(v) / total for k, v in season_confidences.items()}
        if season_confidences:
            _sorted = sorted(season_confidences.items(), key=lambda kv: kv[1], reverse=True)
            top_2_seasons = [(k, v) for k, v in _sorted[:2]]

        predicted_label = f"{main_season.capitalize()}_{subtones[0][0].capitalize()}"

        # ---- palette choose + save palette image (you already do)
        palette_hex = gemini_out.get("hex_palette") if isinstance(gemini_out, dict) else None
        if not palette_hex:
            palette_hex = season_tone_palettes.get(predicted_label, [])
        if is_achromatic(palette_hex):
            fallback_key = f"{main_season.capitalize()}_{subtones[0][0].capitalize()}"
            palette_hex = season_tone_palettes.get(fallback_key, palette_hex)

        palette_vec = palette_to_vector(palette_hex)
        palette_path = save_palette_image(palette_hex, predicted_label, file.filename)
        # also copy palette into run_dir with a clean name
        pal_name = f"04_palette_{predicted_label}.png"
        pal_target = os.path.join(run_dir, pal_name)
        Image.open(palette_path).save(pal_target)
        palette_path = pal_target  # prefer the run_dir path

        # ---- color mood
        mood = estimate_emotion_from_palette(hex_palette=palette_hex)

        # ---- local emotion
        input_tensor = emotion_transform(face_for_emotion).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = emotion_model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        expr_idx = int(np.argmax(probs))
        expr_emotion = emotion_labels[expr_idx]
        expr_conf = float(probs[expr_idx])

        print(f"💡 Emotion: Local model result {expr_emotion} (Conf: {expr_conf:.2f})")

        # ---- optional Gemini fallback (FUSION)
        from io import BytesIO
        face_buf = BytesIO()
        face_for_emotion.save(face_buf, format="JPEG", quality=85)
        face_bytes = face_buf.getvalue()

        emotion_source = "local"
        fused = expr_emotion

        # Local-first fusion logic
        if expr_conf >= T_HIGH_LOCAL:
            fused = expr_emotion
            emotion_source = "local"
        else:
            if expr_conf < T_LOW_LOCAL:
                # Fallback to Gemini for better result
                print(f"⚠️ Emotion: Local confidence is low ({expr_conf:.2f}), falling back to Gemini API...")
                try:
                    from gemini_agent import infer_expression_facecrop
                    gemo = infer_expression_facecrop(face_bytes)
                    gem_label = gemo.get("label");
                    gem_conf = float(gemo.get("confidence", 0.0))
                except Exception:
                    gem_label, gem_conf = None, 0.0

                if gem_label and gem_conf >= T_GEM_FINAL:
                    fused = gem_label
                    emotion_source = "gemini_fallback"
                    print(f"✅ Emotion: Fused with Gemini result {gem_label} (Conf: {gem_conf:.2f})")
                else:
                    # If Gemini fallback is also low confidence, stick to the original local model result
                    fused = expr_emotion
                    emotion_source = "local_low_conf"
                    print(f"⚠️ Emotion: Gemini fallback also low/failed. Using original local result.")
            else:
                # Confidence is between T_LOW_LOCAL and T_HIGH_LOCAL, trust local result
                fused = expr_emotion
                emotion_source = "local_med_conf"

        explanation = (
            f"Facial expression: {expr_emotion} (conf {expr_conf:.2f}). "
            f"Palette mood: {mood['color_mood'].lower()}. "
            f"Final emotion: {fused} (source: {emotion_source})."
        )
        if expr_conf < 0.40: warnings.append("Facial emotion is low-confidence.")
        if (mood.get('confidence') or 0) < 0.30: warnings.append("Color mood signal is weak or achromatic.")
        if is_achromatic(palette_hex): warnings.append("Palette was achromatic; used seasonal fallback.")

        # ---- save a side-by-side collage
        compare_path = os.path.join(run_dir, "05_comparison_grid.jpg")
        save_side_by_side(
            images=[image, face_for_emotion, seg_filled_pil, stabilized_pil],
            labels=["Original", "Face Crop", "Segmented (Filled)", "Stabilized (WB+Exposure)"],
            path=compare_path,
            cols=2
        )

        # ---- write a summary JSON
        summary = {
            "tone": {
                "season": main_season,
                "subtype": subtones[0][0],
                "confidence": confidence,
                "season_confidences": season_confidences,
                "top_2_seasons": top_2_seasons,
                "top_2_subtones": subtones,
                "predicted_label": predicted_label,
            },
            "palette": {
                "hex": palette_hex,
                "rgb_vector_15d": palette_vec.tolist(),
                "image_path": palette_path,
            },
            "emotion": {
                "facial_expression": expr_emotion,
                "facial_expression_confidence": expr_conf,
                "color_vibe": mood["color_mood"],
                "color_vibe_confidence": mood["confidence"],
                "final_interpretation": fused,
                "emotion_source": emotion_source,
            },
            "reasoning": (gemini_out.get("reasoning") if isinstance(gemini_out, dict) else None),
            "explanation": explanation,
            "warnings": warnings,
            "artifacts": {
                "original": os.path.join(run_dir, "01_original.jpg"),
                "face_crop": os.path.join(run_dir, "01a_face_crop.jpg"),
                "segmented_filled": os.path.join(run_dir, "02_segmented_filled.jpg"),
                "skin_mask": os.path.join(run_dir, "02a_skin_mask.png"),
                "stabilized_for_gemini": os.path.join(run_dir, "03_stabilized_for_gemini.jpg"),
                "palette_image": palette_path,
                "comparison_grid": compare_path,
                "run_dir": run_dir
            }
        }
        summary_path = os.path.join(run_dir, "result_summary.json")
        write_json(summary, summary_path)

        # ---- normal API response + debug paths
        return summary

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}