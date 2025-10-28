import io
import os
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from helper import (
    is_human_face,
    season_tone_palettes,
    palette_to_vector,
    save_palette_image
)
from predictor import predict_skin_tone, predict_emotion, analyze_full

# === Config ===
client_initialized = False

# Create directories for debugging artifacts
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

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
    """
    Skin tone prediction with comprehensive preprocessing:
    - Face segmentation (BiSeNet + DeepFace fallback)
    - Color stabilization (AWB + exposure normalization)
    - ConvNeXt model prediction
    """
    if not client_initialized:
        return {"error": "Please call the root endpoint (/) before using this API."}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print("\n" + "=" * 70)
        print(f"Processing: {file.filename}")
        print("=" * 70)

        # Face detection check
        print("\n[INFO] Checking for human face...")
        if not is_human_face(image):
            print("[ERROR] No face detected.\n")
            return {
                "error": "No human face detected in the image. Please upload a clear face photo."
            }
        print("[INFO] Face detected successfully.\n")

        # Create run directory for this prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RUNS_DIR, f"skin_tone_{timestamp}_{file.filename.split('.')[0]}")
        os.makedirs(run_dir, exist_ok=True)

        # Save original image
        image.save(os.path.join(run_dir, "01_original.jpg"))
        print(f"[INFO] Run directory: {run_dir}\n")

        # Call prediction with full preprocessing pipeline
        (
            main_season, confidence, top_2_subtones, season_confidences,
            top_2_seasons, gemini_out, filled_pil, stabilized_pil, skin_mask
        ) = predict_skin_tone(image, run_dir=run_dir)

        # Generate label and get palette
        predicted_label = f"{main_season.capitalize()}_{top_2_subtones[0][0].capitalize()}"
        palette_hex = season_tone_palettes.get(predicted_label, [])
        palette_vec = palette_to_vector(palette_hex)
        palette_path = save_palette_image(palette_hex, predicted_label, file.filename, run_dir)

        result = {
            "predicted_label": predicted_label,
            "season": main_season,
            "subtype": top_2_subtones[0][0],
            "confidence": confidence,
            "top_2_subtones": top_2_subtones,
            "season_confidences": season_confidences,
            "top_2_seasons": top_2_seasons,
            "hex_palette": palette_hex,
            "rgb_vector_15d": palette_vec.tolist(),
            "palette_image_path": palette_path,
            "run_directory": run_dir,
            "preprocessing": {
                "face_segmentation": "BiSeNet (skin + neck)",
                "color_stabilization": "AWB + exposure normalization",
                "model": "ConvNeXt Large"
            }
        }

        print("\n" + "=" * 70)
        print("API Response Ready")
        print("=" * 70 + "\n")

        return result

    except Exception as e:
        print(f"\n[ERROR] {str(e)}\n")
        return {"error": str(e)}


@app.post("/predict_emotion")
async def predict_emotion_endpoint(file: UploadFile = File(...)):
    """
    Emotion prediction with face cropping preprocessing:
    - DeepFace face detection and cropping
    - DenseNet121 emotion classification
    """
    if not client_initialized:
        return {"error": "Please call the root endpoint (/) before using this API."}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print("\n" + "=" * 70)
        print(f"Processing: {file.filename}")
        print("=" * 70)

        # Face detection check
        print("\n[INFO] Checking for human face...")
        if not is_human_face(image):
            print("[ERROR] No face detected.\n")
            return {"error": "No human face detected in the image."}
        print("[INFO] Face detected successfully.\n")

        # Create run directory for this prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RUNS_DIR, f"emotion_{timestamp}_{file.filename.split('.')[0]}")
        os.makedirs(run_dir, exist_ok=True)

        # Save original image
        image.save(os.path.join(run_dir, "01_original.jpg"))
        print(f"[INFO] Run directory: {run_dir}\n")

        # Call prediction with face cropping
        emotion_result = predict_emotion(image, run_dir=run_dir)

        emotion_result["run_directory"] = run_dir
        emotion_result["preprocessing"] = {
            "face_detection": "DeepFace RetinaFace",
            "face_cropping": "Tight crop around detected face",
            "model": "DenseNet121 (RAF-DB)"
        }

        print("\n" + "=" * 70)
        print("API Response Ready")
        print("=" * 70 + "\n")

        return emotion_result

    except Exception as e:
        print(f"\n[ERROR] {str(e)}\n")
        return {"error": f"Emotion prediction failed: {str(e)}"}


@app.post("/analyze_full")
async def analyze_full_endpoint(file: UploadFile = File(...)):
    """
    Complete analysis: skin tone + emotion in one call.
    Performs all preprocessing and saves all intermediate images.
    """
    if not client_initialized:
        return {"error": "Please call the root endpoint (/) before using this API."}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print("\n" + "=" * 70)
        print(f"Full Analysis: {file.filename}")
        print("=" * 70)

        # Face detection check
        print("\n[INFO] Checking for human face...")
        if not is_human_face(image):
            print("[ERROR] No face detected.\n")
            return {
                "error": "No human face detected in the image. Please upload a clear face photo."
            }
        print("[INFO] Face detected successfully.\n")

        # Create run directory for this full analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RUNS_DIR, f"full_{timestamp}_{file.filename.split('.')[0]}")
        os.makedirs(run_dir, exist_ok=True)

        # Save original image
        image.save(os.path.join(run_dir, "01_original.jpg"))
        print(f"[INFO] Run directory: {run_dir}\n")

        # Call full analysis pipeline
        (
            main_season, confidence, top_2_subtones, season_confidences,
            top_2_seasons, gemini_out, filled_pil, stabilized_pil, skin_mask,
            emotion_result
        ) = analyze_full(image, run_dir=run_dir)

        # Generate skin tone label and palette
        predicted_label = f"{main_season.capitalize()}_{top_2_subtones[0][0].capitalize()}"
        palette_hex = season_tone_palettes.get(predicted_label, [])
        palette_vec = palette_to_vector(palette_hex)
        palette_path = save_palette_image(palette_hex, predicted_label, file.filename, run_dir)

        result = {
            "skin_tone": {
                "predicted_label": predicted_label,
                "season": main_season,
                "subtype": top_2_subtones[0][0],
                "confidence": confidence,
                "top_2_subtones": top_2_subtones,
                "season_confidences": season_confidences,
                "top_2_seasons": top_2_seasons,
                "hex_palette": palette_hex,
                "rgb_vector_15d": palette_vec.tolist(),
                "palette_image_path": palette_path,
            },
            "emotion": emotion_result,
            "run_directory": run_dir,
            "preprocessing": {
                "face_segmentation": "BiSeNet (skin + neck)",
                "color_stabilization": "AWB + exposure normalization",
                "face_cropping": "DeepFace RetinaFace",
                "skin_tone_model": "ConvNeXt Large",
                "emotion_model": "DenseNet121 (RAF-DB)"
            },
            "intermediate_images": {
                "original": "01_original.jpg",
                "segmented": "02_segmented_filled.jpg",
                "skin_mask": "02a_skin_mask.png",
                "stabilized": "03_stabilized_for_model.jpg",
                "cropped_face": "04_cropped_face_for_emotion.jpg"
            }
        }

        print("\n" + "=" * 70)
        print("Full Analysis Complete - API Response Ready")
        print("=" * 70 + "\n")

        return result

    except Exception as e:
        print(f"\n[ERROR] {str(e)}\n")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print(" " * 15 + "Starting Skin Tone & Emotion API Server")
    print("=" * 70)
    print("\nPreprocessing Features:")
    print("  ✓ BiSeNet face segmentation (skin + neck)")
    print("  ✓ DeepFace fallback for failed segmentation")
    print("  ✓ AWB (Auto White Balance)")
    print("  ✓ Exposure normalization (target V=0.55)")
    print("  ✓ Face detection and cropping for emotion")
    print("\nModels:")
    print("  ✓ ConvNeXt Large (skin tone)")
    print("  ✓ DenseNet121 (emotion)")
    print("\nEndpoints:")
    print("  • GET  /              - Initialize client")
    print("  • POST /predict_skin_tone  - Skin tone only")
    print("  • POST /predict_emotion    - Emotion only")
    print("  • POST /analyze_full       - Both analyses")
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)