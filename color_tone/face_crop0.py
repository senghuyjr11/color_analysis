import os
import cv2
import mediapipe as mp
from rembg import remove, new_session
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
import numpy as np

# === Paths ===
input_dir = "fake_dataset/00000"
output_dir = "fake_dataset/cropped_faces"
os.makedirs(output_dir, exist_ok=True)
batch_size = 8

# === MediaPipe Face Detection (always CPU) ===
mp_face_detection = mp.solutions.face_detection

# === Auto GPU session selection for rembg ===
ort.set_default_logger_severity(3)  # Suppress ONNX spam
available_providers = ort.get_available_providers()

use_gpu = 'CUDAExecutionProvider' in available_providers
print(f"rembg will use: {'GPU' if use_gpu else 'CPU'}")

session = new_session("u2net", providers=["CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"])

# === Helper: Crop face using MediaPipe
def crop_face(image):
    height, width, _ = image.shape
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.detections:
        d = results.detections[0]
        bbox = d.location_data.relative_bounding_box
        x, y, w, h = bbox.xmin * width, bbox.ymin * height, bbox.width * width, bbox.height * height
        pad = 0.2
        x1 = max(int(x - w * pad), 0)
        y1 = max(int(y - h * pad), 0)
        x2 = min(int(x + w * (1 + pad)), width)
        y2 = min(int(y + h * (1 + pad)), height)
        return image[y1:y2, x1:x2]
    return None

# === Main Loop ===
file_list = os.listdir(input_dir)
images_batch = []
names_batch = []

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
    for fname in tqdm(file_list, desc="🔄 Batching for rembg", ncols=80):
        try:
            path = os.path.join(input_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            face_crop = crop_face(img)
            if face_crop is None:
                tqdm.write(f"❌ No face detected in {fname}")
                continue

            pil_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            images_batch.append(pil_image)
            names_batch.append(fname)

            # === Process batch when full ===
            if len(images_batch) == batch_size:
                for img, name in zip(images_batch, names_batch):
                    out_img = remove(img, session=session)
                    save_path = os.path.join(output_dir, os.path.splitext(name)[0] + ".png")
                    out_img.save(save_path)
                images_batch.clear()
                names_batch.clear()

        except Exception as e:
            tqdm.write(f"⚠️ Error on {fname}: {e}")

    # === Final leftover batch
    if images_batch:
        for img, name in zip(images_batch, names_batch):
            out_img = remove(img, session=session)
            save_path = os.path.join(output_dir, os.path.splitext(name)[0] + ".png")
            out_img.save(save_path)

