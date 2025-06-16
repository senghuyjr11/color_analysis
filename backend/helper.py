# helper.py
import numpy as np
from PIL import Image
from deepface import DeepFace

def is_human_face(image: Image.Image) -> bool:
    try:
        img_array = np.array(image)
        results = DeepFace.extract_faces(img_array, enforce_detection=True)
        print(f"[DeepFace] Detected {len(results)} face(s).")
        return len(results) > 0
    except Exception as e:
        print(f"[DeepFace ERROR] {e}")
        return False
