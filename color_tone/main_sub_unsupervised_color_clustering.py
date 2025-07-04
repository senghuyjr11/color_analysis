import os
import shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === Settings ===
input_dir = "../clustered_seasons_kmeans/Winter"
output_base = "clustered_seasons_kmeans/Winter"
n_subclusters = 3

# === Load CLIP model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Extract features ===
features = []
filenames = []

for fname in tqdm(os.listdir(input_dir)):
    fpath = os.path.join(input_dir, fname)
    try:
        image = Image.open(fpath).convert("RGB").resize((224, 224))
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        features.append(emb.cpu().numpy().flatten())
        filenames.append(fname)
    except Exception as e:
        print(f"Skipped {fname}: {e}")

features = np.vstack(features)
scaled = StandardScaler().fit_transform(features)

# === KMeans Sub-clustering ===
kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
labels = kmeans.fit_predict(scaled)

# === Save to subfolders ===
for sub_id in range(n_subclusters):
    os.makedirs(os.path.join(output_base, f"Sub_{sub_id}"), exist_ok=True)

for fname, label in zip(filenames, labels):
    src = os.path.join(input_dir, fname)
    dst = os.path.join(output_base, f"Sub_{label}", fname)
    shutil.move(src, dst)  # Move instead of copy to avoid duplication

print("Winter sub-clustering complete.")
