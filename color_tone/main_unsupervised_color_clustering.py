import os
import shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === Paths ===
input_dir = "dataset/all_faces"
output_dir = "clustered_seasons_kmeans"
os.makedirs(output_dir, exist_ok=True)

# === Load CLIP model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# === Cluster (KMeans) ===
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(scaled)

# === Save clustered images ===
for cluster_id in range(4):
    os.makedirs(os.path.join(output_dir, f"Cluster_{cluster_id}"), exist_ok=True)

for fname, label in zip(filenames, labels):
    src = os.path.join(input_dir, fname)
    dst = os.path.join(output_dir, f"Cluster_{label}", fname)
    shutil.copyfile(src, dst)

print("✅ Clustering completed. Check the folders under:", output_dir)
