import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import itertools
from tqdm import tqdm

# === Feature Extraction ===
def extract_hsv_features(folder):
    hsv_features = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        hist /= hist.sum()
        feature = np.concatenate([mean_hsv, hist])
        hsv_features.append(feature)
    if len(hsv_features) == 0:
        return np.zeros(11)
    return np.mean(hsv_features, axis=0)

# === Paths ===
cluster_root = "fake_dataset/main_season/summer"
clusters = sorted([d for d in os.listdir(cluster_root) if d.startswith("Sub_")])
tones = ["soft", "light", "cool"]  # Common Summer subtypes

# === Extract features ===
print("🔍 Extracting features...")
cluster_features = {
    c: extract_hsv_features(os.path.join(cluster_root, c)) for c in tqdm(clusters)
}

# === Simulated tone vectors for consistent mapping
tone_vectors = {
    "soft": np.array([160, 50, 60] + [0.1]*8),
    "light": np.array([150, 40, 85] + [0.11]*8),
    "cool": np.array([170, 60, 75] + [0.12]*8),
}

# === Find best unique mapping
best_mapping = {}
best_total_dist = float("inf")

for perm in itertools.permutations(tones):
    total_dist = 0
    temp_map = {}
    for c, t in zip(clusters, perm):
        dist = euclidean(cluster_features[c], tone_vectors[t])
        total_dist += dist
        temp_map[c] = t
    if total_dist < best_total_dist:
        best_total_dist = total_dist
        best_mapping = temp_map

# === Print and Rename Folders ===
print("\n✅ Best Sub-Cluster to Summer Tone Mapping (based on HSV):")
for cluster, tone in best_mapping.items():
    old_path = os.path.join(cluster_root, cluster)
    new_path = os.path.join(cluster_root, tone)
    os.rename(old_path, new_path)
    print(f"  {cluster} → {tone}")
