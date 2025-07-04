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
cluster_root = "clustered_seasons_kmeans/Autumn"

clusters = sorted([d for d in os.listdir(cluster_root) if d.startswith("Sub_")])
tones = ["Warm", "Deep", "Soft"]  # Expected tone categories — adjust if needed

# === Extract features ===
print("🔍 Extracting features...")
cluster_features = {c: extract_hsv_features(os.path.join(cluster_root, c)) for c in tqdm(clusters)}

# === Dummy tone vectors for matching (manual matching placeholder) ===
# We'll simulate tone HSVs just to make mapping logic consistent
tone_vectors = {
    "Warm": np.array([30, 100, 70] + [0.1]*8),
    "Deep": np.array([25, 90, 40] + [0.12]*8),
    "Soft": np.array([20, 60, 60] + [0.11]*8),
}

# === Match best unique mapping ===
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

# === Print Mapping ===
print("\n✅ Best Sub-Cluster to Autumn Tone Mapping (based on HSV):")
for cluster, tone in best_mapping.items():
    print(f"  {cluster} → {tone} Autumn")
