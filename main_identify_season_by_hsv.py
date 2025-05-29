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
        return np.zeros(11)  # 3 mean + 8 hist = 11 features
    return np.mean(hsv_features, axis=0)

# === Paths ===
cluster_root = "clustered_seasons_kmeans"
season_root = "dataset"

clusters = sorted([d for d in os.listdir(cluster_root) if d.startswith("Cluster")])
seasons = ["Autumn", "Spring", "Summer", "Winter"]

# === Extract features ===
print("🔍 Extracting features...")
cluster_features = {c: extract_hsv_features(os.path.join(cluster_root, c)) for c in tqdm(clusters)}
season_features = {s: extract_hsv_features(os.path.join(season_root, s)) for s in seasons}

# === Find best unique mapping ===
best_mapping = {}
best_total_dist = float("inf")

for perm in itertools.permutations(seasons):
    total_dist = 0
    temp_map = {}
    for c, s in zip(clusters, perm):
        dist = euclidean(cluster_features[c], season_features[s])
        total_dist += dist
        temp_map[c] = s
    if total_dist < best_total_dist:
        best_total_dist = total_dist
        best_mapping = temp_map

# === Print final mapping ===
print("Best Cluster-to-Season Mapping (unique):")
for cluster, season in best_mapping.items():
    print(f"  {cluster} → {season}")
