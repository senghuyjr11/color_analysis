import os
import cv2
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

def load_images(folder, size=(128, 128), max_images=30):
    paths = glob(os.path.join(folder, '*'))
    images = []
    for path in paths[:max_images]:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

def plot_grid(images, rows=5, cols=6, title=''):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def visualize_clusters(base_dir, max_images=30):
    clusters = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    for cluster in clusters:
        print(f"\n📂 {cluster}")
        images = load_images(os.path.join(base_dir, cluster), max_images=max_images)
        plot_grid(images, title=cluster)

# === Run ===
if __name__ == '__main__':
    visualize_clusters('clustered_seasons_kmeans')
