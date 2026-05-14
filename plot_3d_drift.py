import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# ==========================
# IMAGE FOLDERS
# ==========================

REF_DIR = "/home/navavision/Desktop/Avantik_Tasks/YOLOX/ref_images"
PROD_DIR = "/home/navavision/Desktop/Avantik_Tasks/YOLOX/prod_images"

# ==========================
# LOAD IMAGES
# ==========================

def load_images(folder):

    features = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))

        feature = img.flatten()

        features.append(feature)

    return np.array(features)

ref_features = load_images(REF_DIR)
prod_features = load_images(PROD_DIR)

# ==========================
# COMBINE DATA
# ==========================

X = np.vstack([ref_features, prod_features])

labels = (
    ["Reference"] * len(ref_features)
    + ["Production"] * len(prod_features)
)

# ==========================
# PCA → 3D
# ==========================

pca = PCA(n_components=3)

X_3d = pca.fit_transform(X)

# ==========================
# 3D PLOT
# ==========================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')

for label in ["Reference", "Production"]:

    idx = [i for i, l in enumerate(labels) if l == label]

    ax.scatter(
        X_3d[idx, 0],
        X_3d[idx, 1],
        X_3d[idx, 2],
        label=label
    )

ax.set_title("3D Data Drift Visualization")

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")

ax.legend()

plt.savefig("3d_drift_visualization.png")


