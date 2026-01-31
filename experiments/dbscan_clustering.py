"""
Offline DBSCAN clustering on PDF chunk embeddings.
This script does NOT use FAISS at all.
"""

import numpy as np
import pickle
from sklearn.cluster import DBSCAN

EMBEDDINGS_PATH = "data/meta/embeddings.npy"
CHUNKS_PATH = "data/meta/chunks.pkl"

print("Loading embeddings for DBSCAN...")
embeddings = np.load(EMBEDDINGS_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks_map = pickle.load(f)

print(f"Loaded {embeddings.shape[0]} embeddings")

print("Running DBSCAN...")
# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

dbscan = DBSCAN(
    eps=0.5,
    min_samples=2,
    metric="euclidean"
)


labels = dbscan.fit_predict(embeddings)

clusters = {}
noise = 0

for i, label in enumerate(labels):
    if label == -1:
        noise += 1
        continue
    clusters.setdefault(label, []).append(i)

print("\n===== DBSCAN Results =====")
print(f"Total clusters: {len(clusters)}")
print(f"Noise / outliers: {noise}")

for cid, idxs in clusters.items():
    print(f"\n--- Cluster {cid} (size={len(idxs)}) ---")
    for i in idxs[:3]:
        text = chunks_map.get(i, "")
        print("-", text[:200].replace("\n", " "))
