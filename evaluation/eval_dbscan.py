"""
Offline DBSCAN clustering on PDF chunk embeddings.
This script does NOT use FAISS at all.
Evaluation metrics added.
"""

import numpy as np
import pickle
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

EMBEDDINGS_PATH = "data/meta/embeddings.npy"
CHUNKS_PATH = "data/meta/chunks.pkl"

print("Loading embeddings for DBSCAN...")
embeddings = np.load(EMBEDDINGS_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks_map = pickle.load(f)

print(f"Loaded {embeddings.shape[0]} embeddings")

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print("Running DBSCAN...")

start = time.time()

dbscan = DBSCAN(
    eps=0.5,
    min_samples=2,
    metric="euclidean"
)

labels = dbscan.fit_predict(embeddings)

end = time.time()
dbscan_time = end - start

# ----- Cluster & Noise Stats -----
clusters = {}
noise = 0

for i, label in enumerate(labels):
    if label == -1:
        noise += 1
        continue
    clusters.setdefault(label, []).append(i)

n_clusters = len(clusters)
noise_ratio = noise / len(labels)

print("\n===== DBSCAN Results =====")
print(f"Total clusters: {n_clusters}")
print(f"Noise / outliers: {noise}")
print(f"Noise ratio: {noise_ratio:.3f}")
print(f"DBSCAN runtime (sec): {dbscan_time:.3f}")

# ----- Quality Metrics (only if >1 cluster) -----
valid_mask = labels != -1

if len(set(labels[valid_mask])) > 1:
    sil = silhouette_score(embeddings[valid_mask], labels[valid_mask])
    dbi = davies_bouldin_score(embeddings[valid_mask], labels[valid_mask])

    print(f"Silhouette Score: {sil:.4f}")
    print(f"Davies-Bouldin Index: {dbi:.4f}")
else:
    sil, dbi = None, None
    print("Not enough clusters for silhouette/DB index.")

# ----- Show sample clusters -----
for cid, idxs in list(clusters.items())[:3]:
    print(f"\n--- Cluster {cid} (size={len(idxs)}) ---")
    for i in idxs[:3]:
        text = chunks_map.get(i, "")
        print("-", text[:200].replace("\n", " "))
