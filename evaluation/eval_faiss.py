import time
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "data/meta/faiss.index"
CHUNKS_PATH = "data/meta/chunks.pkl"

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks_map = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Total vectors in FAISS: {index.ntotal}")

# ---- Sample queries from existing chunks ----
sample_ids = list(chunks_map.keys())[:20]
queries = [chunks_map[i][:200] for i in sample_ids]

top_k = 5

all_distances = []
all_times = []

for q in queries:
    qvec = model.encode([q], convert_to_numpy=True).astype(np.float32)

    start = time.time()
    D, I = index.search(qvec, top_k)
    end = time.time()

    all_times.append(end - start)
    all_distances.extend(D[0].tolist())

avg_query_time = np.mean(all_times)
avg_topk_distance = np.mean(all_distances)

print("\n===== FAISS Evaluation (Proposed System) =====")
print(f"Avg Query Time (sec): {avg_query_time:.6f}")
print(f"Avg Top-K L2 Distance: {avg_topk_distance:.4f}")

# ---- Optional: Cosine Similarity (more interpretable) ----
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_sims = []

for q in queries:
    qvec = model.encode([q], convert_to_numpy=True)[0]
    qvec = qvec / np.linalg.norm(qvec)

    D, I = index.search(qvec.reshape(1, -1).astype(np.float32), top_k)

    for idx in I[0]:
        if idx == -1:
            continue
        doc_vec = model.encode([chunks_map[idx]], convert_to_numpy=True)[0]
        doc_vec = doc_vec / np.linalg.norm(doc_vec)
        cos_sims.append(cosine_sim(qvec, doc_vec))

print(f"Avg Top-K Cosine Similarity: {np.mean(cos_sims):.4f}")
