# build_index.py (minimal, run after installing packages)
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import math

# 1) Extract text (simple concatenation of pages)
def extract_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n\n".join(pages)

# 2) Simple chunker (split by approx n characters)
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# 3) Embed with sentence-transformers (all-MiniLM-L6-v2)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# 4) Build FAISS index (simple, flat)
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # change to your file
    text = extract_text(pdf_path)
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    # save index and chunks for retrieval
    faiss.write_index(index, "faiss.index")
    import pickle
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Done. {len(chunks)} chunks indexed.")
