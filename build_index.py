# build_index.py
# Helpers to extract, chunk, embed and add to a FAISS index with persistent metadata.
import os
import pdfplumber
import numpy as np

# --- Text extraction ---
def extract_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n\n".join(pages)

# --- Chunking ---
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Simple character-based chunker with overlap.
    Returns list of chunk strings.
    """
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

# --- Batch embed wrapper (model should be a SentenceTransformer) ---
def embed_chunks(model, chunks, batch_size=64):
    """
    Returns numpy array of dtype float32 shape (n_chunks, dim).
    """
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
    # ensure float32 (faiss expects float32)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings

# --- Top-level function to index a single PDF into an existing faiss index ---
def index_pdf(
    pdf_path,
    model,
    faiss_index,
    next_id,
    chunks_map,
    source_map,
    chunk_size=1000,
    overlap=200,
    embed_batch_size=64,
):
    """
    Extracts text from pdf_path, chunks it, embeds chunks, and adds them to faiss_index.
    Maintains:
      - next_id : integer id to start assigning vector ids from (will return updated next_id)
      - chunks_map : dict mapping int id -> chunk text
      - source_map : dict mapping int id -> source filename (or path)
    Returns updated next_id.
    """
    text = extract_text(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return next_id  # nothing to add

    embeddings = embed_chunks(model, chunks, batch_size=embed_batch_size)
    n = embeddings.shape[0]

    # prepare ids
    ids = np.arange(next_id, next_id + n, dtype='int64')

    # FAISS expects (n, dim) float32
    # faiss.IndexIDMap requires int64 ids in add_with_ids
    faiss_index.add_with_ids(embeddings, ids)

    # update metadata maps
    filename = os.path.basename(pdf_path)
    for i, chunk in enumerate(chunks):
        doc_id = int(ids[i])
        chunks_map[doc_id] = chunk
        source_map[doc_id] = filename

    return next_id + n
