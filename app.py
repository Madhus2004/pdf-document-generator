# app.py
import os
import pickle
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

from build_index import index_pdf

# ---------- Config ----------
UPLOAD_FOLDER = "data/uploads"
META_FOLDER = "data/meta"
FAISS_INDEX_PATH = os.path.join(META_FOLDER, "faiss.index")
CHUNKS_PATH = os.path.join(META_FOLDER, "chunks.pkl")
SOURCES_PATH = os.path.join(META_FOLDER, "sources.pkl")
NEXT_ID_PATH = os.path.join(META_FOLDER, "next_id.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(META_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-me-to-a-secret"  # for flash messages
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------- Embedding model ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = model.get_sentence_embedding_dimension()

# ---------- Load or create FAISS index and metadata ----------
def create_empty_index():
    flat = faiss.IndexFlatL2(EMBED_DIM)
    index = faiss.IndexIDMap(flat)  # allows adding explicit integer ids
    return index

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    # read metadata
    with open(CHUNKS_PATH, "rb") as f:
        chunks_map = pickle.load(f)
    with open(SOURCES_PATH, "rb") as f:
        source_map = pickle.load(f)
    with open(NEXT_ID_PATH, "rb") as f:
        next_id = pickle.load(f)
else:
    index = create_empty_index()
    chunks_map = {}   # id -> chunk text
    source_map = {}   # id -> filename
    next_id = 0

# ---------- Save helpers ----------
def save_everything():
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks_map, f)
    with open(SOURCES_PATH, "wb") as f:
        pickle.dump(source_map, f)
    with open(NEXT_ID_PATH, "wb") as f:
        pickle.dump(next_id, f)

# ---------- Retrieval ----------
def retrieve(query, top_k=5):
    qvec = model.encode([query], convert_to_numpy=True)
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32)
    if index.ntotal == 0:
        return []
    distances, ids = index.search(qvec, top_k)
    results = []
    for dist, iid in zip(distances[0], ids[0]):
        if iid == -1:
            continue
        chunk_text = chunks_map.get(int(iid), "[missing chunk]")
        source = source_map.get(int(iid), "unknown")
        results.append({"id": int(iid), "distance": float(dist), "chunk": chunk_text, "source": source})
    return results

# ---------- Call local model (Mistral via Ollama) ----------
def ask_mistral(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "mistral", "prompt": prompt, "stream": False}
    # If Ollama is not running the requests will fail; handle gracefully
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return f"[Error contacting model: {e}]"

# ---------- RAG pipeline ----------
def rag_answer(question, top_k=5):
    retrieved = retrieve(question, top_k=top_k)
    # join the top retrieved snippets, include source for traceability
    context = "\n\n".join([f"[{r['source']}] {r['chunk']}" for r in retrieved])
    prompt = f"""You are an AI assistant. Use ONLY the context below to answer.
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return ask_mistral(prompt), retrieved

# ---------- Flask routes ----------
@app.route("/", methods=["GET", "POST"])
def home():
    global next_id
    answer = ""
    retrieved = []
    if request.method == "POST":
        # Distinguish upload vs question by form name
        if "pdf" in request.files:
            file = request.files["pdf"]
            if file and file.filename.lower().endswith(".pdf"):
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(save_path)
                # Auto-index the uploaded PDF
                next_id = index_pdf(
                    pdf_path=save_path,
                    model=model,
                    faiss_index=index,
                    next_id=next_id,
                    chunks_map=chunks_map,
                    source_map=source_map,
                    chunk_size=1000,
                    overlap=200,
                    embed_batch_size=64,
                )
                save_everything()
                flash(f"Uploaded and indexed {file.filename} ({next_id} total chunks indexed so far).")
                return redirect(url_for("home"))
            else:
                flash("Please upload a PDF file.")
                return redirect(url_for("home"))
        elif "question" in request.form:
            question = request.form.get("question", "").strip()
            if question:
                answer, retrieved = rag_answer(question, top_k=5)
            else:
                flash("Please type a question.")
                return redirect(url_for("home"))
    # Build list of uploaded files (unique)
    uploaded_files = sorted({v for _, v in source_map.items()})
    return render_template("index.html", answer=answer, retrieved=retrieved, uploaded_files=uploaded_files)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
