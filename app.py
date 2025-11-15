from flask import Flask, render_template, request
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

app = Flask(__name__)

# Load FAISS + Chunks + Embedding model
index = faiss.read_index("faiss.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# -----------------------------
# Call Mistral via Ollama API
# -----------------------------
def ask_mistral(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    return response.json()["response"]

# -----------------------------
# RAG pipeline
# -----------------------------
def rag_answer(question):
    retrieved = retrieve(question)
    context = "\n\n".join(retrieved)
    
    prompt = f"""
You are an AI assistant. Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return ask_mistral(prompt)

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = rag_answer(question)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
