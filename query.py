import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import json

# ----------------------------
# Load model & FAISS index
# ----------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("faiss.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ----------------------------
# Function: Retrieve top-k chunks
# ----------------------------

def retrieve_context(query, k=5):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)

    retrieved = [chunks[i] for i in I[0]]
    return "\n\n".join(retrieved)

# ----------------------------
# Function: Ask Ollama Mistral
# ----------------------------

def ask_ollama(question, context):
    prompt = f"""
You are a document assistant. Answer the question using ONLY the context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    return data["response"]

# ----------------------------
# MAIN LOOP
# ----------------------------

if __name__ == "__main__":
    print("PDF QA System Ready!")
    print("Ask a question... (type 'exit' to quit)\n")

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        ctx = retrieve_context(question)
        answer = ask_ollama(question, ctx)

        print("\nAnswer:", answer)
        print("\n" + "-" * 40 + "\n")
