#📄 PDF Question Answering System
Retrieval-Augmented Generation using FAISS (Proposed System) with DBSCAN Comparison
🚀 Project Overview

This project implements a PDF-based Question Answering System using a Retrieval-Augmented Generation (RAG) architecture.

The primary focus (Proposed System) is on:

🔎 High-performance semantic retrieval using FAISS

Additionally, we implemented:

📊 DBSCAN-based clustering as a comparative enhancement technique

The system allows users to upload a PDF document and ask natural language questions. It retrieves semantically relevant content and generates grounded responses using a local Large Language Model (LLM).

🎯 Problem Statement

Large Language Models alone:

May hallucinate

Cannot access external PDF knowledge directly

Lack document grounding

Our solution:

Convert PDF into vector embeddings

Store in FAISS index

Retrieve relevant chunks

Generate context-grounded answers

🏗️ System Architecture
PDF Document
     ↓
Text Extraction
     ↓
Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Index  ← (Proposed Retrieval System)
     ↓
Top-K Relevant Chunks
     ↓
Local LLM
     ↓
Final Answer

Comparative Module:
Embeddings
     ↓
DBSCAN Clustering
     ↓
Cluster-Based Filtering
     ↓
Evaluation & Comparison

🧠 Proposed System: FAISS-Based Retrieval
🔎 Why FAISS?

FAISS (Facebook AI Similarity Search) is a library designed for efficient similarity search over dense vectors.

Advantages:

Fast nearest neighbor search

Scalable to large document collections

Optimized for high-dimensional embeddings

Production-ready retrieval system

Role in This Project

PDF chunks → converted to embeddings

Embeddings stored in FAISS index

Query converted to embedding

Top-k most similar chunks retrieved

Retrieved chunks passed to LLM for answer generation

FAISS acts as the core semantic retrieval engine in our system.

📊 Comparative Approach: DBSCAN Clustering

We implemented clustering using:

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Purpose:

Group semantically similar chunks

Remove noise embeddings

Compare cluster-based retrieval with FAISS-only retrieval

Why DBSCAN?

No need to predefine number of clusters

Works well with semantic embeddings

Identifies dense semantic regions

However:

DBSCAN was used as an experimental comparative module.
FAISS retrieval demonstrated better efficiency and consistency.

🔁 Retrieval-Augmented Generation (RAG)

The system follows the RAG paradigm:

Retrieve relevant document chunks (FAISS)

Augment LLM input with retrieved context

Generate grounded answer

This reduces:

Hallucination

Irrelevant generation

Context loss

⚙️ Technologies Used

Python

FAISS (Primary Retrieval Engine)

Scikit-learn (DBSCAN clustering)

Sentence Transformers (Embeddings)

LangChain

Local LLM (Llama / Mistral via Ollama)

NumPy

Matplotlib (Evaluation)

📈 Evaluation Strategy

We performed separate evaluation experiments for:

1️⃣ FAISS-only retrieval
2️⃣ FAISS + DBSCAN filtering

Metric Used:

Cosine Similarity between generated answer and ground truth

Average similarity score calculated

Sample Result:
Method	Average Similarity Score
FAISS (Proposed)	~66%
DBSCAN-assisted	Slight variation depending on cluster density
Observation:

FAISS provided stable and faster retrieval

DBSCAN helped in noise reduction but increased processing overhead

FAISS proved to be more suitable as a primary retrieval mechanism

🏆 Key Contributions

✔ Designed a complete RAG pipeline
✔ Implemented FAISS as the proposed semantic retrieval engine
✔ Applied DBSCAN for clustering-based comparison
✔ Conducted independent evaluation experiments
✔ Analyzed retrieval quality using similarity metrics
✔ Built modular and scalable architecture
