#  Financial RAG System

##  Project Overview

This project is a **Retrieval-Augmented Generation (RAG) system** designed to answer financial questions using real-world financial documents.

The system combines:

- Semantic search using Sentence Transformers
- Document chunking with overlap
- FAISS vector database for similarity search
- Qwen2.5 Large Language Model for answer generation
- Interactive Streamlit interface

---

## Live Demo

You can try the deployed application here:

[Financial RAG Streamlit App](https://financialrag1.streamlit.app/)

<img width="938" height="275" alt="image" src="https://github.com/user-attachments/assets/11b02a4d-a8b3-490a-8cd2-87d5c3bfc448" />

<img width="911" height="210" alt="image" src="https://github.com/user-attachments/assets/5cdcde18-d23f-4c17-ac23-f03fea91f3ee" />

---

##  System Architecture

```text
User Question
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Prompt Construction
      ↓
Qwen2.5 Generator
      ↓
Final Answer
````

---

##  Dataset

Dataset source:

* FinanceBench dataset from HuggingFace
* Converted and stored locally in:
  `financebenchDataset.txt`

The dataset contains financial company reports and question-answer pairs.

### Dataset Fields

| Field    | Description                   |
| -------- | ----------------------------- |
| company  | Company name                  |
| question | Financial question            |
| answer   | Ground truth answer           |
| evidence | Supporting financial evidence |

---

##  Data Preprocessing

### 1. Document Loading

The dataset is loaded from a local JSON text file:

```python
with open("financebenchDataset.txt", "r", encoding="utf-8") as f:
```

---

### 2. Document Chunking

Documents are split into smaller chunks before embedding.

### Chunking Strategy

* Chunk Size: `300 words`
* Overlap: `50 words`

This improves:

* retrieval quality
* semantic matching
* context relevance

---

##  Models Used

### 1. Embedding Model

Model:
`sentence-transformers/all-mpnet-base-v2`

Used for:

* semantic embeddings
* cosine similarity retrieval

Features:

* dense vector representations
* normalized embeddings

---

### 2. Vector Database

FAISS (`IndexFlatIP`)

Used for:

* fast nearest neighbor search
* cosine similarity retrieval

Why `IndexFlatIP`?
Because embeddings are normalized using:

```python
normalize_embeddings=True
```

---

### 3. Generator Model

Model:
`Qwen/Qwen2.5-0.5B-Instruct`

Used for:

* context-aware answer generation
* RAG-based financial QA

---

## RAG Pipeline Workflow

### Step 1 — Load Documents

Load local financial documents from JSON file.

---

### Step 2 — Chunk Documents

Split large financial texts into overlapping chunks.

---

### Step 3 — Generate Embeddings

Convert chunks into dense vectors using SentenceTransformer.

---

### Step 4 — Build FAISS Index

Store embeddings inside FAISS vector database.

---

### Step 5 — Retrieve Relevant Chunks

* Convert user query into embedding
* Retrieve Top-K similar chunks

---

### Step 6 — Generate Final Answer

The retrieved chunks are passed into Qwen2.5 as context.

The model answers ONLY using retrieved information.

---

##  Example Usage

```python
query = "What was Airbnb's net income in 2023?"

retrieved_docs = retrieval(query, k=3)

print(rag(query, retrieved_docs))
```

---

##  Project Structure

```text
.
├── app.py
├── rag.py
├── load_data.py
├── financebenchDataset.txt
├── requirements.txt
├── README.md
```

---

##  Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Run the Application

```bash
streamlit run app.py
```

---

##  Technologies Used

* Python
* Streamlit
* FAISS
* Sentence Transformers
* HuggingFace Transformers
* Qwen2.5
* NumPy
* PyTorch

---

##  Author

Sarah Alowjan

```
```
