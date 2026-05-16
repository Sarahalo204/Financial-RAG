# Financial RAG System (FAISS + Qwen2.5 + Streamlit)

##  Project Overview

This project is a **Retrieval-Augmented Generation (RAG) system** that answers financial questions using real-world financial datasets.

It combines:

* Semantic search (Sentence Transformers)
* Vector database (FAISS)
* Large Language Model (Qwen2.5)
* Interactive UI (Streamlit)

---

## 🌐 Live Demo

You can try the deployed application here:

[Financial RAG Streamlit App](https://financialrag1.streamlit.app/?utm_source=chatgpt.com)

---

## System Architecture

```text
User Question
      ↓
SentenceTransformer Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Relevant Documents
      ↓
Prompt Construction
      ↓
Qwen2.5 LLM (Generator)
      ↓
Final Answer
```

---

## Dataset

Used dataset:


### Structure

| Field    | Description               |
| -------- | ------------------------- |
| company  | Company name              |
| question | Financial question        |
| answer   | Correct/reference answer  |
| evidence | Supporting financial text |

---

## Models Used

### 1. Embedding Model

* `sentence-transformers/all-mpnet-base-v2`
* Converts text into dense vectors
* Normalized for cosine similarity

---

### 2. Vector Database

* **FAISS (IndexFlatIP)**
* Fast similarity search using inner product
* Works with normalized embeddings (cosine similarity)

---

### 3. Generator Model

* `Qwen/Qwen2.5-0.5B-Instruct`
* Lightweight instruction-tuned LLM
* Generates final answer using retrieved context

## Example Usage

```python
query = "What was Airbnb's net income in 2023?"

retrieved_docs = retrieval(query, k=3)
print(rag(query, retrieved_docs))
```

---


## Installation

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```text
.
├── app.py
├── rag.py
├── requirements.txt
├── README.md
```

---

## Author

Sarah Alowjan

---
