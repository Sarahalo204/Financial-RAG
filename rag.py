import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import streamlit as st


# 1 load data
# ds = load_dataset("PatronusAI/financebench")
# Company + Question + Answer + Evidence
@st.cache_data
def load_documents():

    with open("financebenchDataset.txt", "r", encoding="utf-8") as f:
        docs = json.load(f)

    return docs


raw_docs = load_documents()

# 2 chunking
def chunk_text(text, chunk_size=300, overlap=50):

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks

chunked_docs = []

for doc in raw_docs:

    chunks = chunk_text(doc)

    chunked_docs.extend(chunks)


# 3 embedding model
@st.cache_resource #i used cache to loaded the model  only once.
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

model = load_embedding_model()


# 4 embedding dataset
@st.cache_resource
def build_faiss_index(_model, docs):
    embedding_dataset = _model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

    dimension = embedding_dataset.shape[1]
    index = faiss.IndexFlatIP(dimension)  # i used IndexFlatIP beucase normalize_embeddings=True
    index.add(embedding_dataset)

    return index, embedding_dataset
  
index, embedding_dataset = build_faiss_index(model, chunked_docs)  

# 5 retrieva by using query

def retrieval(q,k=3):

  q_embedding=model.encode(q,convert_to_numpy=True,normalize_embeddings=True)

  q_embedding = np.array(q_embedding).reshape(1, -1) # (1, D)

  distances, indices =index.search(q_embedding,k)

  result=[]

  for i in range(k):
    doc_index = int(indices[0][i])
    distance=distances[0][i]

    result.append({
        "text": chunked_docs[doc_index],
        "score": float(distance)
        })

  return result

# 6 generate
@st.cache_resource
def load_generator():

    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="auto"
    )

generator = load_generator()

# 7 all pipline rag

def rag(query, retrieval_docs):
  
#   retrieval_docs=retrieval(query,2)

  context = "\n".join([doc["text"] for doc in retrieval_docs])

  prompt = f"""
    You are a financial assistant.

    Answer ONLY using the context below.
    If the answer is not in the context, say: "I don't know based on the provided documents."
    Context:
    {context}

    Question:
    {query}

    Answer in bullet points:
    """

  output = generator(
    prompt,
    max_new_tokens=100,
    temperature=0.1
  )

  response = output[0]["generated_text"]

  response = response.replace(prompt, "").strip()

  return response
