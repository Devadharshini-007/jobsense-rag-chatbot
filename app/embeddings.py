"""
embeddings.py
Handles:
- Splitting raw JD text into overlapping chunks
- Generating sentence embeddings (local, free — no API cost)
- Building a FAISS index from those embeddings
- Searching the index for a given query
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Load embedding model once at startup ─────────────────────────────────────
# "all-MiniLM-L6-v2" is small (80MB), fast, and works great for semantic search
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2


# ── Text chunking ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove excessive whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)          # collapse multiple spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into word-based chunks with overlap.
    - chunk_size: number of words per chunk
    - overlap: number of words shared between consecutive chunks
    
    Overlap ensures context at chunk boundaries is not lost.
    """
    text = clean_text(text)
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


# ── FAISS index ───────────────────────────────────────────────────────────────
def build_faiss_index(chunks: list[str]) -> faiss.IndexFlatL2:
    """
    Embed all chunks and store them in a FAISS flat L2 index.
    Returns the index (keep this in memory / session state).
    """
    if not chunks:
        raise ValueError("No chunks provided to build index.")

    # Generate embeddings — shape: (num_chunks, 384)
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index (IndexFlatL2 = exact nearest neighbour, no training needed)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    return index


def search_similar_chunks(
    query: str,
    faiss_index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = 3
) -> list[str]:
    """
    Embed the query and find top_k most similar chunks from the index.
    Returns a list of chunk strings.
    """
    query_embedding = EMBEDDING_MODEL.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # D = distances, I = indices of nearest neighbours
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):  # -1 means not found
            results.append(chunks[idx])

    return results


# ── Helper: process raw JD text end-to-end ───────────────────────────────────
def process_jd_text(raw_text: str):
    """
    Convenience function: takes raw JD text,
    returns (faiss_index, chunks) ready for querying.
    """
    chunks = split_into_chunks(raw_text)
    index = build_faiss_index(chunks)
    return index, chunks
