import os
import json
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_pdf(pdf_path):
    """Read PDF file and return list of pages (text)."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text.strip())
    return pages


def chunk_texts(pages, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for semantic search.
    Returns list of {text, page}.
    """
    chunks = []
    for i, page_text in enumerate(pages):
        words = page_text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            text = " ".join(chunk_words)
            if text.strip():
                chunks.append({
                    "text": text,    # 👈 Important: 'text' key
                    "page": i + 1
                })
            start += chunk_size - overlap
    return chunks


def build_faiss_index(chunks, model_name=EMBED_MODEL):
    """Build FAISS index from chunks."""
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts, convert_to_numpy=True).astype("float32")

    # Normalize vectors
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, chunks


def save_index(index, chunks, persist_dir="indexes/default"):
    """Save FAISS index + chunks.json."""
    os.makedirs(persist_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(persist_dir, "index.faiss"))
    with open(os.path.join(persist_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def ingest_pdf_and_index(pdf_path, persist_dir="indexes/default"):
    """Process a PDF, build index, and save to disk."""
    print(f"📖 Processing PDF: {pdf_path}")
    pages = read_pdf(pdf_path)
    chunks = chunk_texts(pages)
    print(f"✂️ Total Chunks: {len(chunks)}")

    index, chunks = build_faiss_index(chunks)
    save_index(index, chunks, persist_dir)
    print("✅ Indexing complete!")
    return persist_dir


if __name__ == "__main__":
    # Example run
    test_pdf = "sample.pdf"  # replace with your pdf path
    ingest_pdf_and_index(test_pdf)
