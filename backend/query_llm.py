import os
import json
import faiss
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "indexes/default"

# Decoding params (safe + fast)
GEN_MAX_NEW_TOKENS = 350
GEN_TEMPERATURE = 0.2
TOP_K = 4  # how many chunks to retrieve


# ----------------------------
# CACHED LOADERS (speed-up)
# ----------------------------
@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    # CPU-friendly MiniLM
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_hf_pipeline():
    # CPU fallback small instruct model
    return pipeline("text-generation", model="microsoft/phi-3-mini-4k-instruct", device=-1)


# ----------------------------
# FAISS + Chunks IO
# ----------------------------
def load_index(persist_dir: str = DEFAULT_INDEX_DIR):
    index = faiss.read_index(os.path.join(persist_dir, "index.faiss"))
    with open(os.path.join(persist_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


# ----------------------------
# Retrieval
# ----------------------------
def embed_query(query: str):
    model = get_embedder()
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    # index is IP, so normalize to get cosine similarity
    faiss.normalize_L2(vec)
    return vec


def retrieve_top_k(index, chunks, query_vec, k: int = TOP_K):
    D, I = index.search(query_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        c = chunks[idx]
        # chunks.json must contain {"text": "...", "page": N}
        results.append({
            "page": c["page"],
            "text": c["text"],
            "score": float(score)
        })
    return results


# ----------------------------
# Prompt Builder
# ----------------------------
def _trim(txt: str, limit: int = 1200) -> str:
    txt = (txt or "").strip().replace("\n", " ")
    if len(txt) > limit:
        return txt[:limit] + " ..."
    return txt


def build_prompt(question: str, results):
    # Build compact, cited context
    context_blocks = []
    for i, r in enumerate(results, start=1):
        context_blocks.append(f"[SOURCE {i} | page {r['page']}]\n{_trim(r['text'])}\n")
    context = "\n".join(context_blocks)

    prompt = f"""You are StudyMate, an academic assistant.
Use ONLY the information in the SOURCE blocks to answer the question. If the answer is not in the sources, say you cannot find it.
When stating facts, include page citations in square brackets like [page {{number}}]. Do NOT hallucinate.

SOURCES:
{context}

QUESTION:
{question}

Answer concisely (4-8 sentences) and include source citations by page number.
"""
    return prompt


# ----------------------------
# LLM Calls
# ----------------------------
def call_watsonx(prompt: str, api_key: str, url: str, project_id: str, model_id: str = None):
    """
    Returns None on failure so caller can fallback cleanly.
    """
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        if not api_key or not url or not project_id:
            return None

        creds = Credentials(api_key=api_key, url=url)
        model = ModelInference(
            model_id=model_id or "mistralai/mixtral-8x7b-instruct-v01",
            credentials=creds,
            project_id=project_id
        )

        # Newer SDKs accept generation params via **kwargs:
        out = model.generate_text(
            prompt=prompt,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE
        )

        # SDK may return dict/str depending on version
        if isinstance(out, dict):
            # Some versions: {"output":[{"content":"..."}]}
            if "output" in out and isinstance(out["output"], list) and out["output"]:
                maybe = out["output"][0].get("content")
                return maybe if isinstance(maybe, str) else json.dumps(out)
            # Fallback stringify
            return json.dumps(out)
        return str(out)
    except Exception as e:
        print("⚠️ Watsonx failed:", repr(e))
        return None


def call_huggingface(prompt: str):
    try:
        gen = get_hf_pipeline()
        out = gen(
            prompt,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=False,  # deterministic for academic answers
            temperature=GEN_TEMPERATURE
        )
        if isinstance(out, list) and out:
            return str(out[0].get("generated_text", ""))
        return str(out)
    except Exception as e:
        return f"Local fallback error: {e}"


# ----------------------------
# Main Q&A
# ----------------------------
def answer_question(
    question: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    k: int = TOP_K,
    ibm_api_key: str = None,
    ibm_url: str = None,
    ibm_project: str = None,
    ibm_model_id: str = None
):
    # Load index + chunks
    index, chunks = load_index(index_dir)

    # Embed + retrieve
    qvec = embed_query(question)
    results = retrieve_top_k(index, chunks, qvec, k=max(1, k))

    # Build prompt
    prompt = build_prompt(question, results)

    # Try Watsonx (fast)
    answer = call_watsonx(prompt, api_key=ibm_api_key, url=ibm_url, project_id=ibm_project, model_id=ibm_model_id)

    # Fallback to local HF if needed
    if not answer or (isinstance(answer, str) and not answer.strip()):
        answer = call_huggingface(prompt)

    # Return along with compact sources for UI
    ui_sources = [{"page": r["page"], "text": _trim(r["text"], 400)} for r in results]
    return {"answer": str(answer), "sources": ui_sources}


# ----------------------------
# Debug Run
# ----------------------------
if __name__ == "__main__":
    res = answer_question(
        "What is the main contribution of the paper?",
        index_dir=os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR),
        ibm_api_key=os.getenv("WATSONX_KEY"),
        ibm_url=os.getenv("WATSONX_URL"),
        ibm_project=os.getenv("WATSONX_PROJECT"),
        ibm_model_id=os.getenv("WATSONX_MODEL")  # optional override
    )
    print("\n=== ANSWER ===\n", res["answer"])
    print("\n=== SOURCES ===\n", res["sources"])
