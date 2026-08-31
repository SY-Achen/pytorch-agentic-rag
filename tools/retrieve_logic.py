"""Core retrieval logic — pure function, no framework dependencies."""
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

# -- singleton resources (module-level cache) --
_M = None
_COLL = None
DB_DIR = os.environ.get("DB_DIR", "vector_db")
COLLECTION = "pytorch_docs"
TOP_K = 5
EMB_MODEL = os.environ.get(
    "EMB_MODEL",
    str(Path.home() / ".cache" / "modelscope" / "models"
        / "AI-ModelScope--bge-small-zh-v1.5" / "snapshots" / "master"),
)


def _resources():
    global _M, _COLL
    if _M is None:
        _M = SentenceTransformer(EMB_MODEL)
    if _COLL is None:
        _COLL = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    return _M, _COLL


def _do_retrieve(query: str) -> str:
    """Core retrieval logic — wrapped by @retry in retrieve_tool.py."""
    m, coll = _resources()
    q = m.encode([query], normalize_embeddings=True).tolist()[0]
    res = coll.query(query_embeddings=[q], n_results=TOP_K)
    docs = res["documents"][0]

    # ponytail: warn agent if all chunks have very low similarity (cosine distance > 1.5 ≈ not similar)
    scores = res.get("distances", [[]])[0]
    if scores and all(s > 1.5 for s in scores):
        return "[WARNING] Retrieved %d chunks but all have low similarity. Consider rephrasing." % len(docs)

    if not docs:
        return "[EMPTY] No matching documents found."
    return "\n\n---\n\n".join(docs)
