"""Core retrieval logic — pure function, no framework dependencies."""
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
from typing import Optional

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


def _do_retrieve(query: str, metadata_filter: Optional[dict] = None) -> str:
    """Core retrieval logic — wrapped by @retry in retrieve_tool.py.

    ponytail: metadata_filter enables RBAC / time_decay / department isolation
    at query time, zero code changes to the graph topology.
    """
    m, coll = _resources()
    q = m.encode([query], normalize_embeddings=True).tolist()[0]
    kwargs = {"query_embeddings": [q], "n_results": TOP_K}
    if metadata_filter:
        kwargs["filter"] = metadata_filter  # ChromaDB $and/$gte support
    res = coll.query(**kwargs)
    docs = res["documents"][0]

    # warn agent if all chunks have very low similarity
    scores = res.get("distances", [[]])[0]
    if scores and all(s > 1.5 for s in scores):
        return "[WARNING] Retrieved %d chunks but all have low similarity." % len(docs)

    if not docs:
        return "[EMPTY] No matching documents found."
    return "\n\n---\n\n".join(docs)
