"""Core retrieval logic — pure function, no framework dependencies."""
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import re
from typing import Optional
from dataclasses import dataclass, field


# -- singleton resources (module-level cache) --
_M = None
_COLL = None
_DB_DIR = os.environ.get("DB_DIR", "vector_db")
_COLLECTION = "pytorch_docs"
_TOP_K = 5
_EMB_MODEL_PATH = os.environ.get(
    "EMB_MODEL",
    str(Path.home() / ".cache" / "modelscope" / "models"
        / "AI-ModelScope--bge-small-zh-v1.5" / "snapshots" / "master"),
)


def _resources():
    global _M, _COLL
    if _M is None:
        _M = SentenceTransformer(_EMB_MODEL_PATH)
    if _COLL is None:
        _COLL = chromadb.PersistentClient(path=_DB_DIR).get_collection(_COLLECTION)
    return _M, _COLL


# --------------------------------------------------------------------------- #
# BM25-style keyword scorer (no extra deps)
# ponytail: simplified BM25 — good enough for reranking; upgrade to rank_bm25
#   package when you need production-grade IDF curves.
# --------------------------------------------------------------------------- #
@dataclass
class _DocScore:
    doc_id: str
    text: str
    meta: dict
    combined_score: float = 0.0


def _tokenize(text: str) -> list[str]:
    """Very rough tokenizer: CJK chars each count as a token, ASCII split on whitespace."""
    cjk = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
    ascii = re.findall(r'[a-zA-Z]+', text.lower())
    return cjk + ascii


def _bm25_scores(query_tokens: list[str], docs: list[_DocScore], k=1.2, b=0.75) -> None:
    """In-place BM25 scoring (ponytail: inline IDF calc, assumes uniform doc freq ≈ 1)."""
    if not docs or not query_tokens:
        return
    avg_len = sum(len(_tokenize(d.text)) for d in docs) / len(docs)
    tf_table = {}
    for d in docs:
        tokens = _tokenize(d.text)
        seen = set(tokens)
        tf_table[d.doc_id] = {t: tokens.count(t) for t in seen}
    dl_ratio = [len(_tokenize(d.text)) / avg_len for d in docs]
    for qt in query_tokens:
        numerator = sum(tf_table.get(d.doc_id, {}).get(qt, 0) * (k + 1) for d in docs)
        if numerator == 0:
            continue
        denominator = sum(
            tf_table.get(d.doc_id, {}).get(qt, 0) + k * (1 - b + b * dl_ratio[j])
            for j, d in enumerate(docs)
        )
        score_per_doc = numerator / denominator
        for d in docs:
            d.combined_score += score_per_doc


# --------------------------------------------------------------------------- #
# Hybrid retriever: vector + keyword → merge & re-rank
# --------------------------------------------------------------------------- #
def _do_retrieve(
    query: str,
    metadata_filter: Optional[dict] = None,
    hybrid: bool = True,              # enable BM25 rerank
    top_k: Optional[int] = None,       # override default
) -> str:
    """Core retrieval logic — wrapped by @retry in retrieve_tool.py.

    ponytail: metadata_filter enables RBAC / time_decay / department isolation
    at query time, zero code changes to the graph topology.

    Returns:
        Concatenated top-K chunks WITH source citations in format:
          === SOURCE [id]: {meta} ===
          <text>
    """
    m, coll = _resources()
    actual_top_k = top_k or _TOP_K

    # --- Step 1: Vector search via ChromaDB using query_embeddings ---
    # ponytail: bypass coll.query(query_texts=[...]) which triggers auto-embed
    #   download attempts. We encode locally and pass raw embeddings.
    q_vec = m.encode([query], normalize_embeddings=True).tolist()[0]
    kwargs: dict = {"query_embeddings": [q_vec], "n_results": min(actual_top_k * 2, 20)}
    if metadata_filter:
        kwargs["where"] = metadata_filter
    res = coll.query(**kwargs)

    # Build structured results
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]

    scored: list[_DocScore] = []
    for idx in range(len(ids)):
        scored.append(_DocScore(
            doc_id=ids[idx] if idx < len(ids) else f"chunk_{idx}",
            text=docs[idx] if idx < len(docs) else "",
            meta=metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {},
            combined_score=max(0, 1.0 - dists[idx]) if idx < len(dists) else 0.0,
        ))

    # --- Step 2: Hybrid: add BM25 scores and re-rank ---
    if hybrid and scored:
        q_tokens = _tokenize(query)
        # First pass: compute BM25 independently
        bm25_values = [d.combined_score for d in scored]
        _bm25_scores(q_tokens, scored)
        # Normalize BM25 to same range as vector score (~0-1), then blend
        max_bm25 = max(d.combined_score for d in scored) if scored else 1.0
        if max_bm25 > 0:
            for d in scored:
                d.combined_score = d.combined_score * 0.6 + (d.combined_score / max_bm25) * 0.4

    scored.sort(key=lambda x: x.combined_score, reverse=True)
    final = scored[:actual_top_k]

    if not final:
        return "[EMPTY] No matching documents found."

    # Build output with citation sources
    parts = []
    for i, d in enumerate(final, 1):
        src_meta = d.meta
        dept = src_meta.get("dept", "unknown") if isinstance(src_meta, dict) else "unknown"
        visibility = src_meta.get("visibility", "?") if isinstance(src_meta, dict) else "?"
        src_line = f"[Source #{i}] Dept={dept}, Clearance={visibility}"
        parts.append(f"{src_line}\n{d.text}")

    # Low-similarity warning
    if all(s.combined_score < 0.3 for s in final):
        parts.insert(0, f"[WARNING] Retrieved {len(final)} chunks but all have low similarity.")

    return "\n\n---\n\n".join(parts)
