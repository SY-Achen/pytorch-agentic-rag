"""Seed corpus ingestion for the education KB.

Scans data_docs/ (+ legacy data/), cleans, splits, embeds into the
'education_kb' Chroma collection. Reuses server._extract_text/_simple_split
so formats and chunking stay consistent with user uploads.

Usage: .venv/Scripts/python.exe seed_ingest.py
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import server  # noqa: E402  (reuses embedding path + splitters)


def _clean(text: str) -> str:
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip() or not ln.isspace()).strip()


def main() -> int:
    root = Path(__file__).parent
    docs = []
    seen = set()
    for d, kind in [(server.BASE_DATA_DIR / "data_docs", "seed"), (root / "data_docs", "seed")]:
        if not d.exists():
            continue
        # recursive: persistent /app/data/data_docs takes precedence in Docker.
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in server._SEED_EXTS and f.name not in seen:
                docs.append((f, kind))
                seen.add(f.name)
    if not docs:
        print("没有找到种子文档（data_docs/ 或 data/ 下无支持格式）。")
        return 1

    print(f"发现 {len(docs)} 个种子文档，开始清洗+切分...")
    all_chunks, all_metas, all_ids = [], [], []
    ts = time.strftime("%Y%m%d%H%M%S")
    n = 0
    for f, kind in docs:
        text = server._extract_text(f)
        if text.startswith("[UNSUPPORTED_FORMAT]") or text.startswith("[PARSE_ERROR]") or not text.strip():
            print(f"  跳过 {f.name}: 解析为空或失败")
            continue
        chunks = server._simple_split(_clean(text), chunk_size=600, overlap=120)
        all_chunks += chunks
        course, chapter = server._classify_course(f"{f.name} {text[:3000]}")
        all_metas += [{"source": f.name, "type": kind, "course": course, "chapter": chapter,
                       "access": "public", "owner": "", "allowed_users": ""} for _ in chunks]
        all_ids += [f"seed_{ts}_{n + i}" for i in range(len(chunks))]
        n += len(chunks)
        print(f"  {f.name}: {len(chunks)} chunks")

    if not all_chunks:
        print("没有可嵌入的文本。")
        return 1

    print(f"加载 embedding 模型并向量化 {len(all_chunks)} chunks（batch_size=8）...")
    emb = server._get_embedding_model()
    if emb is None:
        print("embedding 模型加载失败，检查 EMB_MODEL 或网络。")
        return 1
    vecs = emb.encode(all_chunks, batch_size=8, normalize_embeddings=True, show_progress_bar=True)

    print("写入 ChromaDB education_kb ...")
    coll = server._chromadb.PersistentClient(path=str(server.DB_DIR)).get_or_create_collection(
        "education_kb", metadata={"hnsw:space": "cosine"})
    # ponytail: replace only administrator seed vectors; keep every user_upload_* vector intact.
    old_seed_ids = coll.get(where={"type": "seed"}, include=[]).get("ids", [])
    if old_seed_ids:
        coll.delete(ids=old_seed_ids)
        print(f"  已删除 {len(old_seed_ids)} 个旧种子分片")
    coll.upsert(ids=all_ids, documents=all_chunks, embeddings=vecs.tolist(), metadatas=all_metas)
    print(f"完成：{len(all_chunks)} chunks 已入库 education_kb。")
    server.ANSWER_CACHE.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())