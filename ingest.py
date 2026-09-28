"""Split documents into chunks and embed + store in a Chroma vector DB."""
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMB_MODEL = "C:/Users/Administrator/.cache/modelscope/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
COLLECTION = "pytorch_docs"


def structured_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split schema/intent Markdown by headings and table rows before size fallback."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    blocks, current = [], []
    for line in text.splitlines():
        line = line.rstrip()
        is_heading = line.startswith("#")
        if line.startswith("|") and all(ch in "|-: \t" for ch in line):
            continue
        is_table_row = line.startswith("|")
        if is_heading and current:
            blocks.append("\n".join(current).strip())
            current = []
        if is_table_row and current:
            blocks.append("\n".join(current).strip())
            current = []
        if line.strip():
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    chunks = []
    for block in blocks:
        if len(block) <= chunk_size:
            chunks.append(block)
            continue
        # Keep table/schema rows intact; only long prose is windowed.
        step = max(1, chunk_size - CHUNK_OVERLAP)
        chunks.extend(block[i:i + chunk_size] for i in range(0, len(block), step))
    return [c for c in chunks if c.strip() and (c.startswith("|") or len(c.strip()) >= 20)]

def load_docs(data_dir: str) -> list[str]:
    docs = []
    for f in sorted(Path(data_dir).glob("*.md")):
        docs.append(f.read_text(encoding="utf-8"))
    return docs


def main(data_dir: str = "data", db_dir: str = "vector_db"):
    docs = load_docs(data_dir)
    chunks = []
    for doc in docs:
        chunks.extend(structured_chunks(doc))
    print(f"  {len(docs)} docs -> {len(chunks)} structured chunks")

    model = SentenceTransformer(EMB_MODEL)
    print("  embedding ...")
    vecs = model.encode(chunks, batch_size=8, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=db_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    ids = [str(i) for i in range(len(chunks))]
    coll.add(ids=ids, documents=chunks, embeddings=vecs.tolist())
    print(f"  stored {coll.count()} chunks in Chroma @ {db_dir}")


if __name__ == "__main__":
    main()