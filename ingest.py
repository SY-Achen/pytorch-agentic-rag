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


def load_docs(data_dir: str) -> list[str]:
    docs = []
    for f in sorted(Path(data_dir).glob("*.md")):
        text = f.read_text(encoding="utf-8")
        # drop code fences' noise, collapse blank lines — keep it simple
        text = re.sub(r"\n{3,}", "\n\n", text)
        docs.append(text)
    return docs


def main(data_dir: str = "data", db_dir: str = "vector_db"):
    docs = load_docs(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Priority order: code fences > sub-section headers > blank paragraphs > line breaks > words
        separators=[
            "\n```",     # ① 代码块边界（前后各留几行上下文）
            "\n```py",   # Python 代码标记（更精确）
            "\n#",      # 任何级别的 Markdown 标题（# ## ### ####）
            "\n\n",     # 空段落
            "\n",       # 换行
            " ",        # 单词间空格
            "",         # 字符级兜底
        ],
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))
    print(f"  {len(docs)} docs -> {len(chunks)} chunks")

    model = SentenceTransformer(EMB_MODEL)
    print("  embedding ...")
    vecs = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=db_dir)
    coll = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    ids = [str(i) for i in range(len(chunks))]
    coll.upsert(ids=ids, documents=chunks, embeddings=vecs.tolist())
    print(f"  stored {coll.count()} chunks in Chroma @ {db_dir}")


if __name__ == "__main__":
    main()