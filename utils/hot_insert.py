"""rag_agent/utils/hot_insert.py — 热插拔增量入库管道
核心：新文档监听 → 清洗 → Embedding → 追加 ChromaDB，零停机。"""
import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from .data_cleaner import clean
except ImportError:
    from data_cleaner import clean
try:
    from tools.retrieve_logic import _resources
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.retrieve_logic import _resources


def compute_hash(file_path: str) -> str:
    """计算文件 MD5，用于去重。"""
    sha = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def hot_insert(
    file_path: str,
    collection_name: str = "pytorch_docs",
    dept: str = "public",
    visibility: str = "public",
    ttl_days: Optional[int] = None,
) -> dict:
    """将单个新文档增量注入向量库。

    Args:
        file_path: 上传的文件路径（支持 .txt/.md/.pdf 等文本可解析格式）
        collection_name: ChromaDB 集合名
        dept: 部门标签
        visibility: 可见性等级 (public/internal/confidential/secret)
        ttl_days: 生存天数，None 表示永久

    Returns:
        {"status": "ok|skipped|error", "doc_id": "...", "hash": "..."}
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {"status": "error", "reason": "file_not_found"}

        # 1. 计算哈希（防重复）
        file_hash = compute_hash(str(file_path))
        doc_id = f"doc_{file_hash}"

        # 2. 检查是否已存在
        _, coll = _resources()
        existing = coll.get(where={"doc_hash": file_hash})
        if existing and len(existing["ids"]) > 0:
            logger.info(f"Document already exists: {doc_id}")
            return {"status": "skipped", "doc_id": doc_id, "reason": "duplicate"}

        # 3. 读取并清洗文本
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean(text)
        if not cleaned_text.strip():
            return {"status": "error", "reason": "empty_after_cleaning"}

        # 4. 构建元数据
        from datetime import datetime, timedelta
        metadata = {
            "doc_id": doc_id,
            "doc_hash": file_hash,
            "filename": file_path.name,
            "dept": dept,
            "visibility": visibility,
            "created_at": datetime.now().strftime("%Y-%m-%d"),
        }
        if ttl_days:
            metadata["expires_at"] = (datetime.now() + timedelta(days=ttl_days)).strftime("%Y-%m-%d")

        # 5. 调用嵌入模型 + 追加到向量库
        emb_model, _ = _resources()
        embeddings = emb_model.encode([cleaned_text], normalize_embeddings=True).tolist()[0]

        # ponytail: 按段落切块防止单条文本过长，这里简化为整篇插入
        # 生产环境应复用 data_cleaner.chunk_paragraphs()
        coll.add(
            ids=[doc_id],
            documents=[cleaned_text],
            embeddings=[embeddings],
            metadatas=[metadata],
        )

        logger.info(f"✅ Hot inserted: {doc_id} ({len(cleaned_text)} chars)")
        return {"status": "ok", "doc_id": doc_id, "chars": len(cleaned_text), "hash": file_hash}

    except Exception as e:
        logger.error(f"Hot insert failed: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}


if __name__ == "__main__":
    print("=== Hot Insert Test ===")

    # 用一段测试文本模拟新合同
    test_file = Path("/tmp/test_new_contract.txt")
    test_file.write_text(
        "2026年度Q3采购合同\n"
        "甲方：智能制造科技有限公司\n"
        "乙方：八维通机器人系统有限公司\n"
        "采购设备：宇树Go2巡检机器狗 × 2台\n"
        "合同金额：人民币 580,000 元\n\n"
        "注意事项：本合同涉及内部定价，仅限授权人员查阅。"
    )

    result = hot_insert(
        str(test_file),
        dept="procurement",
        visibility="internal",
        ttl_days=365,
    )
    print(f"Result: {result}")
    assert result["status"] == "ok", f"Expected ok, got {result}"

    # 测试去重：再次插入同一个文件
    result2 = hot_insert(str(test_file), dept="procurement", visibility="internal")
    assert result2["status"] == "skipped", "应检测到重复"
    print(f"Duplicate check: {result2['status']} ✅")

    # 清理测试文件
    test_file.unlink(missing_ok=True)

    print("\n✅ All hot_insert tests passed.")
