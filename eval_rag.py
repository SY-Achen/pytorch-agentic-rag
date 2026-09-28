"""
eval_rag.py — RAG Evaluation Script (Ponytail: zero external deps beyond sentence-transformers)
Borrowed metrics from RAGAS framework (https://github.com/explodinggradients/ragas):

  - Context Recall@K     : Did retrieved docs cover the info needed to answer?
  - Context Precision@K  : Was the retrieved set clean / relevant or noisy?
  - Faithfulness          : Does the answer faithfully follow the context (no hallucination)?
  - Answer Relevance      : Is the answer directly relevant to the question?
  - NDCG@K                : Quality-adjusted ranking metric

Usage:
    python eval_rag.py                     # Run all golden tests
    python eval_rag.py --top-k 3           # Override TOP-K
    python eval_rag.py --verbose           # Show per-test details

Metrics are scored on [0, 1] scale. Overall score = average of all subscores.
"""
import os, sys, json, math, re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# --- Paths ---
PROJECT_ROOT = Path("C:/Users/Administrator/rag_agent")
os.environ["DB_DIR"] = str(PROJECT_ROOT / "vector_db")
os.environ.setdefault(
    "EMB_MODEL",
    str(Path.home() / ".cache" / "modelscope" / "models"
        / "AI-ModelScope--bge-small-zh-v1.5" / "snapshots" / "master"),
)
sys.path.insert(0, str(PROJECT_ROOT))

from tools.retrieve_logic import _do_retrieve
from sentence_transformers import SentenceTransformer

# --- Load embedding model (singleton) ---
_emb: Optional[SentenceTransformer] = None

def get_emb():
    global _emb
    if _emb is None:
        _emb = SentenceTransformer(os.environ.get(
            "EMB_MODEL",
            str(Path.home() / ".cache" / "modelscope" / "models"
                / "AI-ModelScope--bge-small-zh-v1.5" / "snapshots" / "master"),
        ))
    return _emb

# --------------------------------------------------------------------------- #
# 1. GOLDEN TEST SET — manual QA pairs with known answers + expected topics
#    ponytail: expand this list as you accumulate real user queries that succeeded
# --------------------------------------------------------------------------- #
GOLDEN_TESTS = [
    {
        "id": "qa_01",
        "question": "PyTorch DataLoader 的 num_workers 参数有什么作用？",
        "expected_keywords": ["worker", "多进程", "并行", "process"],
        "ground_truth": "num_workers 指定 DataLoader 使用多少个子进程加载数据，实现并行加速。",
    },
    {
        "id": "qa_02",
        "question": "torch.nn.Linear 的参数有哪些？它的作用是什么？",
        "expected_keywords": ["in_features", "out_features", "linear", "全连接", "affine"],
        "ground_truth": "torch.nn.Linear 是全连接层，参数包括 in_features（输入特征数）和 out_features（输出特征数），可选 bias。",
    },
    {
        "id": "qa_03",
        "question": "什么是 PyTorch 中的 Automatic Mixed Precision (AMP)？",
        "expected_keywords": ["混合精度", "autocast", "FP16", "FP32", "GPU", "显存"],
        "ground_truth": "AMP 是一种技术，在训练中使用 FP16 减少显存占用并加速计算，同时用 FP32 保持损失缩放稳定性。",
    },
    {
        "id": "qa_04",
        "question": "DataLoader 中的 collate_fn 参数是如何工作的？",
        "expected_keywords": ["collate_fn", "batch", "堆叠", "合并", "custom"],
        "ground_truth": "collate_fn 自定义如何将单个样本组合成 batch，默认将 tensor 堆叠、列表打包。",
    },
    {
        "id": "qa_05",
        "question": "PyTorch 的 Dataset 类和 IterableDataset 有什么区别？",
        "expected_keywords": ["map-style", "iterable", "getitem", "__len__", "数据流"],
        "ground_truth": "Map-style Dataset 支持随机索引访问（getitem+__len__），Iterable-style 按顺序迭代。",
    },
    {
        "id": "qa_06",
        "question": "torch.optim.AdamW 和 Adam 有什么区别？",
        "expected_keywords": ["weight_decay", "解耦", "decoupled", "正则化"],
        "ground_truth": "AdamW 将 weight decay 从梯度更新中解耦，提供更有效的 L2 正则化。",
    },
    {
        "id": "qa_07",
        "question": "PyTorch 中的 torch.no_grad() 有什么作用？",
        "expected_keywords": ["no_grad", "推理", "禁用", "autograd", "内存"],
        "ground_truth": "torch.no_grad() 禁用 autograd 引擎，减少推理时的显存占用和提高速度。",
    },
    {
        "id": "qa_08",
        "question": "什么是 DDP 分布式数据并行？它的基本原理是什么？",
        "expected_keywords": ["DDP", "distributed", "多卡", "all_reduce", "主进程"],
        "ground_truth": "DDP 在多 GPU 间复制模型，每个进程处理不同数据 shard，通过 all_reduce 同步梯度。",
    },
]

# --------------------------------------------------------------------------- #
# 2. UTILITY FUNCTIONS — semantic similarity helpers
# --------------------------------------------------------------------------- #

def cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def encode_text(text: str) -> List[float]:
    """Encode text to normalized embedding vector."""
    emb_model = get_emb()
    vec = emb_model.encode([text], normalize_embeddings=True)[0].tolist()
    return vec


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two texts using embeddings."""
    return cosine_sim(encode_text(text_a), encode_text(text_b))


def keyword_overlap(answer_keywords: List[str], expected_keywords: List[str]) -> float:
    """Compute Jaccard-like overlap between two keyword sets."""
    if not expected_keywords:
        return 1.0
    s1 = set(k.lower() for k in answer_keywords)
    s2 = set(k.lower() for k in expected_keywords)
    intersection = s1 & s2
    return len(intersection) / len(s2)


# --------------------------------------------------------------------------- #
# 3. METRICS IMPLEMENTATION
# --------------------------------------------------------------------------- #

@dataclass
class TestResult:
    test_id: str
    question: str
    retrieved_chunks: List[Dict]
    
    # Retrieval metrics
    context_recall_k: float = 0.0       # Context Recall@K
    context_precision_k: float = 0.0    # Context Precision@K  
    ndcg_k: float = 0.0                 # NDCG@K
    
    # Generation metrics (if answer available)
    faithfulness: float = 0.0           # Answer faithful to context?
    answer_relevance: float = 0.0       # Answer relevant to question?
    answer_correctness: float = 0.0     # Similarity to ground truth
    
    overall: float = 0.0                # Aggregate score


def compute_context_recall(retrieved_docs: List[str], test: Dict) -> float:
    """
    Context Recall@K (RAGAS style):
    For each expected keyword / ground_truth statement, check if ANY retrieved
    chunk semantically covers it. Score = fraction of expected info covered.
    
    ponytail: simplified version of RAGAS context_recall — uses keyword+semantic
    coverage instead of LLM-based claim extraction. Good enough for self-checks.
    """
    if not retrieved_docs:
        return 0.0
    
    coverage_scores = []
    
    # Method 1: Keyword presence in top-1 doc
    top_doc = retrieved_docs[0]
    kw_score = keyword_overlap(top_doc.split(), test["expected_keywords"])
    coverage_scores.append(kw_score)
    
    # Method 2: Semantic coverage of ground truth by any chunk
    gt_vec = encode_text(test["ground_truth"])
    best_sim = max(
        cosine_sim(gt_vec, encode_text(doc[:500]))  # first 500 chars per doc
        for doc in retrieved_docs
    )
    coverage_scores.append(best_sim)
    
    # Weighted combination
    return 0.4 * coverage_scores[0] + 0.6 * coverage_scores[1]


def compute_context_precision(retrieved_docs: List[str], test: Dict) -> float:
    """
    Context Precision@K:
    How many of the retrieved chunks are relevant vs irrelevant.
    Score = (# relevant chunks) / K
    """
    if not retrieved_docs:
        return 0.0
    
    relevance_scores = []
    query_vec = encode_text(test["question"])
    
    for i, doc in enumerate(retrieved_docs):
        # High similarity to query → likely relevant
        sim = cosine_sim(query_vec, encode_text(doc[:500]))
        relevance_scores.append(1.0 if sim > 0.3 else 0.0)
    
    return sum(relevance_scores) / len(relevance_scores)


def compute_ndcg(retrieved_docs: List[str], test: Dict, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @K (NDCG@K):
    Ranks retrieved documents by their usefulness to the query.
    
    DCG@K = sum(rel_i / log2(i+2)) for i=1..K
    NDCG = DCG / IDCG (ideal DCG)
    """
    if not retrieved_docs:
        return 0.0
    
    k = min(k, len(retrieved_docs))
    query_vec = encode_text(test["question"])
    
    rel_scores = []
    for doc in retrieved_docs[:k]:
        sim = cosine_sim(query_vec, encode_text(doc[:500]))
        # Binary relevance: rel=1 if sim>0.3 else 0
        rel_scores.append(1.0 if sim > 0.3 else 0.0)
    
    # DCG
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel_scores))
    
    # IDCG (ideal: all rel=1)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel_scores))))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_faithfulness_from_context(question: str, context: str) -> float:
    """
    Faithfulness heuristic (ponytail: without full LLM judge, use keyword+structure match).
    Score high if: context contains entities/terms from question AND forms coherent response.
    
    NOTE: Production-grade faithfulness needs GPT-4o judgment. This is a lightweight proxy.
    """
    if not context or "[EMPTY]" in context:
        return 0.0
    
    # Check if context actually addresses the question topic
    q_terms = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', question))
    ctx_terms = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', context[:2000]))
    
    term_overlap = len(q_terms & ctx_terms) / max(len(q_terms), 1)
    
    # Bonus if context has multiple source citations (indicates actual retrieval)
    citation_bonus = 0.1 if "[Source #" in context else 0.0
    
    return min(1.0, term_overlap * 0.8 + citation_bonus)


# --------------------------------------------------------------------------- #
# 4. RUN EVALUATION
# --------------------------------------------------------------------------- #

def run_evaluation(top_k: int = 5, verbose: bool = False) -> Tuple[List[TestResult], Dict]:
    """Run all golden tests and return results."""
    print(f"🧪 Running RAG evaluation | TOP-K={top_k} | Tests={len(GOLDEN_TESTS)}\n")
    
    results: List[TestResult] = []
    
    for test in GOLDEN_TESTS:
        # Retrieve
        result_str = _do_retrieve(test["question"], metadata_filter=None, hybrid=True, top_k=top_k)
        
        # Parse retrieved chunks
        chunks = []
        for line in result_str.split("\n"):
            if line.startswith("[Source #") or (line.strip() and not line.startswith("[") and not line.startswith("---")):
                if chunks:
                    chunks[-1] += "\n" + line if chunks[-1] != line else line
                else:
                    chunks.append(line)
        
        # Deduplicate adjacent identical lines
        deduped = []
        for c in chunks:
            if not deduped or c != deduped[-1]:
                deduped.append(c)
        chunks = deduped
        
        # Compute metrics
        tr = TestResult(
            test_id=test["id"],
            question=test["question"],
            retrieved_chunks=[{"text": c} for c in chunks[:top_k]],
        )
        
        if chunks:
            tr.context_recall_k = compute_context_recall(chunks, test)
            tr.context_precision_k = compute_context_precision(chunks, test)
            tr.ndcg_k = compute_ndcg(chunks, test, k=top_k)
            tr.faithfulness = compute_faithfulness_from_context(test["question"], result_str)
        
        tr.overall = (tr.context_recall_k + tr.context_precision_k + tr.ndcg_k + tr.faithfulness) / 4
        
        results.append(tr)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Test: {test['id']}")
            print(f"Q: {test['question']}")
            print(f"  Context Recall@K: {tr.context_recall_k:.3f}")
            print(f"  Context Prec@K:   {tr.context_precision_k:.3f}")
            print(f"  NDCG@K:           {tr.ndcg_k:.3f}")
            print(f"  Faithfulness:     {tr.faithfulness:.3f}")
            print(f"  Overall:          {tr.overall:.3f}")
            if chunks:
                print(f"  Retrieved: {len(chunks)} chunks")
                print(f"  Top chunk preview: {chunks[0][:150]}...")
    
    # Aggregate stats
    n = len(results)
    avg_recall = sum(r.context_recall_k for r in results) / n
    avg_precision = sum(r.context_precision_k for r in results) / n
    avg_ndcg = sum(r.ndcg_k for r in results) / n
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_overall = sum(r.overall for r in results) / n
    
    stats = {
        "total_tests": n,
        "context_recall_avg": round(avg_recall, 3),
        "context_precision_avg": round(avg_precision, 3),
        "ndcg_avg": round(avg_ndcg, 3),
        "faithfulness_avg": round(avg_faith, 3),
        "overall_score": round(avg_overall, 3),
        "per_test": [(r.test_id, round(r.overall, 3)) for r in results],
    }
    
    return results, stats


def print_summary(stats: Dict):
    """Print formatted summary table."""
    print(f"\n{'='*60}")
    print("📊 RAG EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Score':>8}")
    print(f"{'-'*33}")
    print(f"{'Context Recall@K':<25} {stats['context_recall_avg']:>8.3f}")
    print(f"{'Context Precision@K':<25} {stats['context_precision_avg']:>8.3f}")
    print(f"{'NDCG@K':<25} {stats['ndcg_avg']:>8.3f}")
    print(f"{'Faithfulness':<25} {stats['faithfulness_avg']:>8.3f}")
    print(f"{'───':<25} {'─────':>8}")
    print(f"{'OVERALL SCORE':<25} {stats['overall_score']:>8.3f}")
    print(f"{'='*60}")
    print(f"\nPer-test scores:")
    for tid, score in stats["per_test"]:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        emoji = "✅" if score >= 0.6 else ("⚠️" if score >= 0.4 else "❌")
        print(f"  {emoji} {tid:<10} [{bar}] {score:.3f}")
    
    print(f"\n📈 Interpretation:")
    print(f"  • Score ≥ 0.7: Good — system performs well")
    print(f"  • Score ≥ 0.5: Acceptable — room for improvement")
    print(f"  • Score <  0.5: Poor — consider tuning retrieval")
    

# --------------------------------------------------------------------------- #
# 5. MAIN
# --------------------------------------------------------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Evaluation Script")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Show per-test details")
    args = parser.parse_args()
    
    results, stats = run_evaluation(top_k=args.top_k, verbose=args.verbose)
    print_summary(stats)
    
    # Save results to JSON for later comparison
    output_path = PROJECT_ROOT / "eval_results_latest.json"
    export_data = {
        "top_k": args.top_k,
        "stats": stats,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
