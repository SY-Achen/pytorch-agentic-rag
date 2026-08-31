"""
RAG Retrieval Eval Script (2026-08-31 v2)
Purpose: Verify retrieval quality using "Identity Query" strategy.
Method: Pick N random docs -> Embed each doc as its own query via local BGE -> Check if doc finds itself in Top-K.
No external downloads required — uses only locally cached BGE model.
Author: SY-Achen
"""
import sys
import os
import random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing deps. Install: pip install chromadb sentence-transformers")
    sys.exit(1)

# --- Local BGE model path (same as ingest.py / agent.py) ---
BGE_PATH = r"C:/Users/Administrator/.cache/modelscope/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"

class SimpleEvaluator:
    def __init__(self, db_dir="./vector_db", top_k=5, sample_size=20):
        self.db_dir = db_dir
        self.top_k = top_k
        self.sample_size = sample_size
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection("pytorch_docs")
        self.model = SentenceTransformer(BGE_PATH)
        print(f"[OK] Loaded BGE model from {BGE_PATH}")

    def get_all_ids(self):
        return self.collection.get(include=["embeddings"])["ids"]

    def encode(self, texts):
        """Encode texts to vectors."""
        return self.model.encode(texts, normalize_embeddings=True)

    def cosine_similarity(self, q_vec, d_vecs):
        """Compute cosine similarity between query and all documents. Return indices sorted descending."""
        sims = d_vecs @ q_vec
        return np.argsort(sims)[::-1]

    def run_identity_test(self):
        """
        Core Evaluation Logic:
        1. Sample K documents from DB with their embeddings
        2. For each doc, embed its own text as query
        3. Score ALL collection embeddings vs this query
        4. Record if the doc appears in Top-K results
        """
        ids_all = self.get_all_ids()
        n_total = len(ids_all)
        samples = random.sample(ids_all, min(self.sample_size, n_total))
        
        # Load ALL embeddings once for efficient scoring
        print(f"\n[Info] Total docs in DB: {n_total}, loading all embeddings...")
        all_data = self.collection.get(
            include=["documents", "embeddings"], 
            limit=None,
            offset=0
        )
        all_ids = all_data["ids"]
        all_docs = all_data["documents"]
        all_embs = np.array(all_data["embeddings"])
        
        hits = 0
        misses_detail = []
        print(f"\n=== Running Identity Eval (Top-{self.top_k}) ===")
        print(f"Testing {len(samples)} random chunks...\n")
        
        for rank, doc_id in enumerate(samples, 1):
            # Find index in full list
            idx = all_ids.index(doc_id)
            doc_text = all_docs[idx]
            
            # Embed document text as query
            q_vec = self.encode([doc_text])[0]
            
            # Score all docs against this query
            sorted_indices = self.cosine_similarity(q_vec, all_embs)
            
            # Check if doc found itself in Top-K
            top_k_indices = sorted_indices[:self.top_k]
            top_k_ids = [all_ids[i] for i in top_k_indices]
            
            found = doc_id in top_k_ids
            
            if found:
                hits += 1
                top_pos = list(top_k_indices).index(all_ids.index(doc_id)) + 1
                status = f"✅ HIT (rank #{top_pos})"
            else:
                status = f"❌ MISS (top result: ...{all_docs[top_k_indices[0]][:40]}...)"
                misses_detail.append({"id": doc_id, "text": doc_text})
                
            print(f"[{status}] ID: {doc_id} | Text: ...{doc_text[-40:]}")
            
        accuracy = (hits / len(samples)) * 100
        print(f"\n>> RESULTS:")
        print(f">> Hit Rate (Rank-1): {accuracy:.1f}% ({hits}/{len(samples)})")
        
        if misses_detail:
            print(f"\n>> Miss Analysis ({len(misses_detail)} cases):")
            for m in misses_detail[:3]:
                print(f"   Doc: ...{m['text'][-60:]}")
            print(f"   ... see full analysis in your terminal above")
            
        return accuracy


if __name__ == "__main__":
    DB_PATH = os.environ.get("RAG_DB_PATH", "./vector_db")
    
    if not os.path.exists(DB_PATH):
        print(f"Error: DB directory '{DB_PATH}' not found.")
        sys.exit(1)
        
    evaluator = SimpleEvaluator(db_dir=DB_PATH, top_k=5, sample_size=20)
    evaluator.run_identity_test()
