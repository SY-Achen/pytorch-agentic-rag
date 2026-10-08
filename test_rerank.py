import server


class _FakeReranker:
    def predict(self, pairs):
        assert pairs == [("树的遍历", "无关文本"), ("树的遍历", "二叉树遍历包括先序、中序、后序")]
        return [0.1, 0.9]


def test_cross_encoder_reranker_orders_candidates_by_model_score(monkeypatch):
    monkeypatch.setattr(server, "_cross_encoder", _FakeReranker())
    pairs = [("无关文本", {"source": "a"}), ("二叉树遍历包括先序、中序、后序", {"source": "b"})]
    ranked = server._rerank_pairs("树的遍历", pairs, top_k=1)
    assert ranked == [("二叉树遍历包括先序、中序、后序", {"source": "b"})]
