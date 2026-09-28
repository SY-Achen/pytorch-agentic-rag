from trace_eval import evaluate_dataset, evaluate_trace


def test_dataset_evaluation_reports_route_and_keyword_accuracy():
    cases = [
        {"id": 1, "q": "查基金", "expected_tool": "KNOWLEDGE_SEARCH", "must_contain": ["基金"]},
        {"id": 2, "q": "算 1+1", "expected_tool": "CALCULATE", "must_contain": ["2"]},
    ]
    runs = [
        {"id": 1, "route": "KNOWLEDGE_SEARCH", "answer": "基金查询结果"},
        {"id": 2, "route": "CALCULATE", "answer": "结果是 2"},
    ]
    report = evaluate_dataset(cases, runs)
    assert report["cases"] == 2
    assert report["routing_accuracy"] == 1.0
    assert report["answer_keyword_hit_rate"] == 1.0


def test_trace_evaluation_aggregates_latency_cache_and_guard():
    lines = [
        {"trace_id": "a", "step": "plan_1", "status": "ok", "ms": 100},
        {"trace_id": "a", "step": "retrieve", "status": "ok", "ms": 30},
        {"trace_id": "a", "step": "synthesize", "status": "ok", "ms": 70},
        {"trace_id": "b", "step": "cache", "status": "hit", "ms": 0},
        {"trace_id": "c", "step": "inject_guard", "status": "blocked", "ms": 0},
    ]
    report = evaluate_trace(lines)
    assert report["trace_count"] == 3
    assert report["cache_hit_rate"] == 1 / 3
    assert report["guard_block_rate"] == 1 / 3
    assert report["latency_ms"]["p50"] == 200


def test_trace_rows_are_json_like_dicts():
    report = evaluate_trace([{"trace_id": "smoke", "step": "cache", "status": "hit", "ms": 0}])
    assert report["trace_count"] == 1
