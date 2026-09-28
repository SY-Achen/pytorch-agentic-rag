"""Small, dependency-free evaluator for Agent runs and benchmark cases."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_dataset(cases: list[dict], runs: list[dict]) -> dict:
    """Compare expected route/keywords with recorded case results."""
    by_id = {str(item.get("id")): item for item in runs}
    route_hits = keyword_hits = 0
    details = []
    for case in cases:
        run = by_id.get(str(case.get("id")), {})
        answer = str(run.get("answer", run.get("reply", ""))).lower()
        actual_route = run.get("route", run.get("actual"))
        route_ok = actual_route == case.get("expected_tool")
        keywords = [str(k).lower() for k in case.get("must_contain", case.get("expected_keywords", []))]
        keyword_ok = bool(keywords) and all(k in answer for k in keywords)
        route_hits += route_ok
        keyword_hits += keyword_ok
        details.append({"id": case.get("id"), "route_ok": route_ok, "keyword_ok": keyword_ok})
    total = len(cases)
    return {
        "cases": total,
        "routing_accuracy": _rate(route_hits, total),
        "answer_keyword_hit_rate": _rate(keyword_hits, total),
        "details": details,
    }


def evaluate_trace(events: Iterable[dict]) -> dict:
    """Aggregate JSONL trace events without assuming an LLM provider."""
    rows = list(events)
    trace_ids = {str(row.get("trace_id")) for row in rows if row.get("trace_id")}
    trace_count = len(trace_ids)
    cache_hits = sum(row.get("step") == "cache" and row.get("status") == "hit" for row in rows)
    guard_blocks = sum(row.get("step") == "inject_guard" and row.get("status") == "blocked" for row in rows)
    latencies = []
    for trace_id in trace_ids:
        total = sum(float(row.get("ms", 0) or 0) for row in rows if str(row.get("trace_id")) == trace_id)
        if total:
            latencies.append(total)
    latencies.sort()
    latency = {
        "count": len(latencies),
        "avg": statistics.fmean(latencies) if latencies else 0.0,
        "p50": statistics.median(latencies) if latencies else 0.0,
        "max": max(latencies) if latencies else 0.0,
    }
    return {
        "trace_count": trace_count,
        "cache_hit_rate": _rate(cache_hits, trace_count),
        "guard_block_rate": _rate(guard_blocks, trace_count),
        "latency_ms": latency,
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # Accept both a plain run list and the existing strict benchmark envelope.
    return data.get("results", data) if isinstance(data, dict) else data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agent benchmark results and JSONL traces")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--md", type=Path, help="also write a compact Markdown report")
    args = parser.parse_args()
    report = {}
    if args.dataset and args.runs:
        report["dataset"] = evaluate_dataset(_load_json(args.dataset), _load_json(args.runs))
    if args.trace:
        report["trace"] = evaluate_trace(read_jsonl(args.trace))
    if not report:
        parser.error("provide --dataset and --runs, or --trace")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.md:
        args.md.write_text(to_markdown(report), encoding="utf-8")
    print(text)


def to_markdown(report: dict) -> str:
    lines = ["# Agent Evaluation Report", ""]
    dataset = report.get("dataset")
    if dataset:
        lines += ["## Benchmark", "", f"- Cases: {dataset['cases']}", f"- Routing accuracy: {dataset['routing_accuracy']:.1%}", f"- Answer keyword hit rate: {dataset['answer_keyword_hit_rate']:.1%}", ""]
    trace = report.get("trace")
    if trace:
        latency = trace["latency_ms"]
        lines += ["## Trace", "", f"- Traces: {trace['trace_count']}", f"- Cache hit rate: {trace['cache_hit_rate']:.1%}", f"- Guard block rate: {trace['guard_block_rate']:.1%}", f"- Latency p50 / avg / max: {latency['p50']:.0f} / {latency['avg']:.0f} / {latency['max']:.0f} ms", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
