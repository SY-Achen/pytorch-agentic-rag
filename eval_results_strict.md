# Strict benchmark snapshot

Run date: 2026-09-07
Target: deployed financial Agentic RAG service on Alibaba Cloud ECS
Dataset: `50_eval_dataset.json` (60 cases)

## Metrics

| Metric | Result | Denominator / definition |
|---|---:|---|
| Financial retrieval hit rate | 32.0% | 8/25 financial cases returned at least one citation |
| Financial answer keyword accuracy | 48.0% | 12/25 financial cases contained every declared keyword |
| Calculator route accuracy | 100.0% | 10/10 exact CALCULATE route matches |
| General dialogue route accuracy | 30.0% | 3/10 exact DIRECT_ANSWER route matches |
| Prompt-injection block rate | 40.0% | 2/5 exact INJECT_GUARD route matches |
| Memory pass rate | 100.0% | 5/5 second-turn unique values recovered |
| Cache hit rate | 100.0% | 5/5 repeated-query pairs hit CACHE |
| Average elapsed time | 2448.6 ms | Average over the 60 dataset cases |

## Cache latency samples

| Pair | First request | Repeated request |
|---|---:|---:|
| c1 | 3079.4 ms | 49.0 ms |
| c2 | 2920.8 ms | 40.3 ms |
| c3 | 1392.8 ms | 40.1 ms |
| c4 | 2228.7 ms | 48.0 ms |
| c5 | 3505.0 ms | 38.7 ms |

## Interpretation

This snapshot is a real baseline, not a resume claim. The current strengths are calculator routing, memory recovery, and exact-query caching. The current weaknesses are financial retrieval quality, general-dialogue routing, and prompt-injection coverage. Any future resume metric must be generated from a later run after these issues are fixed.
