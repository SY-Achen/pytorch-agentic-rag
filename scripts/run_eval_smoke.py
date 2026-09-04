#!/usr/bin/env python3
"""Smoke eval wrapper — runs eval_rag with top_k=5 and asserts overall exists.
Not a full suite; ad-hoc regression hook.
"""
import subprocess, sys, json
from pathlib import Path

ROOT = Path(r"C:\Users\Administrator\rag_agent")
out = ROOT / "eval_results_latest.json"
cmd = [sys.executable, str(ROOT / "eval_rag.py"), "--top-k", "5"]
print("RUN:", " ".join(cmd))
p = subprocess.run(cmd, cwd=str(ROOT))
if p.returncode != 0:
    sys.exit(p.returncode)
data = json.loads(out.read_text(encoding="utf-8"))
overall = data.get("stats", {}).get("overall_score")
print("OVERALL:", overall)
assert overall is not None
print("SMOKE EVAL OK")
