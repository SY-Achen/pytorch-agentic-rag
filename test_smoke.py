"""Opt-in live E2E smoke; offline regression tests must never depend on API credentials."""
import os, sys
import pytest

if os.environ.get("RUN_LIVE_E2E") != "1":
    pytest.skip("set RUN_LIVE_E2E=1 and valid LLM credentials to run live smoke", allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent
from langchain_core.messages import AIMessage

KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE = os.environ.get("LLM_BASE_URL", "https://api.shuaiapi.com/v1")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash-0731")
assert KEY, "set DEEPSEEK_API_KEY"
agent.set_llm(KEY, base_url=BASE, model=MODEL)

# 1. routing is truly agentic: doc question -> tool call happened
g = agent.build_graph()
msgs = [{"role": "user", "content": "What does torch.optim.Adam do?"}]
out = g.invoke({"messages": msgs})
tool_called = any(getattr(m, "tool_calls", None) for m in out["messages"])
assert tool_called, "agent did NOT call the retriever tool for a doc question"
assert isinstance(out["messages"][-1].content, str) and len(out["messages"][-1].content) > 30
print("[e2e] doc question -> tool call + answer:", out["messages"][-1].content[:80].replace("\n", " "))
print("VERIFY_OK")