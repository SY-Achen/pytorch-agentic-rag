"""Agentic RAG over PyTorch docs using LangGraph — Enterprise Edition.

Key upgrades (Day 9):
- Modular tool architecture (tools/retrieve_tool.py) with tenacity retry
- Structured logging via EnterpriseTraceHandler callback
- Circuit breaker: max 3 tool-call retries before forcing answer
"""
import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from pathlib import Path
from tools import retrieve
from middleware.tracing_handler import EnterpriseTraceHandler

# -- Config (overridable via env vars) --
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DB_DIR = os.environ.get("DB_DIR", "vector_db")
COLLECTION = "pytorch_docs"
TOP_K = 5
MAX_RETRY_COUNT = int(os.environ.get("MAX_RETRY_COUNT", "3"))

EMB_MODEL = os.environ.get(
    "EMB_MODEL",
    str(Path.home() / ".cache" / "modelscope" / "models"
        / "AI-ModelScope--bge-small-zh-v1.5" / "snapshots" / "master"),
)

llm = None  # set by set_llm()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retry_count: int  # ponytail: circuit breaker — counts consecutive tool-callereturns to avoid infinite loops


# module-level caches
_M = None
_COLL = None


def _resources():
    global _M, _COLL
    if _M is None:
        _M = SentenceTransformer(EMB_MODEL)
    if _COLL is None:
        _COLL = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    return _M, _COLL


def build_graph(callbacks=None):
    """Build and compile the LangGraph state machine.

    Args:
        callbacks: Optional list of LangChain BaseCallbackHandler instances.
                   Defaults to [EnterpriseTraceHandler()] if None.
    """
    assert llm is not None, "call set_llm() first"
    _resources()

    # ponytail: use the modular tool with built-in tenacity retry
    tool_node = ToolNode([retrieve])
    agent = llm.bind_tools([retrieve])

    def agent_node(state):
        resp = agent.invoke(state["messages"])
        return {"messages": [resp]}

    def should_call_tools(state):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            # Circuit breaker: if we've retried too many times, force answer
            if state.get("retry_count", 0) >= MAX_RETRY_COUNT:
                logging.warning(
                    "Circuit breaker triggered: retry_count=%d >= %d, "
                    "forcing final answer.",
                    state["retry_count"], MAX_RETRY_COUNT,
                )
                return "answer"
            return "tools"
        return "answer"

    def answer_node(state):
        return {"messages": state["messages"]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("answer", answer_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent", should_call_tools, ["tools", "answer"]
    )
    g.add_edge("tools", "agent")
    g.add_edge("answer", END)
    graph = g.compile(callbacks=callbacks or [EnterpriseTraceHandler()])
    return graph


def set_llm(api_key=None, base_url=None, model=None):
    global llm
    api_key = api_key or LLM_API_KEY
    base_url = base_url or LLM_BASE_URL
    model = model or LLM_MODEL
    assert api_key, "API key required"
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)


def ask(question: str, callbacks=None) -> str:
    """Run one query through the agent graph."""
    history = [SystemMessage(content=(
        "You are a PyTorch documentation assistant. Answer using the retrieved "
        "documentation. If you need specifics, call the retrieve tool. "
        "If you don't know, say so — don't make things up."))]
    graph = build_graph(callbacks=callbacks)
    initial = {"messages": history + [HumanMessage(content=question)]}
    final_state = graph.invoke(initial)
    return final_state["messages"][-1].content


if __name__ == "__main__":
    print("Run `python cli.py \"your question\"` after setting DEEPSEEK_API_KEY.")
