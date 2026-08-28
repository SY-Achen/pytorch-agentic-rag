"""Agentic RAG over PyTorch docs using LangGraph.

The LLM (orchestrator) decides: answer directly, or call the retriever tool
then answer from retrieved context. This makes retrieval agent-driven rather
than a fixed retrieve-then-generate pipeline.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from pathlib import Path
from typing import Annotated, TypedDict

# EMB_MODEL env var overrides; default resolves the ModelScope cache under the user's home dir.
EMB_MODEL = os.environ.get(
    "EMB_MODEL",
    str(Path.home() / ".cache" / "modelscope" / "models"
        / "AI-ModelScope--bge-small-zh-v1.5" / "snapshots" / "master"),
)
DB_DIR = os.environ.get("DB_DIR", "vector_db")
COLLECTION = "pytorch_docs"
TOP_K = 5

llm = None  # set by set_llm()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# module-level caches: load BGE model + Chroma client once, reuse across calls.
_M = None
_COLL = None


def _resources():
    global _M, _COLL
    if _M is None:
        _M = SentenceTransformer(EMB_MODEL)
    if _COLL is None:
        _COLL = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    return _M, _COLL


@tool
def retrieve(query: str) -> str:
    """Retrieve relevant passages from the PyTorch documentation given a natural-language question."""
    m, coll = _resources()
    q = m.encode([query], normalize_embeddings=True).tolist()[0]
    res = coll.query(query_embeddings=[q], n_results=TOP_K)
    return "\n\n---\n\n".join(res["documents"][0])


def build_graph():
    assert llm is not None, "call set_llm() first"
    _resources()  # init model + chroma client on the caller (main) thread, NOT inside tool threads
    tool_node = ToolNode([retrieve])
    agent = llm.bind_tools([retrieve])

    def agent_node(state):
        resp = agent.invoke(state["messages"])
        return {"messages": [resp]}

    def should_call_tools(state):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "answer"  # no tool call -> go to answer node (which ends)

    def answer_node(state):
        """When the agent produced a final answer, pass it through."""
        return {"messages": state["messages"]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("answer", answer_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_call_tools, ["tools", "answer"])
    g.add_edge("tools", "agent")
    g.add_edge("answer", END)
    return g.compile()


def set_llm(api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
    global llm
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)


def ask(question: str) -> str:
    history = [SystemMessage(content=(
        "You are a PyTorch documentation assistant. Answer using the retrieved "
        "documentation. If you need specifics, call the retrieve tool. "
        "If you don't know, say so — don't make things up."))]
    graph = build_graph()
    final_state = graph.invoke({"messages": history + [HumanMessage(content=question)]})
    return final_state["messages"][-1].content


if __name__ == "__main__":
    # demo with a fake LLM is not possible pre-key; ensure graph builds when key present.
    print("Run `python cli.py \"your question\"` after set_llm().")