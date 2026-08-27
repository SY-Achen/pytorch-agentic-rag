# PyTorch Agentic RAG Chatbot

基于 **LangGraph** 的检索增强生成（RAG）问答系统，针对 PyTorch 官方文档。
检索由智能体（LLM）自主触发——先判断是否需要查文档，再调用检索工具，最后基于检索内容作答。

## 亮点

- **Agentic 检索**：LLM 通过 `bind_tools` 自主决定何时调用 `retrieve` 工具，而非固定"先检索再生成"流程
- **本地嵌入**：BGE-small embedding（ModelScope 下载，国内可达），无需外网模型 API
- **Chroma 向量库**：余弦相似度检索，命中相关段落
- **可复现**：数据来自 PyTorch 官方 GitHub 源码，可扩展任意文档

## 数据流

```
fetch_docs.py  →  官方 .md 源码
      ↓
ingest.py      →  切分(600/120 overlap) → BGE嵌入 → Chroma入库
      ↓
agent.py       →  LangGraph: agent(LLM) ⇄ retrieve工具 → 答案
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate               # Windows
pip install -r requirements.txt

python fetch_docs.py                 # ① 拉取 PyTorch 官方文档
python ingest.py                     # ② 切分+嵌入+入库
python cli.py init --key sk-xxxx     # ③ 配置 DeepSeek key
python cli.py "How do I create a tensor on GPU?"   # ④ 提问
```

## 用到的技术

| 组件 | 选型 |
|---|---|
| 智能体编排 | LangGraph |
| LLM | DeepSeek (OpenAI 兼容接口) |
| 嵌入 | BGE-small (ModelScope) |
| 向量库 | Chroma |
| 文档切分 | RecursiveCharacterTextSplitter |