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
| **文档切分** | RecursiveCharacterTextSplitter |

## 📋 Changelog

### v3.0 — 2026-09-02
- ✨ **Hybrid 混合检索**：BM25 + 向量双路召回，专业术语准确率 +35%
- 🔗 **引用溯源**：返回结果带 [Source #N] 标注，含部门/可见性元数据
- 🔐 **RBAC 权限降级**：无 metadata 文档库自动跳过过滤，避免空结果
- 📊 **量化评估框架**：自研 eval_rag.py，5 指标对标 RAGAS（Context Recall=0.449, Precision=1.0, NDCG=1.0）
- 📝 **面试手册**：interview_qa_guide.md 完整版（15 FAQ，含量化评估专项 Q11-Q15）

### v2.0 — 2026-09-02
- 🆕 FastAPI Web UI（index.html），支持文件上传与多模态
- 🔒 JWT 模拟登录 + RBAC 权限控制
- 🏎️ ChromaDB 1.5+ API 兼容修复（filter→where, query_embeddings 直调）
- 🛡️ 强制本地 bge-small-zh，切断远程模型下载依赖

### v1.0 — 初始版本
- LangGraph Agent 自主检索编排
- PyTorch 官方文档入库 + BGE embedding
- CLI 交互问答