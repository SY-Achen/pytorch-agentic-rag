# 智链星图 · 工业级 Agentic RAG 智能问答与决策协同系统

> **在线演示 (Live Demo)**: `http://47.114.33.146:8000`  
> 纯 Docker 容器化云原生部署，具备 Plan-Act-Observe 自主决策循环、L1~L3 纵深安全防御与毫秒级全链路可观测性。

---

## 🌟 系统核心特性与架构亮点

1. **自主 Agent Loop 决策机制**
   - 摒弃传统的“先检索再生成”死板管线，采用 **Plan-Act-Observe-Synthesize** 自主状态机循环。
   - 结合前置意图快速分流，实现多工具自主调用（知识库密集检索、实时联网搜索、受限安全沙箱计算、日常闲聊直答）。

2. **多路混合检索与动态门限控制 (Distance Threshold)**
   - 基于 `BGE-small-zh-v1.5` 稠密向量索引与 BM25 关键词加权重排。
   - 引入**余弦距离门限过滤**（`Distance Threshold <= 0.55`），物理截断弱相关召回，彻底解决 RAG 系统在闲聊和无知识场景下的“强行召回与虚假引用”痛点。

3. **纵深安全防御体系 (L1~L3 Guardrails)**
   - **L1 物理隔离**：外部检索上下文统一以 `<UNTRUSTED_DATA>` XML 标签包裹隔离，System 级硬约束禁止执行标签内指令；
   - **L2 前置门禁**：启发式正则过滤，即时掐断提示词直接/间接越狱尝试；
   - **L3 RBAC 权限隔离**：基于 JWT 与部门密级标签（Clearance Level），在检索和生成前双重拦截越权数据。

4. **轻量生产就绪与低配云原生调优**
   - 纯 Python 标准库 + FastAPI 单文件极简架构，零外部沉重微服务依赖；
   - 优化 Embedding 模型加载与 Batch Size（`batch_size=8`），在 2核2G 低配服务器下平稳运行，内存占用率稳定在 45% 左右。

---

## 🏗️ 架构数据流图

```text
[Client / Web UI] ──(SSE Stream)──► [FastAPI Gateway]
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                [Intent Fast-Route]                  [L1/L2 Security Guard]
                        │                                     │
                        ▼                                     ▼
          [DeepSeek Planner (Agent Loop)] ◄───► [In-process Tool Registry]
                        │                              ├── KNOWLEDGE_SEARCH (ChromaDB + BGE)
                        │                              ├── WEB_SEARCH (DuckDuckGo)
                        │                              └── CALCULATE (Math Sandbox)
                        ▼
                [Synthesizer Stream] ──► [Trace Logger & Feedback DB]
```

---

## 🚀 快速开始 (Quick Start)

### 1. 本地直接运行

```bash
# 克隆仓库并创建虚拟环境
git clone git@github.com:SY-Achen/pytorch-agentic-rag.git
cd pytorch-agentic-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装最小生产依赖
pip install -r requirements.txt

# 配置环境变量 (从模板复制)
cp .env.template .env
# 编辑 .env 填入你的 DEEPSEEK_API_KEY

# 启动服务
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. Docker Compose 一键生产部署

```bash
cp .env.docker.template .env
docker compose up -d --build
```

---

## 🛠️ 技术选型栈

| 模块 | 核心选型 | 说明 |
|---|---|---|
| **后端框架** | FastAPI + Uvicorn | 异步高性能 Web 与 SSE 流式事件推送 |
| **推理大模型** | DeepSeek V3 / Chat | 驱动 Agent 规划、意图识别与内容合成 |
| **向量嵌入** | BGE-small-zh-v1.5 | 本地 512 维稠密向量编码 |
| **向量数据库** | ChromaDB (HNSW Cosine) | 持久化向量索引与余弦距离检索 |
| **安全与认证** | HMAC-SHA256 JWT + Bcrypt | 部门 RBAC 权限隔离与无状态鉴权 |
| **容器编排** | Docker / Compose | 多阶段 slim 构建与 1800MB 资源限制 |

---

## 📋 变更记录 (Changelog)

- **v3.2**：完成生产级 Docker Compose 云原生部署与内存调优；接入 DeepSeek 官方引擎；引入余弦距离门限过滤与前置闲聊快速分流。
- **v3.0**：实现多步自主 Agent Loop；落地 L1~L3 纵深提示词注入防御与 JWT/RBAC 权限隔离。
- **v2.0**：上线现代化 FastAPI Web UI 与多类型文档异步入库管线。
- **v1.0**：完成最小可用 Agent 检索原型验证。
