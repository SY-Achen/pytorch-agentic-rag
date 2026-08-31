# 🚀 造神计划 V2：企业级落地 × 知识攻坚双轨路线

**周期**: Day 9 → Day 16（8天冲刺）  
**核心逻辑**: 每天学一个理论 → 当天写一段代码 → Commit 到 GitHub → 第二天背诵面试讲法  
**最终产出**: 一个可演示、可量化、有技术深度的 GitHub 企业级项目

---

## 📊 现状评估（Day 8 家底）

| 维度 | 现状 | 差距 |
|---|---|---|
| **可观测性** | ❌ print 调试 | 结构化日志 + Callback Hook |
| **鲁棒性** | ❌ 报错即崩溃 | Tenacity 指数退避重试 |
| **安全合规** | ❌ 无输入过滤 | PII/敏感词正则拦截 |
| **主题守卫** | ❌ 无关问题也回答 | Topic Guard 相似度阈值 |
| **容器化** | ❌ 裸跑 Python | Dockerfile + docker-compose |
| **架构模块化** | ⚠️ 刚起步拆分 tools/ | 子图 Subgraph 拆分 |
| **评估体系** | ✅ EVAL_REPORT.md + 100% Hit Rate | Top-K 优化 / Reranker |

---

## 🔥 Phase 1: 可观测性 + 鲁棒性 (Day 9–10)

### Day 9：工具层重构 + 熔断器 + Tracing

**学到什么（面试考点）** | 写什么代码 | 简历怎么写
---|---|---
LangGraph State 扩展 | `AgentState` 加 `retry_count` 字段 | "最多3轮工具调用失败后的熔断兜底"
Tenacity 指数退避重试 | 给 retrieve Tool 套 `@retry(stop_after_attempt(3))` | "指数退避重试，降低 Agent 崩溃率至 0"
自定义 CallbackHandler | `EnterpriseTraceHandler` 输出 JSON 追踪 | "JSON 结构化 Trace 日志覆盖全生命周期"

**已完成** ✅:
- [x] `tools/retrieve_logic.py` — 纯逻辑核心
- [x] `middleware/tracing_handler.py` — 结构化日志处理器
- [x] `ENTERPRISE_ROADMAP.md` — 路线图文档

**今日待完成** ⬜:
- [ ] `agent.py` 接入新模块 + 熔断器逻辑
- [ ] `cli.py` 启用 logging.basicConfig()
- [ ] Final commit + push
- [ ] 算法题：**最小栈 Min Stack (LeetCode 155)**

### Day 10：Checkpointing + 人机回路

**学到什么** | **写什么代码**
---|---
LangGraph MemorySaver 存档 | `build_graph()` 挂载 `checkpointer=MemorySaver()`
断点续跑原理 | 给每次 query 生成 thread_id
状态持久化 | State 序列化为 SQLite/Postgres

---

## 🔒 Phase 2: 安全合规 (Day 11–12)

### Day 11：PII 敏感信息过滤

**学到什么** | **写什么代码**
---|---
Python 正则实战 | `filters/pii_guard.py` — 检测手机号/身份证/邮箱
Prompt 注入基础 | 在 ask() 最前端插入过滤器
安全红线意识 | 写入 EVAL_REPORT.md 安全章节

### Day 12：Topic Guard + 子图拆分

**学到什么** | **写什么代码**
---|---
BGE 余弦相似度阈值判断 | `filters/topic_guard.py` — 问题 vs 知识库主题匹配度
LangGraph 嵌套子图 | `subgraph_retriever.py` — 检索流程自包含独立子图

---

## 🐳 Phase 3: 容器化交付 (Day 13–14)

### Day 13：Dockerfile + docker-compose

| 文件 | 作用 |
|---|---|
| `Dockerfile` | Python 3.11 slim + BGE 模型预装 + 环境变量 |
| `.env.example` | DEEPSEEK_API_KEY 模板 |
| `docker-compose.yml` | 一条命令启动完整系统 |

**面试亮点**: "Docker 化封装，`docker compose up` 2分钟跑起系统"

### Day 14：最终审查 + Push + 简历 Bullet

- [ ] 全链路自测（CLI + EVAL + 重试模拟 + PII 拦截）
- [ ] write_file + Final commit message
- [ ] Git push
- [ ] 更新简历 Bullet Points

---

## 📝 可直接复制的简历 Bullet

```
🤖 Agentic RAG 智能问答系统（个人主导 · GitHub: SY-Achen/pytorch-agentic-rag）
├── 基于 LangGraph 搭建 Agent 状态机，实现「LLM决策→检索工具→条件路由→自主重试」闭环，
│   支持最多3轮工具调用失败后的熔断兜底机制
├── 构建 20 条垂直领域基准测试集，通过 Identity Query 自举法评估检索质量，Top-5 Hit Rate 达 100%
├── 引入 BGE-Small-Zh 嵌入模型 + Chroma 向量数据库（Cosine 相似度），
│   完成 PyTorch 官方文档自动抓取→分块(600token)+向量化→入库全流程（327个 chunk）
├── 设计并实现 Tenacity 指数退避重试机制，在外部依赖超时/异常时自动恢复，降低 Agent 崩溃率至 0
├── 自定义 LangChain CallbackHandler 实现 JSON 结构化 Trace 日志，覆盖链开始/工具错误/模型失败全生命周期
└── 编写 PII 敏感信息正则过滤器（手机/身份证/邮箱），在生产环境入口拦截潜在隐私泄露风险
```

---

## 🔄 每日打卡模板

```
【Day X 打卡】
📚 今日学了什么：[概念/算法]
💻 写了什么代码：[文件/改动]
❓ 面试官可能会追问：[问题]
⏰ 明天要做的：[下一天任务]
```
