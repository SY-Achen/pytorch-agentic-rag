# RAG Agent — Docker 部署指南

## 1. 本地构建镜像

```bash
cd rag_agent

# 构建
docker build -t smart-rag-agent .

# 验证（本地不挂载数据）
docker run --rm -p 8000:8000 \
  -e DEEPSEEK_API_KEY=your-key \
  -e DEEPSEEK_MODEL=your-model \
  -e DEEPSEEK_BASE_URL=https://api.shuaiapi.com/v1 \
  smart-rag-agent
```

访问 `http://localhost:8000` 确认 Web UI 正常。

---

## 2. 云服务器部署

### 前置条件

- Linux 服务器（Ubuntu 20+/CentOS 7+ / Alpine）
- Docker ≥ 20.10, docker-compose ≥ 2.x
- 公网 IP 或反向代理（nginx/caddy）
- 已配置 `DEEPSEEK_API_KEY`（`.env` 中填写）

### Step-by-step

```bash
# 1) 上传代码到服务器
scp -r rag_agent user@server:/opt/rag_agent

# 2) SSH 登录并进入项目目录
ssh user@server
cd /opt/rag_agent

# 3) 准备 .env 文件
cp .env.docker.template .env
nano .env          # 填入 API Key、JWT Secret 等

# 4) 预下载 Embedding 模型（避免首次启动超时）
python download_emb_model.py         # 默认下载到 ./models/bge-small-zh-v1.5
# 然后把 EMB_MODEL 路径写入 .env

# 5) 构建并启动
docker compose up -d --build

# 6) 查看日志 & 健康状态
docker compose logs -f
docker compose ps
curl http://localhost:8000/api/system/status
```

### 持久化卷说明

容器内 `/app/data` 单点挂载到宿主机 `./data`，包含：

- `sessions.db` — 用户会话历史
- `feedback.jsonl` — 用户点赞/踩记录
- `traces/` — Agent 追踪日志
- `uploads/` — 用户上传文档原文
- `vector_db/` — ChromaDB 向量索引

---

## 3. Nginx 反代（推荐）

```nginx
server {
    listen 80;
    server_name rag.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

然后加 SSL（letsencrypt）即可线上使用。

---

## 4. 环境变量清单（.env）

| 变量名 | 必填 | 说明 | 默认值 |
|-------|------|------|-------|
| `DEEPSEEK_API_KEY` | ✅ | 你的 LLM API Key | 空 |
| `DEEPSEEK_MODEL` | ✅ | 使用的模型名称 | `claude-haiku-4-5` |
| `DEEPSEEK_BASE_URL` | ✅ | API Base URL | `https://api.shuaiapi.com/v1` |
| `DEEPSEEK_BASE_URL_FALLBACK` | ❌ | 故障转移备用 URL | 空 |
| `ROUTE_FALLBACK_TIMEOUT` | ❌ | 路由超时秒数 | `20` |
| `JWT_SECRET` | ❌ | JWT signing key | dev-secret-change-me |
| `HOST_PORT` | ❌ | 宿主机端口 | `8000` |
| `DATA_PATH` | ❌ | 数据持久化路径 | `./data` |
| `VECTOR_DB_PATH` | ❌ | 向量库持久化路径 | `./vector_db` |
| `UPLOAD_PATH` | ❌ | 上传目录持久化路径 | `./uploads` |
| `EMB_MODEL` | ❌ | Embedding 模型路径 | `/app/models/bge-small-zh-v1.5/snapshots/master` |

---

## 5. 常用运维命令

```bash
# 重启
docker compose restart

# 查看实时日志
docker compose logs -f

# 进入容器调试
docker exec -it smart-rag-agent sh

# 重建容器（保留数据）
docker compose down
docker compose up -d

# 更新代码后重建
git pull
docker compose up -d --build

# 清理未使用的镜像
docker image prune -f
```

---

## 6. FAQ

**Q: 第一次启动卡很久？**  
A: Embedding 模型首次下载需要几分钟。用 `download_emb_model.py` 预下载或设置 `EMB_MODEL` 指向已有路径。

**Q: 容器重启后用户数据丢失？**  
A: 确保 `.env` 中的 `DATA_PATH`/`VECTOR_DB_PATH`/`UPLOAD_PATH` 指向宿主机真实目录。不要裸跑 `docker run` 不挂载卷。

**Q: 云端连接 API 超时？**  
A: 检查防火墙放行 + DNS 配置（阿里云用 223.5.5.5），可用 `curl https://api.shuaiapi.com/v1/chat/completions` 测试连通性。
