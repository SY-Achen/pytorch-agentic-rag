"""
smart_qa_server.py — Enterprise RAG UI Server
FastAPI backend: login, RBAC routing, hybrid retrieval, multi-modal upload.
"""
import os, json, base64, shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Environment ---
APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
STATIC_DIR = UPLOAD_DIR / "images"
os.makedirs(STATIC_DIR, exist_ok=True)

# --- Mock Users ---
MOCK_USERS = {
    "zhangsan": {"password": "123", "name": "张三", "dept": "sales",      "level": 3},
    "lisi":     {"password": "123", "name": "李四", "dept": "engineering", "level": 6},
    "wangwu":  {"password": "123", "name": "王五", "dept": "admin",        "level": 9}
}

app = FastAPI(title="智链星图 · Enterprise Agentic RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if STATIC_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(STATIC_DIR)), name="uploads")

# --- Models ---
class LoginRequest(BaseModel): username: str; password: str
class ChatRequest(BaseModel):
    message: str; files: list[str] = []; username: str = ""; dept: str = ""
class ChatResponse(BaseModel):
    reply: str; images: list[str] = []; sources: list[dict] = []

# --- Core ---
@app.post("/api/login")
def login(req: LoginRequest):
    user = MOCK_USERS.get(req.username)
    if user and user["password"] == req.password:
        return {"success": True, "token": f"session_{req.username}", "user_info": {**user, "username": req.username}}
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    dest_path = STATIC_DIR / unique_name
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"url": f"/uploads/{unique_name}"}

@app.get("/api/collections/info")
def collection_info():
    """Debug endpoint: show collection stats."""
    try:
        import chromadb
        c = chromadb.PersistentClient(path=os.environ.get("DB_DIR", "vector_db"))
        coll = c.get_collection("pytorch_docs")
        count = coll.count()
        # Sample first doc
        sample = coll.get(limit=1)["metadatas"][0] if count > 0 else {}
        return {"status": "ok", "total_chunks": count, "sample_metadata": sample}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def _has_metadata_sample(collection_name="pytorch_docs", sample_size=50):
    """Check if collection has non-empty metadata (ponytail: avoid filtering empty meta)."""
    import chromadb
    client = chromadb.PersistentClient(path=os.environ.get("DB_DIR", "vector_db"))
    coll = client.get_collection(collection_name)
    docs_with_meta = 0
    try:
        batch = coll.get(limit=min(sample_size, coll.count()))
        for meta in batch.get("metadatas", []):
            if meta and isinstance(meta, dict) and len(meta) > 0:
                docs_with_meta += 1
        return docs_with_meta > sample_size * 0.5
    except Exception:
        return False


@app.post("/api/chat")
def rag_chat(req: ChatRequest):
    # 1. Build metadata filter — only if collection actually has metadata
    user_meta = MOCK_USERS.get(req.username)
    if not user_meta:
        return ChatResponse(reply="[ERROR] Unknown user")
    
    metadata_filter = None  # Start with no filter
    
    # Check if collection has metadata (to avoid filtering out all docs when meta is empty)
    has_meta = _has_metadata_sample()
    
    if has_meta:
        # Apply RBAC filter only when collection has visibility/dept fields
        metadata_filter = {
            "$and": [
                {"visibility": {"$gte": 0}},
                {"visibility": {"$lte": user_meta["level"]}}
            ]
        }
        if req.dept:
            metadata_filter["$and"].append({"dept": req.dept})
    
    # 2. Context injection based on department
    system_prompt_map = {
        "sales": "You are an expert Sales Assistant. Focus on pricing, ROI, and competitive advantages.",
        "engineering": "You are a Senior Engineer. Be technical, precise, focus on troubleshooting and architecture.",
        "admin": "You are a System Administrator. Focus on security, logs, and deployment."
    }
    context = system_prompt_map.get(req.dept, "You are a general AI assistant.")
    
    # 3. Hybrid retrieval with citation sources
    try:
        from tools.retrieve_logic import _do_retrieve
        result_str = _do_retrieve(req.message, metadata_filter=metadata_filter, hybrid=True)
        
        # Parse citations from result
        sources = []
        current_src = None
        for line in result_str.split('\n'):
            if line.startswith('[Source #'):
                # Extract metadata from source line
                meta_part = line.replace('[Source', ' ').replace(']', ':')
                sources.append({"id": line.split('#')[1].strip().rstrip(']'), "meta": meta_part.strip()})
                current_src = len(sources)
            elif line.startswith('---'):
                continue
            elif current_src and not line.startswith('['):
                sources[current_src-1]["preview"] = line[:200]
        
        reply = (f"[System]:\n{context}\n\n"
                 f"[User Question]:\n{req.message}\n\n"
                 f"[Retrieved Context]:\n{result_str}\n\n"
                 f"💡 *Answer generated based on your {req.dept.upper()} context.*")
        
        return ChatResponse(reply=reply, images=req.files, sources=sources)
    
    except Exception as e:
        import traceback
        return ChatResponse(reply=f"[ERROR] {str(e)}\n{traceback.format_exc()}")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(APP_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()
