"""
smart_qa_server.py — Enterprise RAG UI Server
Lazy Stack: FastAPI (Async) + Vanilla JS (No framework overhead)
Handles: Login, Session Management, Tool Routing, Image/File Serving
"""
import os
import json
import base64
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Environment Setup ---
APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
STATIC_DIR = UPLOAD_DIR / "images"
os.makedirs(STATIC_DIR, exist_ok=True)

# --- Mock User Database (Simulating Org Context) ---
MOCK_USERS = {
    "zhangsan": {"password": "123", "name": "张三", "dept": "sales", "level": 3},
    "lisi":     {"password": "123", "name": "李四", "dept": "engineering", "level": 6},
    "wangwu":  {"password": "123", "name": "王五", "dept": "admin",     "level": 9}
}

# --- App Initialization ---
app = FastAPI(title="智链星图 - Enterprise Agentic RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], allow_headers=["*"]
)

# Serve uploaded images statically
if STATIC_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(STATIC_DIR)), name="uploads")

# --- Pydantic Models ---
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    files: Optional[List[str]] = [] # Base64 strings
    username: str = "" 
    dept: str = ""

class ChatResponse(BaseModel):
    reply: str
    images: list[str] = []

# --- Core Logic ---

@app.post("/api/login")
def login(req: LoginRequest):
    """Verify credentials and issue session context."""
    user = MOCK_USERS.get(req.username)
    if user and user["password"] == req.password:
        return {"success": True, "token": f"session_{req.username}", "user_info": user}
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Save uploaded file and return relative URL."""
    unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    dest_path = STATIC_DIR / unique_name
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"url": f"/uploads/{unique_name}"}

@app.post("/api/chat")
def rag_chat(req: ChatRequest):
    """Core business logic: Parse Identity -> Route -> Execute."""
    
    # 1. Build Metadata Filter based on Department & Clearance
    metadata_filter = {
        "$and": [
            {"visibility": {"$gte": 0}},
            {"visibility": {"$lte": MOCK_USERS[req.username]["level"]} }
        ]
    }
    if req.dept:
        metadata_filter["$and"].append({"dept": req.dept})

    # 2. Inject Context into Prompt (Router Shunting)
    system_prompt_map = {
        "sales": "You are an expert Sales Assistant. Tone: Polite, persuasive, focus on pricing.",
        "engineering": "You are a Senior Engineer. Tone: Technical, precise, focus on troubleshooting.",
        "admin": "You are a System Administrator. Focus on security logs and deployment status."
    }
    
    context = system_prompt_map.get(req.dept, "You are a general AI assistant.")
    
    # 3. Core Retrieval
    try:
        from tools.retrieve_logic import _do_retrieve
        docs = _do_retrieve(req.message, metadata_filter=metadata_filter)
        
        final_query = f"[System]:\n{context}\n\n[User Question]:\n{req.message}\n\n[Context]:\n{docs}"
        
        # In a real app, you'd pass 'final_query' to your LLM here
        reply = f"(🤖 Agent Reply):\n\n---\n{docs}\n---\n\n💡 *Answer generated based on your {req.dept.upper()} context.*"
        
        return ChatResponse(reply=reply, images=req.files)
    except Exception as e:
        return ChatResponse(reply=f"[ERROR] {str(e)}")

# Serve the UI
@app.get("/", response_class=HTMLResponse)
def index():
    with open(APP_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()
