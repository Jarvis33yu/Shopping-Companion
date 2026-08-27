# File: src/serve_mem_fastapi.py

import os
import sys
import argparse
from typing import Dict, Optional
from tqdm import tqdm
from fastapi import FastAPI, Query, HTTPException
import uvicorn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mem.retriever import Retriever
from sentence_transformers import SentenceTransformer


app = FastAPI(
    title="Memory Retrieval Service",
    description="FastAPI version of memory retrieval backend",
    version="1.0"
)

# ================================
# Global Variables
# ================================

server: Optional['RetrievalServer'] = None


# ================================
# Core Logic (unchanged)
# ================================

class RetrievalServer:
    def __init__(self, index_dir: str, sentence_model_name: str):
        self.retrievers: Dict[str, Retriever] = {}

        print(f"🔍 Loading Sentence Transformer model: {sentence_model_name}")
        sentence_model = SentenceTransformer(sentence_model_name)

        print(f"📁 Loading memory indices from: {index_dir}")
        for conversation_id in tqdm(os.listdir(index_dir), desc="Loading retrievers"):
            retriever = Retriever(sentence_model=sentence_model)
            try:
                retriever.load_index(index_dir=index_dir, conversation_id=conversation_id)
                self.retrievers[conversation_id] = retriever
                print(f"✅ Loaded retriever for conversation_id='{conversation_id}'")
            except Exception as e:
                print(f"❌ Failed to load retriever for '{conversation_id}': {e}", file=sys.stderr)

        if not self.retrievers:
            raise RuntimeError(f"No valid retrievers loaded from {index_dir}")

    def call(self, conversation_id: str, **kwargs):
        if conversation_id not in self.retrievers:
            print(f"Conversation ID {conversation_id} not found", file=sys.stderr)
            return None

        retriever = self.retrievers[conversation_id]
        name = kwargs.get("name")

        try:
            if name == "search":
                return retriever.search(query=kwargs.get("query"), k=int(kwargs.get("k")))
            elif name == "view_session":
                return retriever.view_session(idx=int(kwargs.get("idx")))
            elif name == "view_prev_session":
                return retriever.view_prev_session(idx=int(kwargs.get("idx")))
            elif name == "view_next_session":
                return retriever.view_next_session(idx=int(kwargs.get("idx")))
            elif name == "view_sessions_by_date":
                return retriever.view_sessions_by_date(
                    start_date=kwargs.get("start_date"),
                    end_date=kwargs.get("end_date")
                )
            elif name == "find_similar_memories":
                return retriever.find_similar_memories(idx=int(kwargs.get("idx")), k=int(kwargs.get("k")))
            else:
                print(f"Invalid name {name}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"Error during {name} for {conversation_id}: {e}", file=sys.stderr)
            return None


# ====================
# API Routes
# ====================

@app.get("/")
def index():
    return {
        "usage": {
            "/search": "conversation_id,query,k",
            "/view_session": "conversation_id,idx",
            "/view_prev_session": "conversation_id,idx",
            "/view_next_session": "conversation_id,idx",
            "/view_sessions_by_date": "conversation_id,start_date,end_date",
            "/find_similar_memories": "conversation_id,idx,k",
        }
    }


@app.get("/search")
def search(
    conversation_id: str = Query(..., description="Conversation ID"),
    query: str = Query(..., description="Search query"),
    k: int = Query(10, ge=1, le=100, description="Number of results"),
):
    if not server:
        raise HTTPException(status_code=500, detail="Retrieval server not initialized")
    result = server.call(conversation_id=conversation_id, name="search", query=query, k=k)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found or error occurred")
    return result


@app.get("/view_session")
def view_session(
    conversation_id: str = Query(..., description="Conversation ID"),
    idx: int = Query(..., description="Memory index"),
):
    result = server.call(conversation_id=conversation_id, name="view_session", idx=idx)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation or session not found")
    return result


@app.get("/view_prev_session")
def view_prev_session(
    conversation_id: str = Query(..., description="Conversation ID"),
    idx: int = Query(..., description="Current index"),
):
    result = server.call(conversation_id=conversation_id, name="view_prev_session", idx=idx)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation or session not found")
    return result


@app.get("/view_next_session")
def view_next_session(
    conversation_id: str = Query(..., description="Conversation ID"),
    idx: int = Query(..., description="Current index"),
):
    result = server.call(conversation_id=conversation_id, name="view_next_session", idx=idx)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation or session not found")
    return result


@app.get("/view_sessions_by_date")
def view_sessions_by_date(
    conversation_id: str = Query(..., description="Conversation ID"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD"),
):
    result = server.call(
        conversation_id=conversation_id,
        name="view_sessions_by_date",
        start_date=start_date,
        end_date=end_date
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return result


@app.get("/find_similar_memories")
def find_similar_memories(
    conversation_id: str = Query(..., description="Conversation ID"),
    idx: int = Query(..., description="Target memory index"),
    k: int = Query(10, ge=1, le=100, description="Number of similar memories"),
):
    result = server.call(conversation_id=conversation_id, name="find_similar_memories", idx=idx, k=k)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation or memory not found")
    return result


# ==============================
# Startup & Shutdown Events
# ==============================

@app.on_event("startup")
def startup_event():
    import os
    global server

    index_dir = os.getenv("MEM_INDEX_DIR", "data/mem_indexes")
    sentence_model_name = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5632))

    print(f"\n🚀 Starting Memory Retrieval Service")
    print(f"  Index Dir: {index_dir}")
    print(f"  Model: {sentence_model_name}")
    print(f"  Listen: {host}:{port}")
    print(f"  PID: {os.getpid()}\n")

    try:
        server = RetrievalServer(index_dir=index_dir, sentence_model_name=sentence_model_name)
        print("✅ Memory retrieval service ready!")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        raise


@app.on_event("shutdown")
def shutdown_event():
    global server
    print("👋 Memory retrieval service shutting down...")
    if server:
        # 如果 Retriever 有 cleanup 方法，可以在这里调用
        pass
    print("✅ Shutdown complete.")


# nohup uvicorn agentic_rl.serve_mem:app --host 0.0.0.0 --port 5632 --workers 2 --limit-concurrency 1000 --timeout-keep-alive 30 --log-level info > logs/serve_mem 2>&1 &

