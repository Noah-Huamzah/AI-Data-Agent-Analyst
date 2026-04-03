import os
import uuid
import logging
import asyncio
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from text_to_sql import (
    get_async_client,
    query_pipeline_async,
    ConversationSession,
    ensure_history_table,
    get_history,
    cache_clear,
    load_or_build_schema_cache,
    warmup_model_async,           # called ONCE at startup — not per request
    get_pool,
    make_error,
    PIPELINE_TIMEOUT,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session store  (in-memory; swap for Redis for multi-worker deployments)
# ---------------------------------------------------------------------------

_sessions: Dict[str, ConversationSession] = {}

def get_or_create_session(session_id: Optional[str]) -> tuple:
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = ConversationSession()
        log.info(f"New session: {session_id}")
    return session_id, _sessions[session_id]

# ---------------------------------------------------------------------------
# Lifespan — startup tasks run ONCE before any request is served
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Startup: DB pool → history table → model warm-up")
    try:
        get_pool()
        ensure_history_table()
        await warmup_model_async()
    except Exception as e:
        log.warning(f"Startup task failed (non-fatal): {e}")
    
    yield
    
    log.info("Shutdown.")
    
    # Close HTTP client
    try:
        client = get_async_client()
        await client.aclose()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Text-to-SQL API",
    description="Natural language → Oracle SQL → results + insights",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str

class ConversationRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None   # omit to start a new session

# ---------------------------------------------------------------------------
# Standardised response enforcer
#
# All endpoints return the same top-level shape:
# {
#   "success":    bool,
#   "query_id":   str,
#   "answer":     str,        ← business insight
#   "sql":        str,
#   "confidence": float,
#   ...                       ← detail fields
# }
# ---------------------------------------------------------------------------

def _enforce_response_shape(result: dict) -> dict:
    """Guarantee all required top-level keys are always present."""
    defaults = {
        "success":    False,
        "query_id":   "",
        "answer":     "",
        "sql":        "",
        "confidence": 0.0,
    }
    return {**defaults, **result}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest):
    """
    Stateless single question — no conversation history.
    Includes a server-side timeout guard (PIPELINE_TIMEOUT env var, default 60 s).
    """
    result = await query_pipeline_async(req.question)
    result = _enforce_response_shape(result)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/query/conversation")
async def query_conversation(req: ConversationRequest):
    """
    Multi-turn conversation.
    Pass the returned session_id in follow-up requests to carry context forward.
    Omit or send null to start a fresh conversation.
    """
    session_id, session = get_or_create_session(req.session_id)
    result = await query_pipeline_async(req.question, session)
    result = _enforce_response_shape(result)
    return {"session_id": session_id, "result": result}


@app.delete("/query/conversation/{session_id}")
async def clear_conversation(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id].clear()
    return {"message": f"Conversation cleared for session {session_id}"}


@app.get("/history")
async def history(limit: int = Query(20, ge=1, le=200)):
    rows = get_history(limit)
    return [tuple(str(col) for col in row) for row in rows]


@app.post("/cache/clear")
async def clear_cache():
    count = cache_clear()
    return {"message": "Cache cleared", "entries_removed": count}


@app.post("/schema/refresh")
async def refresh_schema():
    try:
        load_or_build_schema_cache(force=True)
        return {"message": "Schema embeddings rebuilt successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
