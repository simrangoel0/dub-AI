from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.pipeline import run_pipeline
from backend.observability.trace_logger import load_trace_by_run_id
from backend.observability.langsmith_setup import init_langsmith

from backend.db import (
    SessionLocal,
    ChatMessage,
    Run,
    ChunkModel,
    RunChunk,
    AttributionRow,
    ContextModification,
)

# ---------- Pydantic request/response models ----------

class RunRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, Any]]] = None
    repo_state: Optional[Dict[str, Any]] = None
    selected_chunk_ids: Optional[List[str]] = None
    top_k: Optional[int] = 8
    conversation_id: Optional[str] = "default"


class RunResponse(BaseModel):
    run_id: str
    message_id: str
    trace_file: Optional[str]
    answer: Dict[str, Any]
    context: Dict[str, Any]
    attribution: Dict[str, Any]
    summary: Optional[str] = None


class TraceResponse(BaseModel):
    run_id: str
    created_at: str
    input: Dict[str, Any]
    steps: Dict[str, Any]
    boost: Dict[str, Any]


class ChatMessageOut(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: datetime


class TimelineItem(BaseModel):
    run_id: str
    message_id: str
    conversation_id: str
    user_query: str
    created_at: datetime
    selection_mode: Optional[str] = None
    top_k: Optional[int] = None
    
    runLabel: Optional[str] = None      
    summary: Optional[str] = None      


class ContextNodeStats(BaseModel):
    times_considered: int
    times_selected: int
    avg_influence: float


class ContextNode(BaseModel):
    chunk_id: str
    source: str
    file_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    text_preview: str
    stats: Optional[ContextNodeStats] = None


class ContextGraphResponse(BaseModel):
    nodes: List[ContextNode]


class RunContextChunk(BaseModel):
    chunk_id: str
    file_path: str
    source: str
    start_line: Optional[int]
    end_line: Optional[int]
    text: str
    similarity_score: Optional[float] = None
    relevance_score: Optional[float] = None
    is_selected: bool
    selection_rationale: Optional[str] = None
    influence_score: float = 0.0
    explanation: Optional[str] = None
    evidence: List[str] = []


class RunContextResponse(BaseModel):
    run_id: str
    selected: List[RunContextChunk]
    dropped: List[RunContextChunk]


class ContextModifyRequest(BaseModel):
    run_id: str
    chunk_id: str
    action: str           # "boost" | "exclude" | "pin" | ...
    reason: Optional[str] = None


class ContextModifyResponse(BaseModel):
    status: str
    modification_id: int


class ChunkDetailStats(BaseModel):
    times_considered: int
    times_selected: int
    avg_influence: float


class ChunkDetailResponse(BaseModel):
    chunk_id: str
    source: str
    file_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    text: str
    meta: Optional[Dict[str, Any]] = None
    stats: ChunkDetailStats


# ---------- FastAPI app ----------

app = FastAPI(title="Glass-Box Debugging Agent API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise LangSmith once (optional – will no-op if no key)
init_langsmith(project_name="glass-box-debugger")


# In-memory registry of WebSocket connections (very simple)
connected_trace_clients: List[WebSocket] = []


# ---------- Health ----------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------- Main pipeline run ----------

@app.post("/api/run", response_model=RunResponse)
def api_run(body: RunRequest):
    """
    Main entrypoint for a debugging run.
    """
    payload: Dict[str, Any] = {
        "query": body.query,
        "history": body.history or [],
        "repo_state": body.repo_state or {},
        "selected_chunk_ids": body.selected_chunk_ids,
        "top_k": body.top_k or 8,
        "conversation_id": body.conversation_id or "default",
    }

    result = run_pipeline(payload)
    return RunResponse(
        run_id=result["run_id"],
        message_id=result["message_id"],
        trace_file=result.get("trace_file"),
        answer=result["answer"],
        context=result["context"],
        attribution=result["attribution"],
        summary=result.get("summary"),
    )


# Backwards-compatible alias
@app.post("/run", response_model=RunResponse)
def run_endpoint(body: RunRequest):
    return api_run(body)


# ---------- Trace by run_id (from JSON trace files) ----------

@app.get("/api/trace/{run_id}", response_model=TraceResponse)
def api_trace(run_id: str):
    data = load_trace_by_run_id(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Trace not found")

    return TraceResponse(
        run_id=data["run_id"],
        created_at=data["created_at"],
        input=data["input"],
        steps=data["steps"],
        boost=data["boost"],
    )


# Backwards-compatible alias
@app.get("/trace/{run_id}", response_model=TraceResponse)
def trace_endpoint(run_id: str):
    return api_trace(run_id)


# ---------- 1. GET /api/chat/messages ----------

@app.get("/api/chat/messages", response_model=List[ChatMessageOut])
def get_chat_messages(
    conversation_id: str = Query("default"),
    limit: int = Query(200, ge=1, le=1000),
):
    db = SessionLocal()
    try:
        q = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at.asc())
        )
        if limit:
            q = q.limit(limit)

        rows = q.all()
        return [
            ChatMessageOut(
                id=row.id,
                conversation_id=row.conversation_id,
                role=row.role,
                content=row.content,
                created_at=row.created_at,
            )
            for row in rows
        ]
    finally:
        db.close()


# ---------- 2. GET /api/trace/timeline ----------

@app.get("/api/trace-timeline", response_model=List[TimelineItem])
def get_trace_timeline(limit: int = Query(50, ge=1, le=500)):
    db = SessionLocal()
    try:
        rows = (
            db.query(Run)
            .order_by(Run.created_at.asc())  # oldest first so numbering makes sense
            .limit(limit)
            .all()
        )

        timeline = []
        for i, row in enumerate(rows, start=1):
            timeline.append(
                TimelineItem(
                    run_id=row.run_id,
                    message_id=row.message_id,
                    conversation_id=row.conversation_id,
                    user_query=row.user_query,
                    created_at=row.created_at,
                    selection_mode=row.selection_mode,
                    top_k=row.top_k,
                    runLabel=f"Debug run {i}",     
                    summary=row.run_label,         # <-- SEND SUMMARY 
                )
            )
        return timeline
    finally:
        db.close()


# ---------- 3. GET /api/context/graph ----------

@app.get("/api/context/graph", response_model=ContextGraphResponse)
def get_context_graph():
    """
    Simple graph: just nodes + stats per chunk.
    Edges can be inferred on frontend (same file, etc.).
    """
    db = SessionLocal()
    try:
        chunks = db.query(ChunkModel).all()

        # Stats per chunk_id
        # times_considered + times_selected from RunChunk
        # avg_influence from AttributionRow
        times_considered: Dict[str, int] = {}
        times_selected: Dict[str, int] = {}
        influence_sum: Dict[str, float] = {}
        influence_count: Dict[str, int] = {}

        for rc in db.query(RunChunk).all():
            cid = rc.chunk_id
            times_considered[cid] = times_considered.get(cid, 0) + 1
            if rc.is_selected:
                times_selected[cid] = times_selected.get(cid, 0) + 1

        for ar in db.query(AttributionRow).all():
            cid = ar.chunk_id
            influence_sum[cid] = influence_sum.get(cid, 0.0) + float(ar.influence_score)
            influence_count[cid] = influence_count.get(cid, 0) + 1

        nodes: List[ContextNode] = []
        for c in chunks:
            preview = c.text[:200] + "..." if len(c.text) > 200 else c.text

            tc = times_considered.get(c.chunk_id, 0)
            ts = times_selected.get(c.chunk_id, 0)
            if influence_count.get(c.chunk_id, 0) > 0:
                avg_inf = influence_sum[c.chunk_id] / influence_count[c.chunk_id]
            else:
                avg_inf = 0.0

            stats = ContextNodeStats(
                times_considered=tc,
                times_selected=ts,
                avg_influence=avg_inf,
            )

            nodes.append(
                ContextNode(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    file_path=c.file_path,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    text_preview=preview,
                    stats=stats,
                )
            )

        return ContextGraphResponse(nodes=nodes)
    finally:
        db.close()


# ---------- 4. GET /api/trace/run/{run_id}/context ----------

@app.get("/api/trace/run/{run_id}/context", response_model=RunContextResponse)
def get_run_context(run_id: str):
    db = SessionLocal()
    try:
        # Make sure run exists
        run = db.query(Run).filter(Run.run_id == run_id).one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        run_chunks = (
            db.query(RunChunk)
            .filter(RunChunk.run_id == run_id)
            .all()
        )

        attributions = (
            db.query(AttributionRow)
            .filter(AttributionRow.run_id == run_id)
            .all()
        )
        attr_map = {a.chunk_id: a for a in attributions}

        def to_rc_chunk(rc: RunChunk) -> RunContextChunk:
            attr = attr_map.get(rc.chunk_id)
            influence_score = float(attr.influence_score) if attr else 0.0
            explanation = attr.explanation if attr else None
            evidence: List[str] = []
            if attr and attr.evidence_json:
                try:
                    import json
                    evidence = json.loads(attr.evidence_json)
                except Exception:
                    evidence = []

            return RunContextChunk(
                chunk_id=rc.chunk_id,
                file_path=rc.file_path,
                source=rc.source,
                start_line=rc.start_line,
                end_line=rc.end_line,
                text=rc.text,
                similarity_score=rc.similarity_score,
                relevance_score=rc.relevance_score,
                is_selected=rc.is_selected,
                selection_rationale=rc.selection_rationale,
                influence_score=influence_score,
                explanation=explanation,
                evidence=evidence,
            )

        selected: List[RunContextChunk] = []
        dropped: List[RunContextChunk] = []
        for rc in run_chunks:
            chunk = to_rc_chunk(rc)
            if rc.is_selected:
                selected.append(chunk)
            else:
                dropped.append(chunk)

        return RunContextResponse(
            run_id=run_id,
            selected=selected,
            dropped=dropped,
        )
    finally:
        db.close()


# ---------- 5. POST /api/context/modify ----------

@app.post("/api/context/modify", response_model=ContextModifyResponse)
def modify_context(body: ContextModifyRequest):
    db = SessionLocal()
    try:
        # Optional: ensure run exists
        run = db.query(Run).filter(Run.run_id == body.run_id).one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        cm = ContextModification(
            run_id=body.run_id,
            chunk_id=body.chunk_id,
            action=body.action,
            reason=body.reason,
        )
        db.add(cm)
        db.commit()
        db.refresh(cm)

        return ContextModifyResponse(
            status="ok",
            modification_id=cm.id,
        )
    finally:
        db.close()


# ---------- 6. GET /api/context/{chunk_id} ----------

@app.get("/api/context/{chunk_id}", response_model=ChunkDetailResponse)
def get_chunk_detail(chunk_id: str):
    db = SessionLocal()
    try:
        chunk = db.query(ChunkModel).filter(ChunkModel.chunk_id == chunk_id).one_or_none()
        if chunk is None:
            raise HTTPException(status_code=404, detail="Chunk not found")

        # stats
        tc = (
            db.query(RunChunk)
            .filter(RunChunk.chunk_id == chunk_id)
            .count()
        )
        ts = (
            db.query(RunChunk)
            .filter(RunChunk.chunk_id == chunk_id, RunChunk.is_selected == True)
            .count()
        )
        inf_rows = (
            db.query(AttributionRow)
            .filter(AttributionRow.chunk_id == chunk_id)
            .all()
        )

        if inf_rows:
            total = sum(float(r.influence_score) for r in inf_rows)
            avg_inf = total / len(inf_rows)
        else:
            avg_inf = 0.0

        import json
        meta: Optional[Dict[str, Any]] = None
        if chunk.meta_json:
            try:
                meta = json.loads(chunk.meta_json)
            except Exception:
                meta = None

        return ChunkDetailResponse(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            text=chunk.text,
            meta=meta,
            stats=ChunkDetailStats(
                times_considered=tc,
                times_selected=ts,
                avg_influence=avg_inf,
            ),
        )
    finally:
        db.close()


# ---------- 7. WebSocket /ws/trace (simple stub) ----------

@app.websocket("/ws/trace")
async def ws_trace(websocket: WebSocket):
    """
    Very simple WebSocket stub.

    - Frontend connects to /ws/trace
    - Backend accepts and sends one 'connected' message
    - In a real-time setup, you would push events from TraceLogger here.
    """
    await websocket.accept()
    connected_trace_clients.append(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "Trace WebSocket connected"})
        # Echo loop (keeps connection alive, you can extend this later)
        while True:
            _ = await websocket.receive_text()
            # For now, we don't process incoming messages; just keep the socket open.
    except WebSocketDisconnect:
        if websocket in connected_trace_clients:
            connected_trace_clients.remove(websocket)