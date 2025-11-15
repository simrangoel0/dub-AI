from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.pipeline import run_pipeline
from backend.observability.trace_logger import load_trace_by_run_id
from backend.observability.langsmith_setup import init_langsmith


# ---------- Pydantic request/response models ----------

class RunRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, Any]]] = None
    repo_state: Optional[Dict[str, Any]] = None
    selected_chunk_ids: Optional[List[str]] = None
    top_k: Optional[int] = 8


class RunResponse(BaseModel):
    run_id: str
    answer: Dict[str, Any]
    context: Dict[str, Any]
    attribution: Dict[str, Any]


class TraceResponse(BaseModel):
    run_id: str
    created_at: str
    input: Dict[str, Any]
    steps: Dict[str, Any]
    boost: Dict[str, Any]


# ---------- FastAPI app ----------

app = FastAPI(title="Glass-Box Debugging Agent API")

# Initialise LangSmith once (optional – will no-op if no key)
init_langsmith(project_name="glass-box-debugger")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run_endpoint(body: RunRequest):
    """
    Main entrypoint for a debugging run.

    Frontend sends:
      - query
      - optional history (chat messages)
      - optional selected_chunk_ids (user-selected context)
      - optional top_k

    We call run_pipeline, then return a slimmed-down response
    (run_id + answer + context + attribution).
    """
    payload: Dict[str, Any] = {
        "query": body.query,
        "history": body.history or [],
        "repo_state": body.repo_state or {},
        "selected_chunk_ids": body.selected_chunk_ids,
        "top_k": body.top_k or 8,
    }

    result = run_pipeline(payload)

    return RunResponse(
        run_id=result["run_id"],
        answer=result["answer"],
        context=result["context"],
        attribution=result["attribution"],
    )


@app.get("/trace/{run_id}", response_model=TraceResponse)
def trace_endpoint(run_id: str):
    """
    Return the full JSON trace for a given run_id.

    This is what Dev C will use to render:
      - selected vs dropped context
      - scores
      - influence map
      - etc.
    """
    data = load_trace_by_run_id(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Trace not found")

    # data already has run_id, created_at, input, steps, boost
    return TraceResponse(
        run_id=data["run_id"],
        created_at=data["created_at"],
        input=data["input"],
        steps=data["steps"],
        boost=data["boost"],
    )
