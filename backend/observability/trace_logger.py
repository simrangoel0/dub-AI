from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
TRACES_DIR = BASE_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class TraceLogger:
    """
    Writes a complete JSON record of the run
    for the glass box frontend visualisation layer.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _data: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._data = {
            "run_id": self.run_id,
            "created_at": _now_iso(),
            "input": {},
            "steps": {},
            "boost": {
                "applied": False,
                "boosted_chunks": [],
                "reason": None,
            },
        }

    # INPUT SECTION

    def set_input(
        self,
        *,
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        initial_boost_chunks: Optional[List[str]] = None,
        repo_state_summary: Optional[Dict[str, Any]] = None,
    ) -> None:

        self._data["input"] = {
            "user_query": user_query,
            "conversation_history": conversation_history or [],
            "top_k": top_k,
            "initial_boost_chunks": initial_boost_chunks or [],
            "repo_state_summary": repo_state_summary,
        }

    # CONTEXT SELECTION STEP

    def log_context_selection(self, ctx_result: Dict[str, Any]) -> None:

        selected_chunks = ctx_result.get("selected_chunks", [])
        dropped_chunks = ctx_result.get("dropped_chunks", [])
        top_k = ctx_result.get("top_k")
        query = ctx_result.get("query")
        meta = ctx_result.get("meta", {})

        retrieve_step = {
            "query": query,
            "top_k": top_k,
            "selected_chunks": selected_chunks,
            "dropped_chunks": dropped_chunks,
            "meta": meta,
        }

        scores_by_chunk_id: Dict[str, Any] = {}

        def _add(scored_chunk, flag):
            chunk = scored_chunk["chunk"]
            cid = chunk["chunk_id"]
            scores_by_chunk_id[cid] = {
                "chunk_id": cid,
                "file_path": chunk["file_path"],
                "similarity_score": scored_chunk["similarity_score"],
                "relevance_score": scored_chunk["relevance_score"],
                "rationale": scored_chunk["rationale"],
                "is_selected": flag,
            }

        for sc in selected_chunks:
            _add(sc, True)
        for sc in dropped_chunks:
            _add(sc, False)

        scoring_step = {
            "scores_by_chunk_id": scores_by_chunk_id,
            "policy_notes": meta.get("scoring_notes", []),
        }

        self._data["steps"]["retrieve_step"] = retrieve_step
        self._data["steps"]["scoring_step"] = scoring_step

        if self._data["input"].get("top_k") is None:
            self._data["input"]["top_k"] = top_k

    # SUMMARY (optional)

    def log_summary_step(self, *, used: bool, reason: Optional[str], summaries: Optional[List[Dict[str, Any]]] = None):
        self._data["steps"]["summary_step"] = {
            "used": used,
            "reason": reason,
            "summaries": summaries or [],
        }

    # ANSWER STEP

    def log_answer_step(self, answer_result: Dict[str, Any]) -> None:

        final_prompt = answer_result.get("full_prompt")
        final_answer = answer_result.get("final_answer")
        used_chunks = answer_result.get("used_chunks", [])

        self._data["steps"]["final_prompt"] = final_prompt
        self._data["steps"]["answer"] = {
            "answer_text": final_answer,
            "used_chunks": used_chunks,
        }

    # ATTRIBUTION STEP

    def log_attribution_step(self, attribution_result: Dict[str, Any]) -> None:

        scores = attribution_result.get("influence_scores", {}) or {}

        entries = [{"chunk_id": cid, "score": float(score)} for cid, score in scores.items()]
        total = sum(e["score"] for e in entries)
        normalized = abs(total - 1.0) < 1e-3 if entries else False

        self._data["steps"]["influence_map"] = {
            "entries": entries,
            "normalized": normalized,
        }

        if attribution_result.get("raw") is not None:
            self._data["steps"]["attribution_details"] = attribution_result["raw"]

    # BOOST

    def log_boost(self, *, boost_chunks: Optional[List[str]], reason: Optional[str] = None):
        self._data["boost"] = {
            "applied": bool(boost_chunks),
            "boosted_chunks": boost_chunks or [],
            "reason": reason,
        }

    # SAVE

    def save(self) -> str:

        ts = _timestamp_for_filename()
        path = TRACES_DIR / f"{ts}_{self.run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        return str(path)


def load_trace_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
        matches = sorted(TRACES_DIR.glob(f"*_{run_id}.json"))
        if not matches:
            return None
        with matches[-1].open("r", encoding="utf-8") as f:
            return json.load(f)
