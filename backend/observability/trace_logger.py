from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve repo root and ensure /traces exists
# (…/backend/observability/trace_logger.py -> parents[2] = repo root)
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
    Collects all data for a single pipeline run and writes it to
    /traces/{timestamp}_{run_id}.json.

    This is your custom JSON "flight recorder" for the glassbox UI.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _data: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._data = {
            "run_id": self.run_id,
            "created_at": _now_iso(),
            "input": {},
            "steps": {},
            "boost": {  # you can later rename this to "selection" if you want
                "applied": False,
                "boosted_chunks": [],
                "reason": None,
            },
        }

    # -------------------------------------------------------------------------
    # INPUT SECTION
    # -------------------------------------------------------------------------

    def set_input(
        self,
        *,
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        initial_boost_chunks: Optional[List[str]] = None,
        repo_state_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store high-level inputs for this run.
        """
        self._data["input"] = {
            "user_query": user_query,
            "conversation_history": conversation_history or [],
            "top_k": top_k,
            "initial_boost_chunks": initial_boost_chunks or [],
            "repo_state_summary": repo_state_summary,
        }

    # -------------------------------------------------------------------------
    # STEP: CONTEXT SELECTION (Dev A1)
    # -------------------------------------------------------------------------

    def log_context_selection(self, ctx_result: Dict[str, Any]) -> None:
        """
        Log Dev A1's ContextSelectionResult.

        Expected shape (after .model_dump()):
            {
              "query": str,
              "top_k": int,
              "selected_chunks": List[ScoredChunk],
              "dropped_chunks": List[ScoredChunk],
              "meta": Dict[str, Any]
            }
        """
        selected_chunks = ctx_result.get("selected_chunks", [])
        dropped_chunks = ctx_result.get("dropped_chunks", [])
        top_k = ctx_result.get("top_k")
        query = ctx_result.get("query")
        meta = ctx_result.get("meta", {})

        # 1) retrieve_step: mirror ContextSelectionResult
        retrieve_step = {
            "query": query,
            "top_k": top_k,
            "selected_chunks": selected_chunks,
            "dropped_chunks": dropped_chunks,
            "meta": meta,
        }

        # 2) scoring_step: flatten into a chunk_id → scores map
        scores_by_chunk_id: Dict[str, Any] = {}

        def _add_scored_chunk(scored_chunk: Dict[str, Any], is_selected: bool) -> None:
            chunk = scored_chunk["chunk"]
            chunk_id = chunk["chunk_id"]

            scores_by_chunk_id[chunk_id] = {
                "chunk_id": chunk_id,
                "file_path": chunk["file_path"],
                "similarity_score": scored_chunk["similarity_score"],
                "relevance_score": scored_chunk["relevance_score"],
                "rationale": scored_chunk["rationale"],
                "is_selected": is_selected,
            }

        for sc in selected_chunks:
            _add_scored_chunk(sc, is_selected=True)

        for sc in dropped_chunks:
            _add_scored_chunk(sc, is_selected=False)

        scoring_step = {
            "scores_by_chunk_id": scores_by_chunk_id,
            "policy_notes": meta.get("scoring_notes", []),
        }

        self._data["steps"]["retrieve_step"] = retrieve_step
        self._data["steps"]["scoring_step"] = scoring_step

        # keep top_k in input in sync if it was None before
        if self._data["input"].get("top_k") is None:
            self._data["input"]["top_k"] = top_k

    # -------------------------------------------------------------------------
    # STEP: SUMMARY (Dev A1, optional)
    # -------------------------------------------------------------------------

    def log_summary_step(
        self,
        *,
        used: bool,
        reason: Optional[str] = None,
        summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Optional: log summarisation information when the context manager compresses chunks.
        """
        self._data["steps"]["summary_step"] = {
            "used": used,
            "reason": reason,
            "summaries": summaries or [],
        }

    # -------------------------------------------------------------------------
    # STEP: ANSWER AGENT (Dev A2 – final answer)
    # -------------------------------------------------------------------------

    def log_answer_step(self, answer_result: Dict[str, Any]) -> None:
        """
        Log Dev A2's AnswerAgent output.

        With the current AnswerAgent, we expect:
            answer_result = {
                "final_answer": str,      # human-readable answer
                "used_chunks": [str, ...],
                "full_prompt": str,       # full system + user prompt
            }

        We store:
            steps.final_prompt
            steps.answer.answer_text
            steps.answer.used_chunks
        """
        final_prompt = answer_result.get("full_prompt")
        final_answer = answer_result.get("final_answer")
        used_chunks = answer_result.get("used_chunks", [])

        self._data["steps"]["final_prompt"] = final_prompt
        self._data["steps"]["answer"] = {
            "final_code": None,          # reserved for future patch/diff
            "answer_text": final_answer,
            "used_chunks": used_chunks,
        }

    # -------------------------------------------------------------------------
    # STEP: ATTRIBUTION (Dev A2 – influence scores only)
    # -------------------------------------------------------------------------

    def log_attribution_step(self, attribution_result: Dict[str, Any]) -> None:
        """
        Log Dev A2's AttributionAgent output.

        We expect:
            attribution_result = {
                "influence_scores": { chunk_id: float in [0, 1], ... },
                "raw": <full AttributionOutput dict>   # optional
            }

        We convert that into:
            steps.influence_map = {
              "entries": [ {chunk_id, score}, ... ],
              "normalized": bool
            }
            steps.attribution_details = raw   (if present)
        """
        scores = attribution_result.get("influence_scores", {}) or {}

        entries = [
            {"chunk_id": chunk_id, "score": float(score)}
            for chunk_id, score in scores.items()
        ]

        total = sum(e["score"] for e in entries)
        normalized = abs(total - 1.0) < 1e-3 if entries else False

        self._data["steps"]["influence_map"] = {
            "entries": entries,
            "normalized": normalized,
        }

        raw_details = attribution_result.get("raw")
        if raw_details is not None:
            self._data["steps"]["attribution_details"] = raw_details

    # -------------------------------------------------------------------------
    # BOOST / SELECTION METADATA (optional)
    # -------------------------------------------------------------------------

    def log_boost(
        self,
        *,
        boost_chunks: Optional[List[str]],
        reason: Optional[str] = None,
    ) -> None:
        """
        Record selection / "boost" metadata for this run.

        - boost_chunks: list of chunk_ids the UI considered especially important (if any)
        - reason: short string, e.g. "single run", "user emphasised these chunks"
        """
        applied = bool(boost_chunks)
        self._data["boost"] = {
            "applied": applied,
            "boosted_chunks": boost_chunks or [],
            "reason": reason,
        }

    # -------------------------------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def save(self) -> str:
        """
        Write the trace JSON to /traces/{timestamp}_{run_id}.json.
        Returns the file path as a string.
        """
        ts = _timestamp_for_filename()
        filename = f"{ts}_{self.run_id}.json"
        path = TRACES_DIR / filename

        with path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

        return str(path)


def load_trace_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Helper for /trace/{id}:
    find the latest file matching *_{run_id}.json and return its JSON content.
    """
    matches = sorted(TRACES_DIR.glob(f"*_{run_id}.json"))
    if not matches:
        return None

    path = matches[-1]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


