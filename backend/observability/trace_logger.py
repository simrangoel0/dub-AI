from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# If you want type hints later:
# from agents.context_manager_models import ContextSelectionResult, ScoredChunk

# Resolve repo root and ensure /traces exists
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
    Collects all data for a single pipeline run
    and writes it to /traces/{timestamp}_{run_id}.json.

    This is Dev B's custom JSON "flight recorder".
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

        - user_query: the current question
        - conversation_history: chat so far, list of {role, content}
        - top_k: how many chunks context manager tried to select
        - initial_boost_chunks: any chunk_ids the user boosted for this run
        - repo_state_summary: lightweight info about the repo, e.g. { "files": [...] }
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

        Expected shape (Pydantic):

            class ContextSelectionResult(BaseModel):
                query: str
                top_k: int
                selected_chunks: List[ScoredChunk]
                dropped_chunks: List[ScoredChunk]
                meta: Dict[str, Any] = {}

        Where ScoredChunk has:
            chunk: Chunk
            similarity_score: float
            relevance_score: float
            rationale: str

        This writes:
          - steps.retrieve_step
          - steps.scoring_step
        """
        selected_chunks = ctx_result.get("selected_chunks", [])
        dropped_chunks = ctx_result.get("dropped_chunks", [])
        top_k = ctx_result.get("top_k")
        query = ctx_result.get("query")
        meta = ctx_result.get("meta", {})

        # 1) retrieve_step: mostly a direct mirror of ContextSelectionResult
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
            # Dev A1 can add more details into ctx_result.meta["scoring_notes"]
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

        Example summaries:
            [
              {
                "summary_id": "sum_1",
                "text": "High-level description...",
                "source_chunk_ids": ["chunk_2", "chunk_5"]
              },
              ...
            ]
        """
        self._data["steps"]["summary_step"] = {
            "used": used,
            "reason": reason,
            "summaries": summaries or [],
        }

    # -------------------------------------------------------------------------
    # STEP: ANSWER AGENT (Dev A2)
    # -------------------------------------------------------------------------

    def log_answer_step(self, answer_result: Dict[str, Any]) -> None:
        """
        Log Dev A2's answer agent output.

        Expected keys in answer_result:
            - final_prompt: str
            - final_code: str
            - answer_text: Optional[str]
            - raw_llm_output: Optional[dict]
        """
        final_prompt = answer_result.get("final_prompt")
        final_code = answer_result.get("final_code")
        answer_text = answer_result.get("answer_text")
        raw_llm_output = answer_result.get("raw_llm_output")

        self._data["steps"]["final_prompt"] = final_prompt
        self._data["steps"]["answer"] = {
            "final_code": final_code,
            "answer_text": answer_text,
            "raw_llm_output": raw_llm_output,
        }

    # -------------------------------------------------------------------------
    # STEP: ATTRIBUTION (Dev A2)
    # -------------------------------------------------------------------------

    def log_attribution_step(self, attribution_result: Dict[str, Any]) -> None:
        """
        Log Dev A2's attribution agent output.

        Expected keys in attribution_result:
            - influence_map: Dict[str, float]   (chunk_id -> score 0–1)
            - explanations: Optional[Dict[str, str]]
        """
        influence_map_raw = attribution_result.get("influence_map", {})

        # Convert dict[{chunk_id: score}] to a list of entries for easier plotting
        entries = [
            {"chunk_id": chunk_id, "score": float(score)}
            for chunk_id, score in influence_map_raw.items()
        ]

        total = sum(e["score"] for e in entries)
        normalized = abs(total - 1.0) < 1e-3 if entries else False

        self._data["steps"]["influence_map"] = {
            "entries": entries,
            "normalized": normalized,
        }

        explanations = attribution_result.get("explanations")
        if explanations:
            self._data["steps"]["attribution_details"] = {
                "explanations": explanations
            }

    # -------------------------------------------------------------------------
    # BOOST INFO (still useful even without rerun semantics)
    # -------------------------------------------------------------------------

    def log_boost(
        self,
        *,
        boost_chunks: Optional[List[str]],
        reason: Optional[str] = None,
    ) -> None:
        """
        Record boost actions for this run.

        - boost_chunks: list of chunk_ids the user boosted (if any)
        - reason: short string, e.g. "initial run", "user boosted chunks in UI"
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
    Helper for /trace/{id}: find the latest file matching *_{run_id}.json
    and return its JSON content as a dict.
    """
    matches = sorted(TRACES_DIR.glob(f"*_{run_id}.json"))
    if not matches:
        return None

    path = matches[-1]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

