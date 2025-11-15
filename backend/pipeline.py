from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core.models import ContextSelectionResult
from backend.agents.answer_agent import AnswerAgent
from backend.agents.attribution_agent import AttributionAgent
from backend.observability.trace_logger import TraceLogger
from backend.observability.traced_agents import traced_context_manager

# Instantiate A2 agents once (or do it inside run_pipeline if you prefer)
answer_agent = AnswerAgent(model_name="claude-3-5-sonnet")
attribution_agent = AttributionAgent(model_name="claude-3-5-sonnet")


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate a single glass-box debugging run:

      1. Context selection (ContextManager via traced_context_manager)
      2. Answer generation (AnswerAgent)
      3. Attribution (AttributionAgent)
      4. JSON trace logging (TraceLogger)
    """
    user_query: str = payload["query"]
    conversation_history: List[Dict[str, Any]] = payload.get("history", []) or []
    repo_state: Dict[str, Any] = payload.get("repo_state", {}) or {}

    # if your UI lets the user pick specific chunk_ids, they go here
    user_selected_chunk_ids: Optional[List[str]] = payload.get("selected_chunk_ids")

    # 1) init logger
    trace = TraceLogger()
    trace.set_input(
        user_query=user_query,
        conversation_history=conversation_history,
        top_k=None,  # will be filled from ContextSelectionResult
        initial_boost_chunks=user_selected_chunk_ids or [],
        repo_state_summary={"files": list(repo_state.keys())}
        if isinstance(repo_state, dict)
        else None,
    )

    # 2) context manager (Dev A1) → ContextSelectionResult
    ctx_result: ContextSelectionResult = traced_context_manager(
        user_query=user_query,
        conversation_history=conversation_history,
        repo_state=repo_state,
        user_selected_chunk_ids=user_selected_chunk_ids,
        top_k=payload.get("top_k", 8),
    )

    ctx_dict = ctx_result.model_dump()
    trace.log_context_selection(ctx_dict)

    # 3) answer agent (Dev A2)
    answer_result = answer_agent.run(ctx_result)
    # { "final_answer": str, "used_chunks": [ids], "full_prompt": str }
    trace.log_answer_step(answer_result)

    # 4) attribution agent (Dev A2)
    # AttributionAgent returns a structured AttributionOutput as dict
    attribution_raw = attribution_agent.run(ctx_result, answer_result)

    # Extract a simple influence_scores map from structured output.
    # We assume AttributionOutput has either:
    #   - "influences": [{chunk_id, score, ...}, ...]  OR
    #   - "chunks":     [{chunk_id, score, ...}, ...]
    influence_scores: Dict[str, float] = {}

    influences = attribution_raw.get("influences") or attribution_raw.get("chunks") or []
    for item in influences:
        cid = item.get("chunk_id")
        score = item.get("score")
        if cid is not None and score is not None:
            try:
                influence_scores[cid] = float(score)
            except (TypeError, ValueError):
                influence_scores[cid] = 0.0

    # Fallback: if AttributionOutput was already a {chunk_id: score} map
    if not influence_scores and isinstance(attribution_raw, dict):
        for cid, val in attribution_raw.items():
            try:
                influence_scores[cid] = float(val)
            except (TypeError, ValueError):
                continue

    attribution_result = {
        "influence_scores": influence_scores,
        "raw": attribution_raw,
    }
    trace.log_attribution_step(attribution_result)

    # 5) selection metadata (formerly 'boost')
    trace.log_boost(
        boost_chunks=user_selected_chunk_ids or [],
        reason="single run",
    )

    # 6) save JSON
    trace_path = trace.save()

    # this is what will be sent back to the frontend (later via /run)
    return {
        "run_id": trace.run_id,
        "trace_file": trace_path,          # optional, for debugging
        "answer": answer_result,           # { final_answer, used_chunks, full_prompt }
        "context": ctx_dict,               # ContextSelectionResult as dict
        "attribution": attribution_result, # { influence_scores, raw }
    }
