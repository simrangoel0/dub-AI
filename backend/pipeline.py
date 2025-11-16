from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from backend.core.models import ContextSelectionResult
from backend.agents.answer_agent import AnswerAgent
from backend.agents.attribution_agent import AttributionAgent
from backend.observability.trace_logger import TraceLogger
from backend.observability.traced_agents import traced_context_manager

# Instantiate A2 agents once
answer_agent = AnswerAgent(model_name="claude-3-5-sonnet")
attribution_agent = AttributionAgent(model_name="claude-3-5-sonnet")


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates one full glass-box debugging run.

    Steps:
        1. Context selection (ContextManager)
        2. Answer generation (AnswerAgent)
        3. Attribution scoring (AttributionAgent)
        4. Observability logging (TraceLogger)
        5. UI-ready final payload

    Returns a dict consumed directly by the /run API endpoint.
    """

    # -----------------------------------------------
    # Extract request fields
    # -----------------------------------------------
    user_query: str = payload["query"]
    conversation_history: List[Dict[str, Any]] = payload.get("history", []) or []
    repo_state: Dict[str, Any] = payload.get("repo_state", {}) or {}
    user_selected_chunk_ids: Optional[List[str]] = payload.get("selected_chunk_ids")

    # Each UI message requires a separate messageId
    message_id = str(uuid.uuid4())

    # -----------------------------------------------
    # Initialise TraceLogger (record entire run)
    # -----------------------------------------------
    trace = TraceLogger()
    trace.set_input(
        user_query=user_query,
        conversation_history=conversation_history,
        top_k=None,     # filled after context selection
        initial_boost_chunks=user_selected_chunk_ids or [],
        repo_state_summary={"files": list(repo_state.keys())}
        if isinstance(repo_state, dict)
        else None,
    )

    # -----------------------------------------------
    # 1. Context Manager → ContextSelectionResult
    # -----------------------------------------------
    ctx_result: ContextSelectionResult = traced_context_manager(
        user_query=user_query,
        conversation_history=conversation_history,
        repo_state=repo_state,
        user_selected_chunk_ids=user_selected_chunk_ids,
        top_k=payload.get("top_k", 8),
    )

    ctx_dict = ctx_result.model_dump()
    trace.log_context_selection(ctx_dict)

    # -----------------------------------------------
    # 2. Answer Agent → final answer + preliminary context
    # -----------------------------------------------
    answer_result = answer_agent.run(ctx_result)

    # Inject messageId + runId into AnswerAgent's response_context
    answer_result["response_context"]["messageId"] = message_id
    answer_result["response_context"]["runId"] = trace.run_id

    trace.log_answer_step(answer_result)

    # -----------------------------------------------
    # 3. Attribution Agent → influence + evidence + reasoning
    # -----------------------------------------------
    attribution_raw = attribution_agent.run(
        selection=ctx_result,
        answer=answer_result,
        run_id=trace.run_id,
        message_id=message_id,
    )

    # Build a simple influence map for TraceLogger
    influence_scores = {
        entry["chunk_id"]: float(entry["score"])
        for entry in attribution_raw["raw"]["chunks"]
    }

    attribution_result = {
        "influence_scores": influence_scores,
        "raw": attribution_raw["raw"],
        "response_context": attribution_raw["response_context"],
        "explanations": attribution_raw["explanations"],
    }

    trace.log_attribution_step(attribution_result)

    # -----------------------------------------------
    # 4. Log metadata about user-selected chunks
    # -----------------------------------------------
    trace.log_boost(
        boost_chunks=user_selected_chunk_ids or [],
        reason="single run",
    )

    # -----------------------------------------------
    # 5. Save trace JSON (used by /trace)
    # -----------------------------------------------
    trace_path = trace.save()

    # -----------------------------------------------
    # 6. Final payload to the frontend (/run response)
    # -----------------------------------------------
    return {
        "run_id": trace.run_id,
        "message_id": message_id,
        "trace_file": trace_path,
        "answer": answer_result,
        "context": ctx_dict,
        "attribution": attribution_result,
    }
