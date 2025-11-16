from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from backend.core.models import ContextSelectionResult
from backend.agents.answer_agent import AnswerAgent
from backend.agents.attribution_agent import AttributionAgent
from backend.observability.trace_logger import TraceLogger
from backend.observability.traced_agents import traced_context_manager
from backend.db import init_db, persist_run, persist_chat_messages

init_db()
answer_agent = AnswerAgent(model_name="claude-3-5-sonnet")
attribution_agent = AttributionAgent(model_name="claude-3-5-sonnet")


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates context selection, answer generation, attribution, and trace logging.
    """

    user_query = payload["query"]
    conversation_history = payload.get("history", []) or []
    repo_state = payload.get("repo_state", {}) or {}
    user_selected_chunk_ids = payload.get("selected_chunk_ids")
    top_k = payload.get("top_k", 8)

    # Generate message id (UUID4 is fine)
    message_id = str(uuid.uuid4())

    # Step 1: trace initial input
    trace = TraceLogger()
    trace.set_input(
        user_query=user_query,
        conversation_history=conversation_history,
        top_k=None,
        initial_boost_chunks=user_selected_chunk_ids or [],
        repo_state_summary={"files": list(repo_state.keys())}
        if isinstance(repo_state, dict)
        else None,
    )

    conversation_id = payload.get("conversation_id", "default")
    persist_chat_messages(conversation_id, conversation_history)

    # Step 2: context selection
    ctx_result: ContextSelectionResult = traced_context_manager(
        user_query=user_query,
        conversation_history=conversation_history,
        repo_state=repo_state,
        user_selected_chunk_ids=user_selected_chunk_ids,
        top_k=top_k,
    )

    ctx_dict = ctx_result.model_dump()
    trace.log_context_selection(ctx_dict)

    # Step 3: answer agent
    answer_result = answer_agent.run(ctx_result)

    # Insert UI identifiers
    answer_result["response_context"]["messageId"] = message_id
    answer_result["response_context"]["runId"] = trace.run_id

    trace.log_answer_step(answer_result)

    # Step 4: attribution agent
    attribution_raw = attribution_agent.run(
        selection=ctx_result,
        answer=answer_result,
        run_id=trace.run_id,
        message_id=message_id,
    )

    influence_scores = {}
    entries = attribution_raw.get("raw", {}).get("chunks", [])
    for entry in entries:
        cid = entry.get("chunk_id")
        score = entry.get("score")
        if cid is not None and score is not None:
            try:
                influence_scores[cid] = float(score)
            except Exception:
                influence_scores[cid] = 0.0

    attribution_result = {
        "influence_scores": influence_scores,
        "raw": attribution_raw,
        "response_context": attribution_raw.get("response_context"),
        "explanations": attribution_raw.get("explanations"),
    }

    trace.log_attribution_step(attribution_result)

    # Step 5: log selection metadata
    trace.log_boost(
        boost_chunks=user_selected_chunk_ids or [],
        reason="single run",
    )

    # Step 6: save full trace file
    trace_file = trace.save()

    # Step 7: persist into DB
    conversation_id = payload.get("conversation_id", "default")

    persist_run(
        run_id=trace.run_id,
        message_id=message_id,
        conversation_id=conversation_id,
        user_query=user_query,
        ctx=ctx_result,
        answer_result=answer_result,
        attribution_result=attribution_result,
        trace_file=trace_file,
    )

    # Final response to UI
    return {
        "run_id": trace.run_id,
        "message_id": message_id,
        "trace_file": trace_file,
        "answer": answer_result,
        "context": ctx_dict,
        "attribution": attribution_result,
    }
