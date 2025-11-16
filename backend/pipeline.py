from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from backend.core.models import ContextSelectionResult
from backend.agents.answer_agent import AnswerAgent
from backend.agents.attribution_agent import AttributionAgent
from backend.observability.trace_logger import TraceLogger
from backend.observability.traced_agents import traced_context_manager
from backend.db import init_db, persist_run, persist_chat_messages

# Init once
init_db()
answer_agent = AnswerAgent(model_name="claude-3-5-sonnet")
attribution_agent = AttributionAgent(model_name="claude-3-5-sonnet")


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs:
      1. Context selection
      2. Answer generation
      3. Summary generation
      4. Attribution
      5. Persistence + tracing
    """

    user_query = payload["query"]
    conversation_history = payload.get("history", []) or []
    repo_state = payload.get("repo_state", {}) or {}
    user_selected_chunk_ids = payload.get("selected_chunk_ids")
    top_k = payload.get("top_k", 8)
    conversation_id = payload.get("conversation_id", "default")

    # Generate UI message id
    message_id = str(uuid.uuid4())

    # 1. Trace input
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

    persist_chat_messages(conversation_id, conversation_history)

    # 2. Context Selection
    ctx_result: ContextSelectionResult = traced_context_manager(
        user_query=user_query,
        conversation_history=conversation_history,
        repo_state=repo_state,
        user_selected_chunk_ids=user_selected_chunk_ids,
        top_k=top_k,
    )

    ctx_dict = ctx_result.model_dump()
    trace.log_context_selection(ctx_dict)

    # 3. Answer Agent
    answer_result = answer_agent.run(ctx_result)
    answer_result["response_context"]["messageId"] = message_id
    answer_result["response_context"]["runId"] = trace.run_id
    trace.log_answer_step(answer_result)

    # 3.5 Generate Short Summary for UI + DB
    from holistic_ai_bedrock import get_chat_model
    llm = get_chat_model("claude-3-5-sonnet")

    summary_prompt = (
        "Summarize the following debugging answer in 6–12 words. "
        "Be concise and descriptive.\n\n"
        f"{answer_result['final_answer']}"
    )

    summary_label = llm.invoke(summary_prompt).content.strip()
    answer_result["summary"] = summary_label

    # 4. Attribution Agent
    attribution_raw = attribution_agent.run(
        selection=ctx_result,
        answer=answer_result,
        run_id=trace.run_id,
        message_id=message_id,
    )

    influence_scores: Dict[str, float] = {}
    for entry in attribution_raw.get("raw", {}).get("chunks", []):
        cid = entry.get("chunk_id")
        score = entry.get("score")
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

    # 5. Boost log
    trace.log_boost(
        boost_chunks=user_selected_chunk_ids or [],
        reason="single run",
    )

    # 6. Save trace JSON file
    trace_file = trace.save()

    # 7. Persist the run (WITH summary label)
    persist_run(
        run_id=trace.run_id,
        message_id=message_id,
        conversation_id=conversation_id,
        user_query=user_query,
        run_label=summary_label,       # <--- use summary as label
        ctx=ctx_result,
        answer_result=answer_result,
        attribution_result=attribution_result,
        trace_file=trace_file,
    )

    # 8. Response to UI
    return {
        "run_id": trace.run_id,
        "message_id": message_id,
        "trace_file": trace_file,
        "answer": answer_result,
        "summary": summary_label,      # <-- front end reads this
        "context": ctx_dict,
        "attribution": attribution_result,
    }
