from observability.trace_logger import TraceLogger
from observability.traced_agents import (
    traced_context_manager,
    traced_answer_agent,
    traced_attribution_agent,
)

def run_pipeline(payload: dict):
    user_query = payload["query"]
    conversation_history = payload.get("history", [])
    repo_state = payload.get("repo_state", {})
    boost_chunks = payload.get("boost_chunks") or []

    # 1) init logger
    trace = TraceLogger()
    trace.set_input(
        user_query=user_query,
        conversation_history=conversation_history,
        top_k=None,  # filled in by context manager
        initial_boost_chunks=boost_chunks,
        repo_state_summary={"files": list(repo_state.keys())} if isinstance(repo_state, dict) else None,
    )

    # 2) context manager (Dev A1)
    ctx_result = traced_context_manager(
        user_query=user_query,
        conversation_history=conversation_history,
        repo_state=repo_state,
        boost_chunks=boost_chunks,
    )
    trace.log_context_selection(ctx_result)

    selected_chunks = ctx_result["selected_chunks"]

    # 3) answer agent (Dev A2)
    answer_result = traced_answer_agent(
        user_query=user_query,
        selected_chunks=selected_chunks,
    )
    trace.log_answer_step(answer_result)
    final_code = answer_result.get("final_code", "")

    # 4) attribution agent (Dev A2)
    attribution_result = traced_attribution_agent(
        final_code=final_code,
        selected_chunks=selected_chunks,
    )
    trace.log_attribution_step(attribution_result)

    # 5) boost info (even if empty)
    trace.log_boost(
        boost_chunks=boost_chunks,
        reason="single run",  # or "user boosted chunks in UI"
    )

    # 6) save JSON
    trace_path = trace.save()

    # this is what will be sent back to the frontend
    return {
        "run_id": trace.run_id,
        "trace_file": trace_path,      # optional, for debugging
        "answer": answer_result,
        "context": ctx_result,
        "attribution": attribution_result,
    }
