# backend/observability/traced_agents.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langsmith import traceable  # the decorator that logs traces :contentReference[oaicite:6]{index=6}


# TYPE HINTS (you can refine later with Pydantic models)
ContextManagerResult = Dict[str, Any]
AnswerAgentResult = Dict[str, Any]
AttributionResult = Dict[str, Any]


@traceable(
    run_type="chain",
    name="context_manager",
    metadata={"role": "A1", "stage": "retrieval+scoring"},
    tags=["glassbox", "context-manager"],
)
def traced_context_manager(
    *,
    user_query: str,
    conversation_history: List[Dict[str, Any]],
    repo_state: Dict[str, Any],
    boost_chunks: Optional[List[str]] = None,
) -> ContextManagerResult:
    """
    Wrapper around Dev A1's context manager.

    Expected underlying function (to be implemented by Role 1):
        agents.context_manager.run_context_manager(...)
    Returns something like:
        {
          "selected_chunks": [...],
          "dropped_chunks": [...],
          "scores": {...},
          "rationales": {...},
          "maybe_summaries": {...}
        }
    """
    from agents.context_manager import run_context_manager

    return run_context_manager(
        user_query=user_query,
        conversation_history=conversation_history,
        repo_state=repo_state,
        boost_chunks=boost_chunks,
    )


@traceable(
    run_type="llm",
    name="answer_agent",
    metadata={"role": "A2", "stage": "answer"},
    tags=["glassbox", "answer-agent"],
)
def traced_answer_agent(
    *,
    user_query: str,
    selected_chunks: List[Dict[str, Any]],
    system_instructions: Optional[str] = None,
) -> AnswerAgentResult:
    """
    Wrapper around Dev A2's answer agent.

    Underlying function (Role 1 / Dev A2):
        agents.answer_agent.run_answer_agent(...)
    Expected to return:
        {
          "final_code": "...",
          "answer_text": "...",
          "raw_llm_output": {...}
        }
    """
    from agents.answer_agent import run_answer_agent

    return run_answer_agent(
        user_query=user_query,
        selected_chunks=selected_chunks,
        system_instructions=system_instructions,
    )


@traceable(
    run_type="chain",
    name="attribution_agent",
    metadata={"role": "A2", "stage": "attribution"},
    tags=["glassbox", "attribution"],
)
def traced_attribution_agent(
    *,
    final_code: str,
    selected_chunks: List[Dict[str, Any]],
) -> AttributionResult:
    """
    Wrapper around Dev A2's attribution agent.

    Underlying function (Role 1 / Dev A2):
        agents.attribution_agent.run_attribution_agent(...)
    Expected to return:
        {
          "influence_map": {"chunk_id": score, ...},
          "explanations": {...}
        }
    """
    from agents.attribution_agent import run_attribution_agent

    return run_attribution_agent(
        final_code=final_code,
        selected_chunks=selected_chunks,
    )
