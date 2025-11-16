from __future__ import annotations

from typing import Any, Dict, List, Optional

from langsmith import traceable
from backend.core.models import ContextSelectionResult
from backend.agents.context_manager import ContextManager

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
    # repo_state not used by current ContextManager, but kept for future flexibility
    repo_state: Dict[str, Any] | None = None,
    user_selected_chunk_ids: Optional[List[str]] = None,
    top_k: int = 8,
) -> ContextSelectionResult:
    """
    LangSmith-traced wrapper around the ContextManager.

    Underlying object:
        backend.agents.context_manager.ContextManager

    Returns:
        ContextSelectionResult (Pydantic model)
    """
    ctx_manager = ContextManager(
        root_dir="backend/data/codebase",
        model_name="claude-3-5-sonnet",
    )

    return ctx_manager.select(
        query=user_query,
        conversation=conversation_history,
        top_k=top_k,
        user_selected_chunk_ids=user_selected_chunk_ids,
    )
