from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Represents a chunk of context (either CODE or CONVERSATION).

    For code chunks:
        - source = "code"
        - file_path = actual .py file path
        - start_line/end_line = line numbers in file

    For conversation chunks:
        - source = "chat"
        - file_path = "__conversation__"
        - start_line/end_line = message index (1-based)
    """
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    text: str

    # NEW: context source + metadata
    source: str = "code"              # "code" or "chat"
    meta: Dict[str, Any] = {}         # e.g. {"role": "user"} for chat


class ScoredChunk(BaseModel):
    """
    A chunk plus scoring information.

    similarity_score:
        - For this version, not used in ranking (LLM does ranking),
          but kept for compatibility / future extensions.

    relevance_score:
        - Final score that decides selection (here: 1.0 if selected, 0.0 otherwise).

    rationale:
        - Human-readable explanation of why this chunk was selected or dropped.
        - For agent-selected chunks: LLM's explanation.
        - For user-selected chunks: "User-selected chunk."
    """
    chunk: Chunk
    similarity_score: float
    relevance_score: float
    rationale: str


class ContextSelectionResult(BaseModel):
    """
    Final output of the Context Manager Agent for a single query.

    query:
        - The user's current question.

    top_k:
        - Target number of selected chunks (MAX; LLM may choose fewer).

    selected_chunks:
        - Chunks (code + chat) that will be passed to the Answer Agent.

    dropped_chunks:
        - Chunks that were considered as candidates but not selected.

    meta:
        - Extra metadata for observability:
            - selection_mode: "user" or "agent"
            - user_selected_chunk_ids
            - selected_chunk_ids
            - max_code_candidates, etc.
    """
    query: str
    top_k: int
    selected_chunks: List[ScoredChunk]
    dropped_chunks: List[ScoredChunk]
    meta: Dict[str, Any] = {}