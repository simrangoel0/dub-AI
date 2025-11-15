from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Represents a chunk of code (or context) from a file.
    """
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    text: str


class ScoredChunk(BaseModel):
    """
    A chunk plus scoring information.
    - similarity_score: raw cosine similarity from embeddings
    - relevance_score: similarity after boosting/penalties
    - rationale: short explanation of why this chunk was kept/dropped
    """
    chunk: Chunk
    similarity_score: float
    relevance_score: float
    rationale: str


class ContextSelectionResult(BaseModel):
    """
    Final output of your context manager for a single query.
    This is what Dev B (API) and Dev A2 (answer agent) will consume.
    """
    query: str
    top_k: int
    selected_chunks: List[ScoredChunk]
    dropped_chunks: List[ScoredChunk]
    meta: Dict[str, Any] = {}
