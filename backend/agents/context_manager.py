from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Iterable

from backend.core.models import Chunk, ScoredChunk, ContextSelectionResult
from backend.core.embeddings import (
    EmbeddingBackend,
    DummyEmbeddingBackend,
    cosine_similarity,
)


class CodebaseIndex:
    """
    Handles:
    - Loading code files from a root directory
    - Chunking them into manageable pieces
    - Computing and storing embeddings
    - Performing similarity-based retrieval for a query
    """

    def __init__(
        self,
        root_dir: str,
        embedding_backend: Optional[EmbeddingBackend] = None,
        file_globs: Optional[List[str]] = None,
        chunk_max_lines: int = 25,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.embedding_backend = embedding_backend or DummyEmbeddingBackend()
        self.file_globs = file_globs or ["**/*.py"]
        self.chunk_max_lines = chunk_max_lines

        self.chunks: List[Chunk] = []
        self.chunk_embeddings: List[List[float]] = []

    # ---------- Public API ----------

    def build_index(self) -> None:
        """
        Load files from root_dir, create chunks, and compute embeddings.
        Call this once at startup (or when code changes).
        """
        self.chunks = self._load_and_chunk_files()
        texts = [c.text for c in self.chunks]
        self.chunk_embeddings = self.embedding_backend.embed_many(texts)

    def retrieve_raw(
        self,
        query: str,
        top_k: int = 8,
    ) -> List[ScoredChunk]:
        """
        Return top_k chunks ranked purely by cosine similarity.
        This is the 'A1.3 — Initial retrieval' step.
        """
        if not self.chunks or not self.chunk_embeddings:
            raise RuntimeError("Index is empty. Did you call build_index()?")

        query_emb = self.embedding_backend.embed_text(query)
        scores = [
            cosine_similarity(query_emb, emb) for emb in self.chunk_embeddings
        ]

        # Sort indices by score descending
        ranked = sorted(
            enumerate(scores),
            key=lambda t: t[1],
            reverse=True,
        )[:top_k]

        scored_chunks: List[ScoredChunk] = []
        for idx, score in ranked:
            scored_chunks.append(
                ScoredChunk(
                    chunk=self.chunks[idx],
                    similarity_score=score,
                    relevance_score=score,  # will adjust with policy
                    rationale="Initial retrieval by semantic similarity.",
                )
            )
        return scored_chunks

    # ---------- Internal helpers ----------

    def _load_and_chunk_files(self) -> List[Chunk]:
        """
        Iterate over all files matching file_globs and split them
        into chunks of at most `chunk_max_lines` lines.
        """
        chunks: List[Chunk] = []
        for pattern in self.file_globs:
            for path in self.root_dir.glob(pattern):
                if path.is_file():
                    file_chunks = self._chunk_file(path)
                    chunks.extend(file_chunks)
        return chunks

    def _chunk_file(self, path: Path) -> List[Chunk]:
        """
        Simple line-based chunking:
        - Split file into lines
        - Group by `chunk_max_lines`
        - Create Chunk objects with line ranges
        """
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        chunks: List[Chunk] = []

        start = 0
        chunk_id_counter = 0

        while start < len(lines):
            end = min(start + self.chunk_max_lines, len(lines))
            chunk_text = "\n".join(lines[start:end])

            chunk = Chunk(
                chunk_id=f"{path.name}_{chunk_id_counter}",
                file_path=str(path),
                start_line=start + 1,  # 1-based indexing
                end_line=end,
                text=chunk_text,
            )
            chunks.append(chunk)

            chunk_id_counter += 1
            start = end

        return chunks

def apply_relevance_policy(
    scored_chunks: List[ScoredChunk],
    boost_chunks: Optional[List[str]] = None,
    disable_chunks: Optional[List[str]] = None,
) -> List[ScoredChunk]:
    """
    A1.4 — Relevance scoring policy (simple version for now):
    - If chunk_id in disable_chunks -> relevance_score = 0
    - If chunk_id in boost_chunks -> relevance_score *= 1.5
    - Boost chunks containing WARNING / SECURITY in text
    You can make this smarter later or add an LLM mini-judge.
    """
    boost_chunks = set(boost_chunks or [])
    disable_chunks = set(disable_chunks or [])

    updated: List[ScoredChunk] = []

    for sc in scored_chunks:
        c = sc.chunk
        score = sc.similarity_score
        rationale_parts: List[str] = [sc.rationale]

        # Disabled chunks are effectively dropped
        if c.chunk_id in disable_chunks:
            new_score = 0.0
            rationale_parts.append("Explicitly disabled by user.")
        else:
            new_score = score

            # User boost
            if c.chunk_id in boost_chunks:
                new_score *= 1.5
                rationale_parts.append("Boosted due to user 'boost' selection.")

            # Heuristic: boost warnings / security comments
            upper_text = c.text.upper()
            if "WARNING" in upper_text or "SECURITY" in upper_text:
                new_score *= 1.2
                rationale_parts.append("Boosted due to WARNING/SECURITY marker.")

        updated.append(
            ScoredChunk(
                chunk=c,
                similarity_score=score,
                relevance_score=new_score,
                rationale=" ".join(rationale_parts),
            )
        )

    # Re-sort by relevance_score
    updated.sort(key=lambda sc: sc.relevance_score, reverse=True)
    return updated


def select_context(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    boost_chunks: Optional[List[str]] = None,
    disable_chunks: Optional[List[str]] = None,
) -> ContextSelectionResult:
    """
    High-level API Dev B and Dev A2 will use.

    Steps:
    1. Retrieve chunks by raw semantic similarity
    2. Apply relevance policy (boost/penalise)
    3. Split into selected vs dropped chunks
    """
    # Step 1 — initial retrieval
    raw_scored = index.retrieve_raw(query=query, top_k=top_k * 2)
    # (We retrieve more than top_k, then let policy refine.)

    # Step 2 — policy
    scored_with_policy = apply_relevance_policy(
        raw_scored,
        boost_chunks=boost_chunks,
        disable_chunks=disable_chunks,
    )

    # Step 3 — select top_k by relevance_score
    selected = scored_with_policy[:top_k]
    dropped = scored_with_policy[top_k:]

    result = ContextSelectionResult(
        query=query,
        top_k=top_k,
        selected_chunks=selected,
        dropped_chunks=dropped,
        meta={
            "boost_chunks": boost_chunks or [],
            "disable_chunks": disable_chunks or [],
        },
    )
    return result