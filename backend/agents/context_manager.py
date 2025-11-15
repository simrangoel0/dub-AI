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
