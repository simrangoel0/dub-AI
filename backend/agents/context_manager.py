from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from dotenv import load_dotenv

from backend.core.models import Chunk, ScoredChunk, ContextSelectionResult
from backend.core.embeddings import (
    EmbeddingBackend,
    DummyEmbeddingBackend,
    cosine_similarity,
)

# LangGraph / LangChain imports
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from holistic_ai_bedrock import get_chat_model

load_dotenv()
# -----------------------------
# Codebase Index (code chunks)
# -----------------------------
class CodebaseIndex:
    """
    Handles:
    - Loading code files from a root directory
    - Chunking them into manageable pieces
    - Computing embeddings
    - Returning a small set of candidate code chunks for a query
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

    def build_index(self) -> None:
        """
        Load files from root_dir, create chunks, compute embeddings.
        Call once at startup or whenever the codebase changes.
        """
        self.chunks = self._load_and_chunk_files()
        texts = [c.text for c in self.chunks]
        self.chunk_embeddings = self.embedding_backend.embed_many(texts)

    def _load_and_chunk_files(self) -> List[Chunk]:
        """
        Iterate over all files matching file_globs and split them
        into chunks of at most chunk_max_lines lines.
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
        - Group by chunk_max_lines
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
                start_line=start + 1,
                end_line=end,
                text=chunk_text,
                source="code",
                meta={},
            )
            chunks.append(chunk)

            chunk_id_counter += 1
            start = end

        return chunks

    def get_candidate_chunks(
        self,
        query: str,
        max_candidates: int = 12,
    ) -> List[Chunk]:
        """
        Use embeddings only to fetch a manageable set of candidate chunks.
        The LLM agent will do the final ranking and selection.

        Returns up to max_candidates Chunk objects.
        """
        if not self.chunks or not self.chunk_embeddings:
            raise RuntimeError("Index is empty. Did you call build_index()?")

        query_emb = self.embedding_backend.embed_text(query)
        scores = [
            cosine_similarity(query_emb, emb) for emb in self.chunk_embeddings
        ]

        ranked = sorted(
            enumerate(scores),
            key=lambda t: t[1],
            reverse=True,
        )[:max_candidates]

        return [self.chunks[idx] for idx, _ in ranked]


# ---------------------------------------
# Conversation → Chunk conversion
# ---------------------------------------
def build_conversation_chunks(
    conversation: List[Dict[str, Any]],
    max_chars_per_chunk: int = 600,
) -> List[Chunk]:
    """
    Turn the conversation history into Chunk objects.

    Each message becomes its own chunk (possibly truncated):
        chunk_id = "chat_{index}"
        file_path = "__conversation__"
        source = "chat"
        meta["role"] = "user" / "assistant" / etc.
    """
    chunks: List[Chunk] = []

    for idx, msg in enumerate(conversation):
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        preview = content
        if len(preview) > max_chars_per_chunk:
            preview = preview[:max_chars_per_chunk] + "..."

        chunk = Chunk(
            chunk_id=f"chat_{idx}",
            file_path="__conversation__",
            start_line=idx + 1,
            end_line=idx + 1,
            text=f"{role.upper()}: {preview}",
            source="chat",
            meta={"role": role},
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------
# LLM-based Context Selection Agent
# ---------------------------------------
class ContextSelectionAgent:
    """
    LLM-based agent that decides which chunks (code + chat) to select.

    Uses LangGraph's create_react_agent with no external tools for now.
    We feed it the query and candidate chunks and ask it to return JSON
    specifying which chunk_ids to keep and why.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet") -> None:
        llm = get_chat_model(model_name)
        self.agent = create_react_agent(llm, tools=[])

    def select_chunks_with_llm(
        self,
        query: str,
        candidates: List[Chunk],
        top_k: int,
    ) -> Dict[str, str]:
        """
        Ask the LLM agent to choose up to top_k chunk_ids from candidates.

        Returns:
            dict: mapping chunk_id -> rationale string
        """
        candidate_descriptions: List[str] = []
        for i, c in enumerate(candidates):
            preview = c.text.strip()
            if len(preview) > 400:
                preview = preview[:400] + "..."

            kind = c.source  # "code" or "chat"
            file_label = Path(c.file_path).name if c.source == "code" else "__conversation__"

            candidate_descriptions.append(
                f"{i}. chunk_id={c.chunk_id}\n"
                f"   source={kind}\n"
                f"   file={file_label}\n"
                f"   lines={c.start_line}-{c.end_line}\n"
                f"   preview:\n{preview}\n"
            )

        candidates_block = "\n\n".join(candidate_descriptions)

        system_prompt = (
            "You are a context selection agent for a code debugging assistant.\n"
            "Given a user query, previous conversation, and candidate code chunks, your job is to:\n"
            f"1. Decide which chunks (code AND chat) are most relevant to answering the query.\n"
            f"2. Select AT MOST {top_k} chunks.\n"
            "3. Explain briefly why each selected chunk is relevant.\n\n"
            "Each candidate has:\n"
            "- source=code → Python code from the repo\n"
            "- source=chat → a message from the conversation history\n\n"
            "You must respond ONLY in valid JSON with the following shape:\n"
            '{\n'
            '  \"selected_chunk_ids\": [\"chunk_id_1\", \"chunk_id_2\", ...],\n'
            '  \"rationales\": {\n'
            '    \"chunk_id_1\": \"why this chunk is relevant\",\n'
            '    \"chunk_id_2\": \"why this chunk is relevant\"\n'
            '  }\n'
            '}\n'
            "Do not include any other keys or text outside this JSON.\n"
        )

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Here are {len(candidates)} candidate context chunks:\n\n"
            f"{candidates_block}\n\n"
            f"Now choose up to {top_k} chunks that are most relevant to answering the query."
        )

        result = self.agent.invoke({
            "messages": [
                HumanMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        })

        content = result["messages"][-1].content

        try:
            data = json.loads(content)
        except Exception:
            # Fallback: if JSON parsing fails, pick first min(top_k, len(candidates)) candidates
            fallback_ids = [c.chunk_id for c in candidates[:top_k]]
            return {cid: "Fallback selection due to JSON parse error." for cid in fallback_ids}

        selected_ids = data.get("selected_chunk_ids", [])
        rationales = data.get("rationales", {})

        return {
            cid: rationales.get(cid, "Selected by LLM agent as relevant to the query.")
            for cid in selected_ids
        }


# ---------------------------------------
# Context Manager (public API)
# ---------------------------------------
class ContextManager:
    """
    High-level Context Manager Agent.

    Behaviour:
    - If user_selected_chunk_ids is provided and non-empty:
        -> Use those chunks directly (no LLM ranking).
    - Else:
        -> Use LLM agent (ContextSelectionAgent) to select chunks
           from a set of candidates (code + chat).
    """

    def __init__(
        self,
        root_dir: str = "backend/data/codebase",
        embedding_backend: Optional[EmbeddingBackend] = None,
        file_globs: Optional[List[str]] = None,
        chunk_max_lines: int = 25,
        model_name: str = "claude-3-5-sonnet",
    ) -> None:
        self.index = CodebaseIndex(
            root_dir=root_dir,
            embedding_backend=embedding_backend or DummyEmbeddingBackend(),
            file_globs=file_globs,
            chunk_max_lines=chunk_max_lines,
        )
        self.index.build_index()

        self.agent = ContextSelectionAgent(model_name=model_name)

    def select(
        self,
        query: str,
        conversation: List[Dict[str, Any]],
        top_k: int = 8,
        user_selected_chunk_ids: Optional[List[str]] = None,
        max_code_candidates: int = 12,
    ) -> ContextSelectionResult:
        """
        Main entrypoint for the Context Manager Agent.

        Args:
            query: Current user question in this turn.
            conversation: Full conversation history as a list of
                          {"role": "user"/"assistant"/..., "content": "..."}.
            top_k: Max number of chunks the agent should select.
            user_selected_chunk_ids: If provided and non-empty, use these
                                     chunk IDs directly (no LLM ranking).
            max_code_candidates: How many code chunks to pass as candidates
                                 (conversation chunks are all passed).

        Returns:
            ContextSelectionResult with selected vs dropped chunks.
        """
        # Case 1: user manually selected chunks (no LLM ranking)
        if user_selected_chunk_ids:
            id_set = set(user_selected_chunk_ids)

            selected_chunks: List[ScoredChunk] = []
            dropped_chunks: List[ScoredChunk] = []

            # All possible chunks: code + chat
            all_chunks: List[Chunk] = []
            all_chunks.extend(self.index.chunks)  # all code chunks
            all_chunks.extend(build_conversation_chunks(conversation))

            for c in all_chunks:
                is_selected = c.chunk_id in id_set
                sc = ScoredChunk(
                    chunk=c,
                    similarity_score=0.0,
                    relevance_score=1.0 if is_selected else 0.0,
                    rationale=(
                        "User-selected chunk."
                        if is_selected
                        else "Not selected by user."
                    ),
                )
                if is_selected:
                    selected_chunks.append(sc)
                else:
                    dropped_chunks.append(sc)

            return ContextSelectionResult(
                query=query,
                top_k=len(selected_chunks),
                selected_chunks=selected_chunks,
                dropped_chunks=dropped_chunks,
                meta={
                    "selection_mode": "user",
                    "user_selected_chunk_ids": user_selected_chunk_ids,
                },
            )

        # Case 2: no user selection -> LLM agent selects from code + chat

        # 1) code candidates via index
        code_candidates = self.index.get_candidate_chunks(
            query=query,
            max_candidates=max_code_candidates,
        )

        # 2) chat candidates from full conversation
        chat_candidates = build_conversation_chunks(conversation)

        # 3) combine
        candidates: List[Chunk] = []
        candidates.extend(code_candidates)
        candidates.extend(chat_candidates)

        # 4) ask LLM agent to choose up to top_k
        chosen_map = self.agent.select_chunks_with_llm(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        chosen_ids = set(chosen_map.keys())

        selected_chunks: List[ScoredChunk] = []
        dropped_chunks: List[ScoredChunk] = []

        for c in candidates:
            if c.chunk_id in chosen_ids:
                selected_chunks.append(
                    ScoredChunk(
                        chunk=c,
                        similarity_score=0.0,
                        relevance_score=1.0,
                        rationale=chosen_map[c.chunk_id],
                    )
                )
            else:
                dropped_chunks.append(
                    ScoredChunk(
                        chunk=c,
                        similarity_score=0.0,
                        relevance_score=0.0,
                        rationale="Candidate chunk not selected by LLM agent.",
                    )
                )

        return ContextSelectionResult(
            query=query,
            top_k=top_k,
            selected_chunks=selected_chunks,
            dropped_chunks=dropped_chunks,
            meta={
                "selection_mode": "agent",
                "max_code_candidates": max_code_candidates,
                "selected_chunk_ids": list(chosen_ids),
            },
        )