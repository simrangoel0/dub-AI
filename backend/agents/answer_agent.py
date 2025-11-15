from __future__ import annotations
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.models import ContextSelectionResult, ScoredChunk
from holistic_ai_bedrock import get_chat_model


class AnswerAgent:
    """
    Debug-focused answer generator.

    Enhancements:
    - Uses rationales, similarity_score and relevance_score for better grounding.
    - Produces a clearer analysis section tied directly to chunk ids.
    - More defensive about incomplete context.
    - Instructs the LLM to treat code separately from plain text.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.llm = get_chat_model(model_name)

    @staticmethod
    def _format_chunks(chunks: List[ScoredChunk]) -> str:
        blocks = []
        for sc in chunks:
            c = sc.chunk
            blocks.append(
                f"[{c.chunk_id}] from {c.file_path} (lines {c.start_line}-{c.end_line})\n"
                f"Similarity score: {sc.similarity_score:.3f}\n"
                f"Relevance score: {sc.relevance_score:.3f}\n"
                f"Selection rationale: {sc.rationale}\n"
                f"Chunk text:\n{c.text.strip()}\n"
            )
        return "\n".join(blocks)

    def run(self, ctx: ContextSelectionResult) -> Dict:
        """
        Returns:
            dict with final_answer, used_chunks, full_prompt, response_context
        """
        chunk_block = self._format_chunks(ctx.selected_chunks)
        used = [sc.chunk.chunk_id for sc in ctx.selected_chunks]

        system = (
            "You are a debugging oriented assistant. "
            "Your job is to analyse code, identify issues, explain behaviour, "
            "and propose specific fixes based only on the provided chunks.\n\n"
            "Important rules:\n"
            "1. Never use information not present in the chunks.\n"
            "2. If the code snippet looks partial, missing lines, or incomplete, mention this.\n"
            "3. Treat content as code unless clearly plain text.\n"
            "4. Reference chunk ids whenever describing evidence.\n"
            "5. Provide a short high level analysis (not chain of thought).\n"
            "6. Then provide the final debugging answer.\n"
        )

        user = (
            f"User query:\n{ctx.query}\n\n"
            "Relevant context chunks:\n"
            f"{chunk_block}\n\n"
            "Write output with two sections:\n"
            "Analysis: short, high level, grounded in chunk ids.\n"
            "Final Answer: solve the query.\n"
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]

        response_text = self.llm.invoke(messages).content.strip()

        # Pipeline will fill messageId and runId later
        response_context = {
            "messageId": None,
            "runId": None,
            "chunks": [
                {
                    "id": sc.chunk.chunk_id,
                    "selected": True,
                    "influenceScore": 0.0,
                    "rationale": sc.rationale
                }
                for sc in ctx.selected_chunks
            ],
            "droppedChunks": [
                {
                    "id": sc.chunk.chunk_id,
                    "selected": False,
                    "influenceScore": 0.0,
                    "rationale": sc.rationale
                }
                for sc in ctx.dropped_chunks
            ]
        }

        return {
            "final_answer": response_text,
            "used_chunks": used,
            "full_prompt": system + "\n\n" + user,
            "response_context": response_context
        }
