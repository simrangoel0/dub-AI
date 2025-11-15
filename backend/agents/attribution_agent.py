from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import ContextSelectionResult
from backend.agents.react_agent_setup import get_chat_model


class AttributionAgent:
    """
    Computes an influence score for each selected chunk.

    Input:
        context: ContextSelectionResult
        answer_output: dict from AnswerAgent.run

    Output:
        dict:
            {
                "influence_scores": {chunk_id: float in [0, 1], ...}
            }
    """

    def __init__(self) -> None:
        self.llm = get_chat_model()

    @staticmethod
    def _build_prompt(
        context: ContextSelectionResult,
        answer_output: dict,
    ) -> str:
        final_answer = answer_output["final_output"]

        chunk_blocks: list[str] = []
        chunk_ids: list[str] = []

        for scored in context.selected_chunks:
            c = scored.chunk
            chunk_ids.append(c.chunk_id)
            chunk_blocks.append(
                f"[{c.chunk_id}] from {c.file_path} (lines {c.start_line}-{c.end_line})\n"
                f"{c.text.strip()}\n"
            )

        id_list = ", ".join(f'"{cid}"' for cid in chunk_ids)

        schema_hint = (
            "{\n"
            '  "influence_scores": {\n'
            f"    {id_list}: number between 0 and 1\n"
            "  }\n"
            "}"
        )

        return (
            "You are an attribution judge.\n"
            "Your task is to estimate how much each context chunk influenced "
            "the final answer. The score must be between 0 and 1.\n\n"
            f"Final answer:\n{final_answer}\n\n"
            "Context chunks:\n"
            f"{''.join(chunk_blocks)}\n"
            "Return only valid JSON with this shape:\n"
            f"{schema_hint}\n"
        )

    def run(self, context: ContextSelectionResult, answer_output: dict) -> dict:
        prompt = self._build_prompt(context, answer_output)

        messages = [
            SystemMessage(content="Return only valid JSON. No explanation."),
            HumanMessage(content=prompt),
        ]

        raw = self.llm.invoke(messages).content.strip()

        try:
            parsed = json.loads(raw)
            scores = parsed.get("influence_scores", {})
        except Exception:
            # Safe fallback: zero for all chunks
            scores = {sc.chunk.chunk_id: 0.0 for sc in context.selected_chunks}

        # Ensure all chunks have a float score
        influence_scores = {
            sc.chunk.chunk_id: float(scores.get(sc.chunk.chunk_id, 0.0))
            for sc in context.selected_chunks
        }

        return {"influence_scores": influence_scores}
