from __future__ import annotations
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.models import ContextSelectionResult, ScoredChunk
from holistic_ai_bedrock import get_chat_model


class AnswerAgent:
    """
    Debug-focused answer generator.

    Responsibilities:
    - Convert selected chunks into a structured context block.
    - Guide the LLM to behave like a programming-debugging helper.
    - Force grounding strictly in the provided context.
    - Ask the model to explicitly state which chunks it relied on.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.llm = get_chat_model(model_name)

    @staticmethod
    def _format_chunks(chunks: List[ScoredChunk]) -> str:
        out = []
        for sc in chunks:
            c = sc.chunk
            block = (
                f"[{c.chunk_id}] from {c.file_path} "
                f"(lines {c.start_line}-{c.end_line})\n"
                f"{c.text.strip()}\n"
            )
            out.append(block)
        return "\n".join(out)

    def run(self, ctx: ContextSelectionResult) -> Dict:
        chunk_text = self._format_chunks(ctx.selected_chunks)

        system_prompt = (
            "You are a highly reliable programming-debugging assistant. "
            "Your job is to analyse code, detect bugs, explain behaviour, "
            "and propose fixes using ONLY the provided context chunks. "
            "Do not speculate about code that is not included. "
            "Be precise and cite which chunks you rely on."
        )

        user_prompt = (
            f"User query:\n{ctx.query}\n\n"
            "Relevant context chunks:\n"
            f"{chunk_text}\n\n"
            "Instructions:\n"
            "1. Carefully read all chunks.\n"
            "2. Identify which chunks contain information relevant to the query.\n"
            "3. Explain your reasoning in a short 'analysis' section that references chunk IDs.\n"
            "4. Then provide the final answer or debugging guidance.\n"
            "5. NEVER include chain-of-thought. The analysis must be short, high-level, and chunk-grounded.\n"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        resp = self.llm.invoke(messages)
        final = resp.content.strip()

        return {
            "final_answer": final,
            "used_chunks": [sc.chunk.chunk_id for sc in ctx.selected_chunks],
            "full_prompt": system_prompt + "\n\n" + user_prompt,
        }
