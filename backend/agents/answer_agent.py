from __future__ import annotations

from typing import List

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import ContextSelectionResult, ScoredChunk
from backend.agents.react_agent_setup import get_chat_model


class AnswerAgent:
    """
    ReAct agent that produces the final answer using the selected context chunks.

    Input:
        ContextSelectionResult

    Output:
        dict with:
            final_output: str
            full_prompt: str
    """

    def __init__(self, tools: List | None = None) -> None:
        llm = get_chat_model()
        self.agent = create_react_agent(llm, tools or [])

    @staticmethod
    def _format_chunks(chunks: list[ScoredChunk]) -> str:
        """
        Turn selected chunks into a readable block of text for the prompt.
        """
        parts: list[str] = []
        for scored in chunks:
            c = scored.chunk
            header = f"[{c.chunk_id}] from {c.file_path} (lines {c.start_line}-{c.end_line})"
            body = c.text.strip()
            parts.append(f"{header}\n{body}\n")
        return "\n".join(parts)

    def run(self, context: ContextSelectionResult) -> dict:
        """
        Run the ReAct agent on the given context selection.
        """
        chunk_text = self._format_chunks(context.selected_chunks)

        system_prompt = (
            "You are a careful debugging assistant. "
            "Use only the provided context chunks when reasoning about the answer."
        )

        user_prompt = (
            f"User request:\n{context.query}\n\n"
            "Relevant context chunks:\n"
            f"{chunk_text}\n"
            "Instructions:\n"
            "1. Focus on code debugging and explanations.\n"
            "2. If information is missing, state that clearly.\n"
            "3. Do not reveal chain of thought.\n"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        result = self.agent.invoke({"messages": messages})
        final_msg = result["messages"][-1]

        return {
            "final_output": final_msg.content,
            "full_prompt": system_prompt + "\n\n" + user_prompt,
        }
