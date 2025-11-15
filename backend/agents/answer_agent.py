from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import ContextSelectionResult, ScoredChunk
from holistic_ai_bedrock import get_chat_model


class AnswerAgent:
    """
    Final answering agent.

    Takes the selected context from the context manager and the user query,
    and calls the Holistic AI Bedrock LLM once to produce a final answer.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        # Directly use your tutorial helper
        self.llm = get_chat_model(model_name)

    def _format_chunks(self, chunks: List[ScoredChunk]) -> str:
        """
        Convert selected chunks into a readable context string.
        """
        blocks: List[str] = []
        for scored in chunks:
            c = scored.chunk
            blocks.append(
                f"[{c.chunk_id}] {c.file_path} (lines {c.start_line}-{c.end_line})\n"
                f"{c.text.strip()}\n"
            )
        return "\n".join(blocks)

    def run(self, selection: ContextSelectionResult) -> Dict[str, Any]:
        """
        Execute the answer step.

        Args:
            selection: ContextSelectionResult produced by the context manager.

        Returns:
            A dictionary with:
                - "final_answer": str
                - "prompt_used": str (for observability)
        """
        context_text = self._format_chunks(selection.selected_chunks)

        system_prompt = (
            "You are a careful debugging assistant.\n"
            "You are given code chunks and a user question about the code.\n"
            "Use only the information in those chunks. If something is not "
            "present in the context, say that it is not available instead of guessing."
        )

        user_prompt = (
            f"User query:\n{selection.query}\n\n"
            f"Relevant code chunks:\n{context_text}\n\n"
            "Write a clear, step by step explanation that answers the query.\n"
            "If there are multiple plausible interpretations, call them out explicitly."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)

        return {
            "final_answer": response.content.strip(),
            "prompt_used": system_prompt + "\n\n" + user_prompt,
        }
