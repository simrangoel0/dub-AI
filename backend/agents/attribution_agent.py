from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import (
    ContextSelectionResult,
    AttributionOutput
)
from holistic_ai_bedrock import get_chat_model


class AttributionAgent:
    """
    LLM-based chunk attribution + evidence extraction agent.
    Produces structured JSON validated through AttributionOutput.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.base_llm = get_chat_model(model_name)
        self.llm = self.base_llm.with_structured_output(AttributionOutput)

    def _build_prompt(self, selection: ContextSelectionResult, answer: dict) -> str:
        final_answer = answer["final_answer"]

        text = (
            "You are an attribution engine. "
            "Your role is to score how much each chunk influenced the final answer.\n\n"
            "For each chunk:\n"
            "- Give score between 0 and 1\n"
            "- Provide 1-3 short evidence snippets from the final answer\n"
            "- Provide a one-sentence explanation\n\n"
            "### Final Answer\n"
            f"{final_answer}\n\n"
            "### Chunks\n"
        )

        for sc in selection.selected_chunks:
            c = sc.chunk
            text += f"[{c.chunk_id}] {c.text}\n\n"

        return text

    def run(self, selection: ContextSelectionResult, answer: dict):
        prompt = self._build_prompt(selection, answer)
        result: AttributionOutput = self.llm.invoke(prompt)
        return result.model_dump()
