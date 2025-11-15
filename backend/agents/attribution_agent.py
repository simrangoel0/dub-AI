from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.models import ContextSelectionResult, AttributionOutput
from holistic_ai_bedrock import get_chat_model


class AttributionAgent:
    """
    Enhanced LLM-based attribution engine.

    Improvements:
    - Incorporates similarity scores, relevance scores and rationales.
    - Encourages better evidence extraction from final answer.
    - Produces high quality JSON validated through AttributionOutput.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.base_llm = get_chat_model(model_name)
        self.llm = self.base_llm.with_structured_output(AttributionOutput)

    def _build_prompt(self, selection: ContextSelectionResult, answer: dict) -> str:
        final_answer = answer["final_answer"]

        header = (
            "You are an influence attribution engine. "
            "Your job is to determine how much each context chunk affected the final answer.\n\n"
            "For each chunk, you must provide:\n"
            "1. score between 0 and 1\n"
            "2. 1 to 3 short evidence spans quoted from the final answer\n"
            "3. one sentence explanation referencing the chunk id\n\n"
            "Consider the following signals:\n"
            "- similarity_score\n"
            "- relevance_score\n"
            "- chunk selection rationale\n"
            "- overlap between chunk text and final answer\n\n"
        )

        block = "### Final Answer\n" + final_answer + "\n\n### Chunks\n"

        for sc in selection.selected_chunks:
            c = sc.chunk
            block += (
                f"[{c.chunk_id}] from {c.file_path} (lines {c.start_line}-{c.end_line})\n"
                f"Similarity score: {sc.similarity_score}\n"
                f"Relevance score: {sc.relevance_score}\n"
                f"Selection rationale: {sc.rationale}\n"
                f"Text:\n{c.text.strip()}\n\n"
            )

        return header + block

    def run(self, selection: ContextSelectionResult, answer: dict):
        prompt = self._build_prompt(selection, answer)
        result: AttributionOutput = self.llm.invoke(prompt)
        return result.model_dump()
