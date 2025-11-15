import json
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import ContextSelectionResult
from holistic_ai_bedrock import get_chat_model


class AttributionAgent:
    """
    Attribution and observability agent.

    Given the context selection and the final answer, this agent asks the LLM
    to score how influential each chunk was on a 0 to 1 scale.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.llm = get_chat_model(model_name)

    def run(self, selection: ContextSelectionResult, answer_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute per chunk attribution scores.

        Args:
            selection: ContextSelectionResult used for answering.
            answer_result: Output from AnswerAgent.run (must contain "final_answer").

        Returns:
            Dict mapping chunk_id -> influence score in [0, 1].
        """
        final_answer = answer_result.get("final_answer", "")

        # Build chunk display string
        chunk_blocks = []
        for scored in selection.selected_chunks:
            c = scored.chunk
            chunk_blocks.append(
                f"[{c.chunk_id}]\n"
                f"File: {c.file_path} (lines {c.start_line}-{c.end_line})\n"
                f"{c.text.strip()}\n"
            )
        chunk_text = "\n\n".join(chunk_blocks)

        system = SystemMessage(
            content=(
                "You are an analysis tool that outputs JSON only. "
                "Do not include any explanation, only a JSON object."
            )
        )

        user = HumanMessage(
            content=(
                "The assistant produced the following final answer based on some code chunks.\n\n"
                f"Final answer:\n{final_answer}\n\n"
                f"Code chunks:\n{chunk_text}\n\n"
                "For each chunk id, assign an influence score between 0 and 1 that reflects "
                "how important that chunk was for the final answer.\n"
                "Return a single JSON object that maps chunk_id to score. Example:\n"
                "{ \"chunk_a\": 0.8, \"chunk_b\": 0.1 }\n"
            )
        )

        raw = self.llm.invoke([system, user]).content.strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            # Fallback if the model does not return valid JSON
            parsed = {}

        scores: Dict[str, float] = {}
        for scored in selection.selected_chunks:
            cid = scored.chunk.chunk_id
            try:
                scores[cid] = float(parsed.get(cid, 0.0))
            except (TypeError, ValueError):
                scores[cid] = 0.0

        return scores
