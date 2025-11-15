import json
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ValyuChat
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import ContextSelectionResult

load_dotenv()


def get_llm():
    """Create the Valyu LLM client used for attribution scoring."""
    return ValyuChat(
        api_key=os.getenv("VALYU_API_TOKEN"),
        team_id=os.getenv("VALYU_TEAM_ID"),
        model=os.getenv("VALYU_MODEL"),
        endpoint=os.getenv("VALYU_API_ENDPOINT"),
        max_retries=2,
    )


class AttributionAgent:
    """
    Computes influence scores for all selected chunks.

    Input:
        - ContextSelectionResult
        - AnswerAgent output dict

    Output:
        dict with:
            "influence_scores": { chunk_id: float }
    """

    def __init__(self):
        self.llm = get_llm()

    @staticmethod
    def _build_prompt(context: ContextSelectionResult, answer_output: dict) -> str:
        """
        Construct the attribution instruction block.
        """
        final_output = answer_output["final_output"]
        chunk_sections = []

        for scored in context.selected_chunks:
            c = scored.chunk
            chunk_sections.append(f"[{c.chunk_id}]\n{c.text.strip()}\n")

        all_chunks_text = "\n".join(chunk_sections)

        return (
            "Rate how much each chunk influenced the final output. "
            "Assign a score from 0 to 1 for each chunk.\n\n"
            f"Final output:\n{final_output}\n\n"
            f"Chunks:\n{all_chunks_text}\n"
            "Return only valid JSON mapping chunk ids to scores.\n"
        )

    def run(self, context: ContextSelectionResult, answer_output: dict) -> dict:
        """
        Generate influence scores for each chunk.
        """
        prompt = self._build_prompt(context, answer_output)

        messages = [
            SystemMessage(content="Provide only valid JSON."),
            HumanMessage(content=prompt),
        ]

        raw = self.llm.invoke(messages).content.strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {sc.chunk.chunk_id: 0.0 for sc in context.selected_chunks}

        influence_scores = {
            sc.chunk.chunk_id: float(parsed.get(sc.chunk.chunk_id, 0.0))
            for sc in context.selected_chunks
        }

        return {"influence_scores": influence_scores}
