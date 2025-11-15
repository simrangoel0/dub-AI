import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from backend.core.models import ContextSelectionResult, ScoredChunk

load_dotenv()


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(model=model, temperature=temperature)


class AttributionAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = get_llm(model)

    @staticmethod
    def _build_prompt(context: ContextSelectionResult, answer_output: dict) -> str:
        """
        The model sees:
        - final output
        - every selected chunk with chunk_id and text
        
        The model returns JSON:
        { "chunk_1": 0.7, "chunk_4": 0.2 }
        """
        final_output = answer_output.get("final_output", "")

        chunk_blocks = []
        for scored in context.selected_chunks:
            c = scored.chunk
            chunk_blocks.append(f"[{c.chunk_id}]\n{c.text.strip()}\n")

        chunk_text = "\n".join(chunk_blocks)

        return (
            "You are an attribution judge. Rate how much each context chunk influenced "
            "the final output. Return scores from 0 to 1.\n\n"
            f"Final output:\n{final_output}\n\n"
            "Context chunks:\n"
            f"{chunk_text}\n"
            "Return ONLY valid JSON, like:\n"
            "{ \"chunk_1\": 0.5, \"chunk_2\": 0.9 }\n"
        )

    def run(self, context: ContextSelectionResult, answer_output: dict) -> dict:
        prompt = self._build_prompt(context, answer_output)

        raw = self.llm.invoke(
            [
                {"role": "system", "content": "Only return JSON."},
                {"role": "user", "content": prompt}
            ]
        ).content.strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {sc.chunk.chunk_id: 0.0 for sc in context.selected_chunks}

        influence_scores = {}
        for scored in context.selected_chunks:
            cid = scored.chunk.chunk_id
            influence_scores[cid] = float(parsed.get(cid, 0.0))

        return {"influence_scores": influence_scores}