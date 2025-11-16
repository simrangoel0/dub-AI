from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.models import ContextSelectionResult, AttributionOutput
from holistic_ai_bedrock import get_chat_model


class AttributionAgent:
    """
    Converts AttributionOutput into UI-ready structures:
    - influence_scores: {chunk_id: float}
    - response_context: {
         messageId, runId,
         chunks: [...],
         droppedChunks: [...]
      }
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.base_llm = get_chat_model(model_name)
        self.llm = self.base_llm.with_structured_output(AttributionOutput)

    def _build_prompt(self, selection: ContextSelectionResult, answer: dict) -> str:
        final_answer = answer["final_answer"]

        text = (
            "You are an attribution engine.\n"
            "For EACH CHUNK you MUST produce a JSON object with EXACTLY these keys:\n"
            "{\n"
            "  \"chunk_id\": string,\n"
            "  \"score\": float between 0 and 1,\n"
            "  \"evidence\": list of 1–3 short quotes taken from the final answer,\n"
            "  \"explanation\": one sentence\n"
            "}\n\n"
            "Do NOT omit chunk_id. Do NOT change field names.\n"
            "Your output MUST match the JSON schema.\n\n"
            f"### Final Answer\n{final_answer}\n\n"
            "### Chunks (use the EXACT chunk_id shown):\n"
        )

        for sc in selection.selected_chunks + selection.dropped_chunks:
            c = sc.chunk
            text += (
                f"- chunk_id: {c.chunk_id}\n"
                f"  file: {c.file_path} ({c.start_line}-{c.end_line})\n"
                f"  similarity_score: {sc.similarity_score}\n"
                f"  relevance_score: {sc.relevance_score}\n"
                f"  rationale: {sc.rationale}\n"
                f"  text:\n{c.text.strip()}\n\n"
            )

        return text

    def run(
        self,
        selection: ContextSelectionResult,
        answer: dict,
        run_id: str,
        message_id: str,
    ):
        prompt = self._build_prompt(selection, answer)
        
        print("breaking prompt")
        print(prompt)
        structured: AttributionOutput = self.llm.invoke(prompt)

        # influence map for TraceLogger
        influence_scores = {
            entry.chunk_id: float(entry.score)
            for entry in structured.chunks
        }

        # Explanations map
        explanations = {
            entry.chunk_id: {
                "evidence": entry.evidence,
                "explanation": entry.explanation
            }
            for entry in structured.chunks
        }

        # Build response context for the UI
        def make_chunk(sc, selected: bool):
            cid = sc.chunk.chunk_id
            attrib = explanations.get(cid, {})
            return {
                "id": cid,
                "selected": selected,
                "influenceScore": influence_scores.get(cid, 0.0),
                "rationale": sc.rationale,
                "reasoning": attrib.get("explanation"),
                "evidence": attrib.get("evidence", [])
            }

        selected_chunks = [
            make_chunk(sc, True) for sc in selection.selected_chunks
        ]
        dropped_chunks = [
            make_chunk(sc, False) for sc in selection.dropped_chunks
        ]

        response_context = {
            "messageId": message_id,
            "runId": run_id,
            "chunks": selected_chunks,
            "droppedChunks": dropped_chunks,
        }

        return {
            "influence_scores": influence_scores,
            "raw": structured.model_dump(),
            "response_context": response_context,
            "explanations": explanations,
        }
