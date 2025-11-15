from __future__ import annotations

import json
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_valyu import ValyuChat
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import ContextSelectionResult

load_dotenv()


class ChunkInfluence(BaseModel):
    """
    Influence information for a single chunk.
    """
    chunk_id: str = Field(description="The chunk_id from the context manager.")
    score: float = Field(
        description="Influence score between 0 and 1.",
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        description="Short explanation of how this chunk influenced the answer."
    )


class AttributionOutput(BaseModel):
    """
    Structured attribution result used by the glassbox UI.
    """
    influences: List[ChunkInfluence] = Field(
        description="List of influence entries, one per selected chunk."
    )


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
        - answer_output dict from AnswerAgent.run

    Output dict:
        - influence_scores: { chunk_id: float }
        - influence_details: AttributionOutput as dict
    """

    def __init__(self):
        self.llm = get_llm()

    @staticmethod
    def _build_json_schema_hint(context: ContextSelectionResult) -> str:
        """
        Build a description of the JSON format we expect from the model.
        """
        chunk_ids = [sc.chunk.chunk_id for sc in context.selected_chunks]
        ids_str = ", ".join(chunk_ids)

        return (
            "Return ONLY valid JSON with the following shape:\n"
            "{\n"
            '  "influences": [\n'
            "    {\n"
            '      "chunk_id": string,   // one of: '
            f"{ids_str}\n"
            '      "score": number,      // between 0 and 1\n'
            '      "rationale": string   // short explanation\n'
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n"
        )

    def _build_prompt(self, context: ContextSelectionResult, answer_output: dict) -> str:
        """
        Construct the attribution instruction block.

        The model sees:
            - final_output from the AnswerAgent
            - all selected chunks with their text
        """
        final_output = answer_output["final_output"]
        chunk_sections = []

        for scored in context.selected_chunks:
            c = scored.chunk
            chunk_sections.append(
                f"[{c.chunk_id}] from {c.file_path} (lines {c.start_line}-{c.end_line})\n"
                f"{c.text.strip()}\n"
            )

        all_chunks_text = "\n".join(chunk_sections)
        schema_hint = self._build_json_schema_hint(context)

        return (
            "You are an attribution judge for a debugging assistant. "
            "Your task is to estimate how much each context chunk contributed to "
            "the final answer.\n\n"
            "Guidelines:\n"
            "- Score 0.0 means the chunk was not used at all.\n"
            "- Score near 1.0 means the answer depends heavily on this chunk.\n"
            "- Use intermediate values for partial influence.\n"
            "- Base your judgement only on the content of the chunks and the final answer.\n\n"
            f"Final answer:\n{final_output}\n\n"
            f"Context chunks:\n{all_chunks_text}\n\n"
            "Output format:\n"
            f"{schema_hint}\n"
            "Return only the JSON object, with no extra text."
        )

    def run(self, context: ContextSelectionResult, answer_output: dict) -> dict:
        """
        Generate influence scores for each chunk.

        If the JSON is malformed, all scores fall back to zero.
        """
        prompt = self._build_prompt(context, answer_output)

        messages = [
            SystemMessage(content="Return only JSON, no prose."),
            HumanMessage(content=prompt),
        ]

        raw = self.llm.invoke(messages).content.strip()

        # Try to parse and validate the JSON
        try:
            data = json.loads(raw)
            structured = AttributionOutput(**data)
        except (json.JSONDecodeError, ValidationError):
            # Safe fallback - no influence
            influences = [
                ChunkInfluence(
                    chunk_id=sc.chunk.chunk_id,
                    score=0.0,
                    rationale="Failed to parse attribution JSON.",
                )
                for sc in context.selected_chunks
            ]
            structured = AttributionOutput(influences=influences)

        # Build the simple dict for compatibility
        influence_scores = {
            inf.chunk_id: inf.score for inf in structured.influences
        }

        return {
            "influence_scores": influence_scores,
            "influence_details": structured.model_dump(),
        }
