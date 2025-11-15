from __future__ import annotations

import json
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_valyu import ValyuChat
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import ContextSelectionResult, ScoredChunk

load_dotenv()


class DebugAnswer(BaseModel):
    """
    Structured answer used for debugging help.
    This is what the LLM is asked to produce as JSON.
    """
    summary: str = Field(
        description="One or two sentence summary of what is going wrong."
    )
    root_cause: str = Field(
        description="Clear description of the most likely root cause."
    )
    suggested_fix: str = Field(
        description="High level explanation of how to fix the issue."
    )
    patch: str = Field(
        description="Concrete code snippet or pseudo code showing the fix.",
        default=""
    )
    validation_steps: List[str] = Field(
        description="Step by step instructions to validate the fix.",
        default_factory=list,
    )
    used_chunk_ids: List[str] = Field(
        description="List of chunk_ids that were most important for your reasoning.",
        default_factory=list,
    )


def get_llm():
    """Create the Valyu LLM client using team credentials."""
    return ValyuChat(
        api_key=os.getenv("VALYU_API_TOKEN"),
        team_id=os.getenv("VALYU_TEAM_ID"),
        model=os.getenv("VALYU_MODEL"),
        endpoint=os.getenv("VALYU_API_ENDPOINT"),
        max_retries=2,
    )


class AnswerAgent:
    """
    Generates a structured debugging answer using the selected context.

    Input:
        ContextSelectionResult

    Output dict:
        - final_output: human readable answer (combined sections)
        - structured: validated DebugAnswer as dict
        - full_prompt: full system + user prompt used
    """

    def __init__(self):
        self.llm = get_llm()

    @staticmethod
    def _format_chunks(chunks: List[ScoredChunk]) -> str:
        """
        Convert selected chunks into a readable block for the model.
        """
        blocks = []
        for scored in chunks:
            c = scored.chunk
            header = f"[{c.chunk_id}] from {c.file_path} (lines {c.start_line}-{c.end_line})"
            blocks.append(f"{header}\n{c.text.strip()}\n")
        return "\n".join(blocks)

    @staticmethod
    def _build_json_schema_hint() -> str:
        """
        Small textual schema hint for the model, based on DebugAnswer.
        We do not rely on with_structured_output here, only prompt plus JSON.
        """
        return (
            "Return ONLY valid JSON with the following shape:\n"
            "{\n"
            '  "summary": string,\n'
            '  "root_cause": string,\n'
            '  "suggested_fix": string,\n'
            '  "patch": string,\n'
            '  "validation_steps": [string, ...],\n'
            '  "used_chunk_ids": [string, ...]\n'
            "}\n"
        )

    def run(self, context: ContextSelectionResult) -> dict:
        """
        Produce a structured debugging answer using the LLM.

        The model is instructed to:
        - only use the selected chunks
        - explicitly reference chunk_ids in reasoning where helpful
        - respond in strict JSON matching DebugAnswer
        """
        chunk_text = self._format_chunks(context.selected_chunks)
        json_schema_hint = self._build_json_schema_hint()

        system_prompt = (
            "You are an expert debugging assistant for Python and general code. "
            "You receive a user request plus a set of context chunks extracted from "
            "their codebase or conversation. Your job is to diagnose the issue and "
            "propose a clear, actionable fix based only on the provided chunks."
        )

        user_prompt = (
            f"User request:\n{context.query}\n\n"
            f"Relevant context chunks:\n{chunk_text}\n"
            "Guidelines:\n"
            "- Treat this as a debugging task.\n"
            "- When you draw evidence from a chunk, mention its chunk_id like [fileA_0].\n"
            "- Focus on the minimal set of chunks that explain the issue.\n"
            "- If the context is incomplete, say what is missing instead of guessing.\n\n"
            "Output format:\n"
            f"{json_schema_hint}\n"
            "Return only the JSON object, with no extra text."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # Try to parse and validate the JSON
        try:
            data = json.loads(raw)
            structured = DebugAnswer(**data)
        except (json.JSONDecodeError, ValidationError):
            # Fallback: wrap the raw text into a minimal DebugAnswer
            structured = DebugAnswer(
                summary="Model did not return valid JSON, falling back to raw text.",
                root_cause="See raw model output.",
                suggested_fix="Manually inspect the explanation from the model.",
                patch="",
                validation_steps=["Rerun the agent or adjust the context selection."],
                used_chunk_ids=[sc.chunk.chunk_id for sc in context.selected_chunks],
            )

        # Build a user friendly final_output from the structured fields
        lines = [
            "Summary:",
            structured.summary,
            "",
            "Most likely root cause:",
            structured.root_cause,
            "",
            "Suggested fix:",
            structured.suggested_fix,
        ]

        if structured.patch:
            lines += ["", "Proposed patch:", structured.patch]

        if structured.validation_steps:
            lines += ["", "How to validate the fix:"]
            for step in structured.validation_steps:
                lines.append(f"- {step}")

        if structured.used_chunk_ids:
            lines += [
                "",
                "Chunks that were most influential:",
                ", ".join(structured.used_chunk_ids),
            ]

        final_output = "\n".join(lines)

        return {
            "final_output": final_output,
            "structured": structured.model_dump(),
            "full_prompt": system_prompt + "\n\n" + user_prompt,
        }
