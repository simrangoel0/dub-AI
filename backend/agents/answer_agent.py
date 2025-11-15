from typing import Tuple
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from backend.core.models import ContextSelectionResult, ScoredChunk

load_dotenv()


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    # LLM loader
    return ChatOpenAI(model=model, temperature=temperature)

class AnswerAgent:
    """
    Final answer using the selected context from ContextSelectionResult.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = get_llm(model)

    @staticmethod
    def _format_chunks(chunks: list[ScoredChunk]) -> str:
        """
        Build a readable block for the LLM.
        For each ScoredChunk:
            [chunk_id] from path (lines X-Y)
            <chunk.text>
        """
        blocks = []
        for scored in chunks:
            c = scored.chunk
            header = f"[{c.chunk_id}] from {c.file_path} (lines {c.start_line}-{c.end_line})"
            body = c.text.strip()
            blocks.append(f"{header}\n{body}\n")
        return "\n".join(blocks)

    def _build_prompts(self, context: ContextSelectionResult) -> Tuple[str, str]:
        """
        Convert ContextSelectionResult into system + user prompts.
        """
        query = context.query
        chunk_block = self._format_chunks(context.selected_chunks)

        system_prompt = (
            "You are a precise and context grounded assistant. "
            "Your output should rely only on the provided context chunks. "
            "Output must include explanations."
        )

        user_prompt = (
            f"User request:\n{query}\n\n"
            "Relevant context chunks:\n"
            f"{chunk_block}\n"
            "Instructions:\n"
            "1. Only use the information in these chunks.\n"
            "2. Produce the exact final output the user needs.\n"
            "3. Do not include hidden reasoning or chain of thought.\n"
        )

        return system_prompt, user_prompt

    def run(self, context: ContextSelectionResult) -> dict:
        """
        Execute the AnswerAgent using the selected chunks.
        Returns a dict containing:
            final_output: str
            full_prompt: str
        """
        system_prompt, user_prompt = self._build_prompts(context)

        response = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return {
            "final_output": response.content.strip(),
            "full_prompt": system_prompt + "\n\n" + user_prompt,
        }