# backend/core/bedrock.py

from __future__ import annotations

import os
from typing import List, Optional

import httpx
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration

load_dotenv()


def _to_openai_dict(msg: BaseMessage) -> dict:
    """Convert LangChain messages into simple role/content dicts."""
    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    else:
        role = "user"
    return {"role": role, "content": msg.content}


class HolisticAIBedrockChat(BaseChatModel):
    """
    Minimal LangChain compatible wrapper for the Holistic AI Bedrock proxy.
    Uses the HTTP API described in the hackathon docs.
    """

    def __init__(
        self,
        team_id: str,
        api_token: str,
        endpoint: str,
        model: str,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.team_id = team_id
        self.api_token = api_token
        self.endpoint = endpoint
        self.model = model
        self.client = client or httpx.Client(timeout=30.0)

    @property
    def _llm_type(self) -> str:
        return "holistic_ai_bedrock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        """Core call used by LangChain and LangGraph."""
        formatted = [_to_openai_dict(m) for m in messages]

        payload = {
            "team_id": self.team_id,
            "model": self.model,
            "messages": formatted,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Token": self.api_token,
        }

        resp = self.client.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Hackathon API returns {"message": {"role": "assistant", "content": "..."}}
        if isinstance(data, dict) and "message" in data:
            content = data["message"]["content"]
        else:
            # Fallback if schema is slightly different
            content = str(data)

        ai_msg = AIMessage(content=content)
        generation = ChatGeneration(message=ai_msg)
        return ChatResult(generations=[generation])


def get_chat_model(model_name: Optional[str] = None) -> HolisticAIBedrockChat:
    """
    Helper that reads configuration from the environment and returns a chat model.
    """
    team_id = os.getenv("HOLISTIC_AI_TEAM_ID")
    api_token = os.getenv("HOLISTIC_AI_API_TOKEN")
    endpoint = os.getenv("HOLISTIC_AI_API_ENDPOINT")
    default_model = os.getenv("HOLISTIC_AI_MODEL")

    if not team_id or not api_token or not endpoint:
        raise RuntimeError(
            "Holistic AI Bedrock env vars are missing. "
            "Set HOLISTIC_AI_TEAM_ID, HOLISTIC_AI_API_TOKEN, HOLISTIC_AI_API_ENDPOINT, HOLISTIC_AI_MODEL."
        )

    model = model_name or default_model
    if not model:
        raise RuntimeError("HOLISTIC_AI_MODEL is not set and no model_name was provided.")

    return HolisticAIBedrockChat(
        team_id=team_id,
        api_token=api_token,
        endpoint=endpoint,
        model=model,
    )
