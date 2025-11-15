import os
from dotenv import load_dotenv

from backend.core.bedrock import get_chat_model as _base_get_chat_model

load_dotenv()


def get_chat_model(model_name: str | None = None):
    """
    Shared helper for all agents.

    If model_name is None, uses HOLISTIC_AI_MODEL from the environment.
    """
    name = model_name or os.getenv("HOLISTIC_AI_MODEL")
    return _base_get_chat_model(name)
