from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langsmith import Client


def _load_env(env_path: str | Path = ".env") -> None:
    """
    Load environment variables from a .env file if it exists.
    This lets each dev keep their own keys locally.
    """
    path = Path(env_path)
    if path.exists():
        load_dotenv(path)
        print(f"[LangSmith] Loaded environment from {path}")
    else:
        print(f"[LangSmith] No {path} file found – relying on OS env vars.")


def init_langsmith(project_name: str = "glass-box-debugger") -> Optional[Client]:
    """
    Initialise LangSmith tracing for the whole backend.

    - Loads .env
    - Ensures LANGSMITH_TRACING is enabled
    - Ensures LANGSMITH_PROJECT is set (if provided)
    - Returns a langsmith.Client you can reuse (optional)
    """
    _load_env()

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("[LangSmith] ❌ LANGSMITH_API_KEY not set – tracing DISABLED.")
        print("           Get a key from https://smith.langchain.com")
        return None

    # LANGSMITH_TRACING must be "true" for @traceable to actually send traces
    # (defaults to true, but we set it explicitly to be safe). :contentReference[oaicite:2]{index=2}
    if os.getenv("LANGSMITH_TRACING") is None:
        os.environ["LANGSMITH_TRACING"] = "true"
        print("[LangSmith] LANGSMITH_TRACING set to 'true'")

    # Default SaaS endpoint (change if you ever self-host). :contentReference[oaicite:3]{index=3}
    if not os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        print("[LangSmith] LANGSMITH_ENDPOINT set to 'https://api.smith.langchain.com'")

    # Group traces under a project (shows up in the LangSmith UI). :contentReference[oaicite:4]{index=4}
    if project_name:
        os.environ.setdefault("LANGSMITH_PROJECT", project_name)

    client = Client()  # picks up env vars

    key_preview = api_key[:6] + "..." if len(api_key) > 6 else "********"
    project = os.getenv("LANGSMITH_PROJECT", "default")

    print("[LangSmith] ✅ Tracing ENABLED")
    print(f"           Project: {project}")
    print(f"           API key: {key_preview}")
    print("           Dashboard: https://smith.langchain.com")

    return client