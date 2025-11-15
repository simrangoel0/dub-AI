from __future__ import annotations

from backend.observability.langsmith_setup import init_langsmith
from langsmith import Client


def main():
    # This will load .env and set LANGSMITH_* env vars
    client = init_langsmith(project_name="default")

    if client is None:
        print("❌ init_langsmith returned None (no API key?)")
        return

    try:
        result = client.list_runs()
        print(result)
        print("✅ LangSmith API reachable.")
    except Exception as e:
        print("❌ Could not read project:", 'default')
        print("   Error:", repr(e))


if __name__ == "__main__":
    main()
