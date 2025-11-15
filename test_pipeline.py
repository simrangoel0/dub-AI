from __future__ import annotations

from pprint import pprint

from backend.pipeline import run_pipeline


def main():
    # Minimal fake chat history
    history = [
        {
            "role": "user",
            "content": "I added a logout function but the session cookie is still valid.",
        }
    ]

    payload = {
        "query": "Why is my logout not invalidating the session?",
        "history": history,
        # ContextManager reads files from backend/data/codebase, so this can stay empty for now
        "repo_state": {},
        # None / [] => let the context manager choose chunks automatically
        "selected_chunk_ids": None,
        # how many chunks to select max
        "top_k": 8,
    }

    result = run_pipeline(payload)

    print("\n=== RUN ID ===")
    print(result["run_id"])

    print("\n=== ANSWER (final_answer) ===")
    print(result["answer"]["final_answer"])

    print("\n=== USED CHUNKS ===")
    print(result["answer"].get("used_chunks", []))

    print("\n=== CONTEXT (summary) ===")
    ctx = result["context"]
    print("query:", ctx.get("query"))
    print("top_k:", ctx.get("top_k"))
    print("selected_chunks:", len(ctx.get("selected_chunks", [])))
    print("dropped_chunks:", len(ctx.get("dropped_chunks", [])))

    print("\n=== ATTRIBUTION (influence_scores) ===")
    pprint(result["attribution"].get("influence_scores", {}))

    print("\n=== TRACE FILE PATH ===")
    print(result["trace_file"])


if __name__ == "__main__":
    main()
