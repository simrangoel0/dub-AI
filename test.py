from backend.agents.context_manager import ContextManager

# Fake conversation history for testing
conversation = [
    {"role": "user", "content": "Hi, I'm getting an error when refreshing auth tokens."},
    {"role": "assistant", "content": "Can you share the error message and relevant code?"},
    {"role": "user", "content": "The token seems to still work even after it should be expired."},
]

cm = ContextManager(root_dir="backend/data/codebase")

# CASE 1: Let the agent choose (no manual chunk selection)
result = cm.select(
    query="Why is token refresh accepting expired tokens?",
    conversation=conversation,
    top_k=5,
)

print("Selection mode:", result.meta["selection_mode"])
print("\n=== SELECTED CHUNKS ===")
for sc in result.selected_chunks:
    print(f"- {sc.chunk.chunk_id} ({sc.chunk.source})")
    print(f"  File: {sc.chunk.file_path}")
    print(f"  Rationale: {sc.rationale}")
    print()

print("\n=== DROPPED CHUNKS ===")
for sc in result.dropped_chunks:
    print(f"- {sc.chunk.chunk_id} ({sc.chunk.source})")
    print(f"  Rationale: {sc.rationale}")
    print()

# CASE 2: Simulate user manually selecting chunks
result_user = cm.select(
    query="Check token logic",
    conversation=conversation,
    top_k=5,
    user_selected_chunk_ids=["auth.py_2", "chat_2"],  # example IDs
)

print("\nSelection mode (user path):", result_user.meta["selection_mode"])
print("\n=== USER-SELECTED CHUNKS ===")
for sc in result_user.selected_chunks:
    print(f"- {sc.chunk.chunk_id} ({sc.chunk.source}) -> {sc.rationale}")