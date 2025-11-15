from backend.agents.context_manager import CodebaseIndex, select_context

index = CodebaseIndex(root_dir="backend/data/codebase")
index.build_index()

res = select_context(index, query="Why does auth fail on invalid token?")
print(len(res.selected_chunks), "selected")
for sc in res.selected_chunks:
    print(sc.chunk.chunk_id, sc.relevance_score)
    print(sc.rationale)