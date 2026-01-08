from src.answer_llm import answer_question
from src import retrieve

question = "Whom do I notify if I lose my job in the first 12 months?"

results = retrieve.search(question, top_k=8)

# convert to what answer_llm expects (dicts)
chunks_for_llm = []
for r in results:
    chunks_for_llm.append({
        "text": r["text"][:1200],
        "url": r["url"],
        "chunk_id": r["chunk_id"],
        "jurisdiction": r.get("jurisdiction"),
        "section": r.get("section"),
        "score": r.get("score"),
    })

answer = answer_question(question, chunks_for_llm)
print(answer)
