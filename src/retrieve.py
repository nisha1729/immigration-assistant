import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "corpus/index.faiss"
META_PATH = "corpus/meta.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 8  # start with 8; later you'll rerank/filter

def load_meta(path: str):
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question"')
        raise SystemExit(1)

    query = sys.argv[1]

    index = faiss.read_index(INDEX_PATH)
    meta = load_meta(META_PATH)

    model = SentenceTransformer(EMBED_MODEL)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q, TOP_K)

    print("\nQUERY:", query)
    print("\nTOP RESULTS:\n")

    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx == -1:
            continue
        c = meta[idx]
        preview = c["text"][:300].replace("\n", " ")
        print(f"#{rank} score={float(score):.3f}  {c['chunk_id']}")
        print(f"   source_id={c['source_id']} | jurisdiction={c.get('jurisdiction')} | authority={c.get('authority_level')} | section={c.get('section')}")
        print(f"   url={c.get('url')}")
        print(f"   preview: {preview}...")
        print()

if __name__ == "__main__":
    main()
