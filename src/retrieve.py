import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH")
META_PATH = os.getenv("META_PATH")
EMBED_MODEL = os.getenv("EMBED_MODEL")


def load_meta(path: str):
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def search(query, top_k=8):
    index = faiss.read_index(INDEX_PATH)
    meta = load_meta(META_PATH)

    model = SentenceTransformer(EMBED_MODEL)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q, top_k)

    print("\nQUERY:", query)
    print("\nTOP RESULTS:\n")


    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx == -1:
            continue
        c = meta[idx]
        results.append({
            "text": c["text"],
            "chunk_id": c["chunk_id"],
            "source_id": c["source_id"],
            "url": c.get("url"),
            "jurisdiction": c.get("jurisdiction"),
            "authority_level": c.get("authority_level"),
            "section": c.get("section"),
            "score": float(score),
            "rank": rank,
        })

        # for debugging    
        preview = c["text"][:300].replace("\n", " ")
        print(f"#{rank} score={float(score):.3f}  {c['chunk_id']}")
        print(f"   source_id={c['source_id']} | jurisdiction={c.get('jurisdiction')} | authority={c.get('authority_level')} | section={c.get('section')}")
        print(f"   url={c.get('url')}")
        print(f"   preview: {preview}...")
        print()

    return results
