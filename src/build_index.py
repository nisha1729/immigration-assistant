import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

CHUNKS_PATH = os.getenv("CHUNKS_PATH")
INDEX_PATH = os.getenv("INDEX_PATH")
META_PATH = os.getenv("META_PATH")
EMBED_MODEL = os.getenv("EMBED_MODEL")

def load_chunks(path: str):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    os.makedirs("corpus", exist_ok=True)

    chunks = load_chunks(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(EMBED_MODEL)

    # normalize_embeddings=True makes cosine similarity == dot product
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    dim = embeddings.shape[1]

    # IndexFlatIP = inner product; with normalized vectors this is cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    # Save metadata in the same order as embeddings were added
    with open(META_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            # can drop text here to reduce size, but keep it for now for debugging
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[OK] Indexed {len(chunks)} chunks")
    print(f"[OK] Wrote FAISS index -> {INDEX_PATH}")
    print(f"[OK] Wrote metadata   -> {META_PATH}")

if __name__ == "__main__":
    main()
