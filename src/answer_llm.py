from dotenv import load_dotenv
import os
import requests

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")

import os
import re
import requests
from typing import Any, Iterable

BASE_URL = "https://router.huggingface.co/v1"
MODEL = "HuggingFaceTB/SmolLM3-3B:hf-inference"

def _clean(text: str) -> str:
    # remove <think>...</think> if the model outputs it
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def _chunk_to_block(chunk: Any, idx: int) -> str:
    """
    Accepts:
      - string
      - dict with keys like: preview/text, url, source_id, section, jurisdiction, score, id/doc_id
    """
    if isinstance(chunk, str):
        text = chunk.strip()
        meta = ""
    elif isinstance(chunk, dict):
        text = (chunk.get("text") or chunk.get("preview") or "").strip()

        url = chunk.get("url", "")
        doc_id = chunk.get("doc_id") or chunk.get("id") or ""
        source_id = chunk.get("source_id", "")
        section = chunk.get("section", "")
        jurisdiction = chunk.get("jurisdiction", "")
        score = chunk.get("score", "")

        meta_parts = [p for p in [
            f"url={url}" if url else "",
            f"doc_id={doc_id}" if doc_id else "",
            f"source_id={source_id}" if source_id else "",
            f"jurisdiction={jurisdiction}" if jurisdiction else "",
            f"section={section}" if section else "",
            f"score={score}" if score != "" else "",
        ] if p]

        meta = " | ".join(meta_parts)
    else:
        text = str(chunk).strip()
        meta = ""

    if meta:
        return f"[{idx}] {text}\nMETA: {meta}"
    return f"[{idx}] {text}"

def answer_question(question: str, retrieved_chunks: Iterable[Any]) -> str:
    blocks = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        blocks.append(_chunk_to_block(ch, i))

    context = "\n\n".join(blocks)

    prompt = f"""
                You must follow these rules strictly.

                RULES:
                - You must do exactly ONE of the following:
                1) If the answer is contained in the TEXT, output ONLY the final answer.
                2) If the answer is NOT contained in the TEXT, output EXACTLY: I don't know.
                - Do NOT include reasoning or <think>.
                - Keep the answer short.

                TEXT:
                {context}

                QUESTION:
                {question}

                FINAL ANSWER:
                """.strip()

    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        # print helpful error
        try:
            print("HF router error:", r.json())
        except Exception:
            print("HF router error text:", r.text)
        r.raise_for_status()

    out = r.json()["choices"][0]["message"]["content"]
    return out
