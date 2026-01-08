from dotenv import load_dotenv
import os
import requests

load_dotenv()  # reads .env automatically

HF_TOKEN = os.getenv("HF_API_TOKEN")

# OpenAI-compatible endpoint hosted by Hugging Face router
BASE_URL = "https://router.huggingface.co/v1"

# Pick a model that is confirmed to work with the hf-inference provider (per HF docs example)
MODEL = "HuggingFaceTB/SmolLM3-3B:hf-inference"

def answer_question(question: str, retrieved_texts: list[str]) -> str:
    context = "\n\n".join(retrieved_texts)

    prompt = f"""Answer the question using ONLY the information in TEXT.

                IMPORTANT:
                - Do NOT explain your reasoning.
                - Do NOT include thoughts, analysis, or tags like <think>.
                - Give only the final answer.

                If the answer is not in TEXT, say:
                "I don't know."

                TEXT:
                {context}

                QUESTION:
                {question}
                """

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }

    r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=90)

    if r.status_code != 200:
        # Show the real error message (super helpful)
        try:
            print("HF router error:", r.json())
        except Exception:
            print("HF router error text:", r.text)
        r.raise_for_status()

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


if __name__=="__main__":
    chunks = [
        "The responsible authority for X is Authority A.",
        "The notification deadline for X is 14 days."
    ]

    print(answer_question(
        "Who is the authority and what is the deadline for X?",
        chunks
    ))
