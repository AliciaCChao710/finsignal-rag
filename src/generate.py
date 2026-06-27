"""Combine retrieved chunks with a prompt and call the LLM to generate an answer.

The final stage of RAG: retrieve the most relevant 10-K passages with retrieve.py,
assemble them into a "context", and send it together with the question to the LLM
(Google Gemini, free tier), instructing it to answer ONLY from the provided
passages and to cite its sources. Source-traceable answers are exactly what makes
RAG stronger than asking the LLM directly.

Requires GEMINI_API_KEY in .env (free key at https://aistudio.google.com/apikey).

CLI:
    python src/generate.py "Compare the main risks for tech vs banks"
    python src/generate.py "What are Google's revenue sources?" --sector tech --k 5
"""

import argparse
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieve import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"  # free tier, fast; swap to gemini-2.5-pro for more power (smaller free quota)
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a financial-document analysis assistant. The user gives you a
question plus passages retrieved from public companies' 10-K annual reports
(each tagged with [number] and ticker).

Rules:
1. Answer ONLY from the provided passages. Do not add facts that are not in them.
2. Cite the passage [number] after each claim so the reader can trace the source.
3. If the passages are insufficient to answer, say so explicitly. Do not fabricate.
4. When comparing companies or sectors, state the differences clearly.
Answer in English."""


def build_context(hits: list[dict]) -> str:
    """Assemble retrieved chunks into a numbered context string with provenance."""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        blocks.append(
            f"[{i}] ticker={m['ticker']} sector={m['sector']} filing_date={m['filing_date']}\n"
            f"{h['text'].strip()}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, hits: list[dict]) -> str:
    """Build context from already-retrieved hits, call Gemini, return a cited answer."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env (https://aistudio.google.com/apikey)")

    if not hits:
        return "No relevant passages found. Build the vector store first (python src/embed.py)."

    context = build_context(hits)
    user_message = (
        f"Question: {query}\n\n"
        f"Passages retrieved from 10-K filings:\n\n{context}\n\n"
        f"Answer the question using only these passages, and cite the [number] sources."
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return response.text or ""


def answer(query: str, k: int = 6, sector: str | None = None) -> str:
    """Retrieve top-k chunks, then generate a cited answer (CLI convenience wrapper)."""
    hits = retrieve(query, k=k, sector=sector)
    return generate_answer(query, hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG question answering")
    parser.add_argument("query", help="the question to ask")
    parser.add_argument("--k", type=int, default=6, help="how many chunks to retrieve as context (default 6)")
    parser.add_argument("--sector", choices=["tech", "banks"], help="restrict to one sector")
    args = parser.parse_args()

    print("\nRetrieving and generating answer...\n")
    print(answer(args.query, k=args.k, sector=args.sector))


if __name__ == "__main__":
    main()
