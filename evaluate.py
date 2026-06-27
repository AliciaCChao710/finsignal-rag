"""Lightweight RAGAS-style evaluation of the RAG pipeline.

The `ragas` library (0.4.x) is incompatible with this project's langchain 1.x
stack, so this script implements the two label-free metrics RAGAS is best known
for, using Gemini as the judge:

  * Faithfulness     — is every claim in the answer supported by the retrieved
                       passages? (penalises hallucination)
  * Context precision — are the retrieved passages actually relevant to the
                       question? (rank-aware, the RAGAS definition)

Neither metric needs ground-truth answers. Runs a small question set sequentially
to stay within the Gemini free-tier rate limits.

Run:
    python evaluate.py
"""

import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from generate import MODEL, generate_answer  # noqa: E402
from retrieve import retrieve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_FILE = Path(__file__).resolve().parent / "data" / "evaluation_results.json"
K = 6
SLEEP_BETWEEN_CALLS = 4  # seconds; keeps us under the free-tier requests-per-minute limit

# Small, label-free test set spanning tech / banks / comparison
TEST_QUESTIONS: list[tuple[str, str | None]] = [
    ("What risks does NVIDIA highlight regarding artificial intelligence?", "tech"),
    ("How do banks describe credit risk in their filings?", "banks"),
    ("Compare the main risks technology companies versus banks emphasise.", None),
]


def _judge(prompt: str) -> dict:
    """Call Gemini and parse a JSON object back (with one retry on rate limit)."""
    from google import genai
    from google.genai import types
    import os

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return json.loads(resp.text)
        except Exception as exc:  # rate limit or transient error
            if attempt == 0 and "429" in str(exc):
                log.warning("Rate limited, waiting 45s then retrying...")
                time.sleep(45)
                continue
            raise


def faithfulness(answer: str, contexts: list[str]) -> tuple[float, dict]:
    """Fraction of the answer's claims that are supported by the retrieved passages."""
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    prompt = (
        "Break the ANSWER into atomic factual claims. For each claim, decide whether it is "
        "supported by the CONTEXT passages. Reply as JSON: "
        '{"claims": [{"claim": "...", "supported": true/false}]}.\n\n'
        f"ANSWER:\n{answer}\n\nCONTEXT:\n{ctx}"
    )
    data = _judge(prompt)
    claims = data.get("claims", [])
    if not claims:
        return 0.0, data
    supported = sum(1 for c in claims if c.get("supported"))
    return supported / len(claims), data


def context_precision(question: str, contexts: list[str]) -> tuple[float, dict]:
    """Rank-aware precision: are the top-ranked passages the relevant ones? (RAGAS definition)."""
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    prompt = (
        "For each numbered CONTEXT passage, decide whether it is relevant to answering the "
        "QUESTION. Reply as JSON with a boolean per passage, in order: "
        '{"verdicts": [true, false, ...]}.\n\n'
        f"QUESTION: {question}\n\nCONTEXT:\n{ctx}"
    )
    data = _judge(prompt)
    verdicts = [bool(v) for v in data.get("verdicts", [])]
    total_relevant = sum(verdicts)
    if total_relevant == 0:
        return 0.0, data
    # mean of Precision@i over the positions that are relevant
    running_hits = 0
    weighted = 0.0
    for i, rel in enumerate(verdicts, 1):
        if rel:
            running_hits += 1
            weighted += running_hits / i
    return weighted / total_relevant, data


def main() -> None:
    load_dotenv()
    results = []
    log.info("Evaluating %d questions (model: %s)...", len(TEST_QUESTIONS), MODEL)

    for q, sector in TEST_QUESTIONS:
        log.info("Q: %s (sector=%s)", q, sector or "all")
        hits = retrieve(q, k=K, sector=sector)
        contexts = [h["text"] for h in hits]
        ans = generate_answer(q, hits)
        time.sleep(SLEEP_BETWEEN_CALLS)

        faith, faith_detail = faithfulness(ans, contexts)
        time.sleep(SLEEP_BETWEEN_CALLS)
        prec, prec_detail = context_precision(q, contexts)
        time.sleep(SLEEP_BETWEEN_CALLS)

        log.info("  faithfulness=%.2f  context_precision=%.2f", faith, prec)
        results.append({
            "question": q,
            "sector": sector,
            "answer": ans,
            "faithfulness": round(faith, 3),
            "context_precision": round(prec, 3),
            "faithfulness_detail": faith_detail,
            "context_precision_detail": prec_detail,
        })

    n = len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_prec = sum(r["context_precision"] for r in results) / n

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Evaluation summary =====")
    print(f"{'question':<55} {'faith':>6} {'ctx_prec':>9}")
    for r in results:
        print(f"{r['question'][:53]:<55} {r['faithfulness']:>6.2f} {r['context_precision']:>9.2f}")
    print("-" * 72)
    print(f"{'AVERAGE':<55} {avg_faith:>6.2f} {avg_prec:>9.2f}")
    print(f"\nFull details saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
