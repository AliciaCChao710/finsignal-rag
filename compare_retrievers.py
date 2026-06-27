"""Compare TF-IDF retrieval (W2, sparse) vs embedding retrieval (W3, dense).

For each test question we retrieve top-k with both methods and score retrieval
quality with the same context-precision judge used in evaluate.py. This isolates
the effect of the retrieval representation, holding everything else constant.

Run:
    python compare_retrievers.py
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retrieve import retrieve as retrieve_embed  # noqa: E402  (dense / embeddings)
from retrieve_tfidf import retrieve as retrieve_tfidf  # noqa: E402  (sparse / TF-IDF)
from evaluate import context_precision, TEST_QUESTIONS  # noqa: E402

K = 6
SLEEP = 4  # seconds between judge calls (Gemini free-tier rate limit)


def main() -> None:
    load_dotenv()
    rows = []
    for q, sector in TEST_QUESTIONS:
        emb = retrieve_embed(q, k=K, sector=sector)
        cp_emb, _ = context_precision(q, [h["text"] for h in emb])
        time.sleep(SLEEP)

        tfidf = retrieve_tfidf(q, k=K, sector=sector)
        cp_tfidf, _ = context_precision(q, [h["text"] for h in tfidf])
        time.sleep(SLEEP)

        rows.append((q, cp_tfidf, cp_emb))
        print(f"  {q[:50]:<52} TF-IDF={cp_tfidf:.2f}  Embedding={cp_emb:.2f}")

    n = len(rows)
    avg_tfidf = sum(r[1] for r in rows) / n
    avg_emb = sum(r[2] for r in rows) / n

    print("\n===== Context precision: TF-IDF (W2) vs Embedding (W3) =====")
    print(f"{'question':<52} {'TF-IDF':>7} {'Embed':>7}")
    for q, cp_tfidf, cp_emb in rows:
        print(f"{q[:50]:<52} {cp_tfidf:>7.2f} {cp_emb:>7.2f}")
    print("-" * 68)
    print(f"{'AVERAGE':<52} {avg_tfidf:>7.2f} {avg_emb:>7.2f}")


if __name__ == "__main__":
    main()
