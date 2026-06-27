"""TF-IDF retrieval baseline (sparse vectors) to compare against the dense
embedding retrieval in retrieve.py.

This is the W2 "represent text with math (TF-IDF)" counterpart to the W3
"vector semantics / embeddings" approach. Same interface as retrieve.retrieve()
so the two can be compared head-to-head (see compare_retrievers.py).

CLI:
    python src/retrieve_tfidf.py "What are the main risk factors?" --k 5 --sector banks
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"

# ticker -> sector (same mapping as embed.py) so we can filter by sector
SECTOR = {
    "GOOGL": "tech", "MSFT": "tech", "NVDA": "tech",
    "JPM": "banks", "GS": "banks", "BAC": "banks",
}


@lru_cache(maxsize=1)
def _build():
    """Load chunks and fit the TF-IDF matrix once (cached for reuse)."""
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    matrix = vectorizer.fit_transform(texts)
    return chunks, vectorizer, matrix


def retrieve(query: str, k: int = 5, sector: str | None = None) -> list[dict]:
    """Return top-k chunks by TF-IDF cosine similarity (same shape as retrieve.retrieve)."""
    chunks, vectorizer, matrix = _build()
    sims = cosine_similarity(vectorizer.transform([query]), matrix)[0]

    # rank all, then keep the top-k that match the sector filter
    order = sims.argsort()[::-1]
    hits = []
    for idx in order:
        c = chunks[idx]
        if sector and SECTOR.get(c["ticker"]) != sector:
            continue
        hits.append({
            "text": c["text"],
            "metadata": {
                "ticker": c["ticker"],
                "sector": SECTOR.get(c["ticker"], "unknown"),
                "filing_date": c["filing_date"],
                "accession": c["accession"],
                "source": c["source"],
            },
            "distance": float(1.0 - sims[idx]),  # smaller = more relevant (mirrors Chroma)
        })
        if len(hits) >= k:
            break
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF retrieval test")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sector", choices=["tech", "banks"])
    args = parser.parse_args()

    hits = retrieve(args.query, k=args.k, sector=args.sector)
    print(f"\n[TF-IDF] Query: {args.query} (k={args.k}, sector={args.sector or 'all'})\n")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"--- #{i} | {m['ticker']} ({m['sector']}) | distance {h['distance']:.3f} ---")
        print(h["text"][:300].strip(), "...\n")


if __name__ == "__main__":
    main()
