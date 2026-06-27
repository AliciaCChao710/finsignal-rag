"""Topic modelling of the 10-K filings with Gensim LDA (semantic / topic
modelling — a module syllabus topic).

Discovers latent topics across all filings, then compares how strongly each
sector (tech vs banks) loads on each topic. Fully local and free — no API.

Run:
    python src/topics.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"
OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "topics_results.json"
NUM_TOPICS = 6
PASSES = 5

SECTOR = {
    "GOOGL": "tech", "MSFT": "tech", "NVDA": "tech",
    "JPM": "banks", "GS": "banks", "BAC": "banks",
}

# 10-K boilerplate that otherwise dominates every topic and hides the signal
DOMAIN_STOPWORDS = {
    "company", "companies", "may", "also", "including", "include", "fiscal",
    "year", "years", "million", "billion", "results", "financial", "business",
    "could", "would", "table", "contents", "item", "inc", "llc", "december",
    "june", "january", "ended", "period", "report", "form", "based", "related",
    "certain", "additional", "result", "amounts", "products", "services",
    "operations", "net", "total", "value", "include", "see", "note", "common",
}
STOP = STOPWORDS.union(DOMAIN_STOPWORDS)


def tokenise(text: str) -> list[str]:
    return [t for t in simple_preprocess(text, deacc=True) if t not in STOP and len(t) > 3]


def main() -> None:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    log.info("Tokenising %d chunks...", len(chunks))
    docs = [tokenise(c["text"]) for c in chunks]
    sectors = [SECTOR.get(c["ticker"], "unknown") for c in chunks]

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  # drop ultra-rare / ultra-common
    corpus = [dictionary.doc2bow(d) for d in docs]

    log.info("Training LDA (%d topics, %d passes)...", NUM_TOPICS, PASSES)
    lda = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS,
        passes=PASSES, random_state=42,
    )

    # --- top words per topic ---
    topics = {}
    print("\n===== Discovered topics (top words) =====")
    for tid in range(NUM_TOPICS):
        words = [w for w, _ in lda.show_topic(tid, topn=10)]
        topics[tid] = words
        print(f"Topic {tid}: {', '.join(words)}")

    # --- per-sector topic loading ---
    sector_load = {"tech": defaultdict(float), "banks": defaultdict(float)}
    sector_docs = {"tech": 0, "banks": 0}
    for bow, sec in zip(corpus, sectors):
        if sec not in sector_load:
            continue
        sector_docs[sec] += 1
        for tid, prob in lda.get_document_topics(bow, minimum_probability=0.0):
            sector_load[sec][tid] += prob

    print("\n===== Topic emphasis by sector (share of topic mass) =====")
    sector_share = {}
    for sec in ("tech", "banks"):
        total = sum(sector_load[sec].values()) or 1.0
        share = {tid: float(sector_load[sec][tid] / total) for tid in range(NUM_TOPICS)}
        sector_share[sec] = share
        top = sorted(share.items(), key=lambda x: -x[1])[:3]
        print(f"\n{sec.upper()} (n={sector_docs[sec]}) — strongest topics:")
        for tid, frac in top:
            print(f"  Topic {tid} ({frac:.0%}): {', '.join(topics[tid][:6])}")

    OUT_FILE.write_text(json.dumps(
        {"topics": topics, "sector_share": sector_share}, ensure_ascii=False, indent=2
    ), encoding="utf-8")
    log.info("Saved topics + sector loadings to %s", OUT_FILE)


if __name__ == "__main__":
    main()
