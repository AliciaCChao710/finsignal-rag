"""Query the ChromaDB vector store and return top-k relevant chunks.

Connects to the persistent collection built by embed.py, turns the user's question
into a query embedding, runs a vector-similarity search, and returns the top-k most
relevant chunks (with metadata and distance). Optionally filters by sector
(tech / banks) to support a "tech vs banks" comparison.

CLI:
    python src/retrieve.py "What are the main risk factors?" --k 5 --sector banks
"""

import argparse
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "filings"
EMBED_MODEL = "all-MiniLM-L6-v2"  # must match embed.py, or query vectors won't line up


def get_collection() -> chromadb.Collection:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Vector store {CHROMA_DIR} not found. Run python src/embed.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def retrieve(query: str, k: int = 5, sector: str | None = None) -> list[dict]:
    """Return the top-k most relevant chunks; each has text / metadata / distance (smaller = closer)."""
    collection = get_collection()
    where = {"sector": sector} if sector else None
    res = collection.query(query_texts=[query], n_results=k, where=where)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="vector retrieval test")
    parser.add_argument("query", help="the query")
    parser.add_argument("--k", type=int, default=5, help="how many to return (default 5)")
    parser.add_argument("--sector", choices=["tech", "banks"], help="restrict to one sector")
    args = parser.parse_args()

    hits = retrieve(args.query, k=args.k, sector=args.sector)
    print(f"\nQuery: {args.query} (k={args.k}, sector={args.sector or 'all'})\n")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"--- #{i} | {m['ticker']} ({m['sector']}) | distance {h['distance']:.3f} ---")
        print(h["text"][:300].strip(), "...\n")


if __name__ == "__main__":
    main()
