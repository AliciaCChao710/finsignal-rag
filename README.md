# FinSignal — RAG over SEC 10-K Filings

A Retrieval-Augmented Generation (RAG) system that answers questions about US public
companies' annual reports (SEC **10-K** filings) and **compares risks and outlook across
companies and sectors** — with answers grounded in, and citing, the source passages.

Built for the SMM694 Applied NLP module. Entirely **free / open-source stack**, containerised
with Docker and deployed to **Google Cloud Run** (serverless).

**Live demo:** https://finsignal-rag-2ogmytm6xq-uc.a.run.app *(password-protected — available on request)*

---

## What it does

Ask a question like *"Compare the main risks technology companies versus banks emphasise"*,
optionally filter by sector, and get an answer that:

- is generated **only** from passages retrieved out of the filings (no hallucination), and
- **cites** each claim with a `[number]` pointing back to the source passage.

## Research question

> Can a RAG system compare risks and outlook across companies and sectors *accurately and with
> traceable sources*, using 10-K filings?

## Architecture

```
SEC EDGAR 10-Ks
      │  download.py        (sec-edgar-downloader)
      ▼
  raw filings ── ingest.py ──►  text chunks         (LangChain splitter, 1000/200)
                                    │  embed.py
                                    ▼
                             ChromaDB vector store   (all-MiniLM-L6-v2, local & free)
                                    │  retrieve.py   (top-k + sector filter)
                                    ▼
                          relevant passages ── generate.py ──►  cited answer
                                                  (Google Gemini 2.5 Flash, free tier)
      app.py (Streamlit UI + password gate)   ·   evaluate.py (faithfulness + context precision)
```

## Data

Six companies — three technology vs three banks — most recent 10-K each (≈ FY2025, 5,188 chunks):

| Company | Sector | Filing period (end) |
|---|---|---|
| GOOGL, MSFT, NVDA | tech | 2025-12-31 / 2025-06-30 / 2026-01-25 |
| JPM, GS, BAC | banks | 2025-12-31 |

> Fiscal year-ends differ across companies, so coverage windows differ by up to ~7 months — a
> disclosed limitation (the comparison is qualitative, not financial-figure based).

## Tech stack

| Stage | Tool |
|---|---|
| Acquisition | `sec-edgar-downloader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` (local, free) |
| Vector store | ChromaDB (persistent) |
| Generation | Google Gemini `gemini-2.5-flash` (free tier) |
| UI | Streamlit |
| Deployment | Docker + Google Cloud Run |

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
                            # and SEC_USER_NAME / SEC_USER_EMAIL

python src/download.py      # download 10-K filings → data/raw/
python src/ingest.py        # parse → chunks → data/processed/chunks.json
python src/embed.py         # embed → chroma_db/
streamlit run app.py        # launch the UI

python evaluate.py          # run the evaluation
```

CLI without the UI:

```bash
python src/retrieve.py "What are the main risk factors?" --k 5 --sector banks
python src/generate.py "Compare risks for tech vs banks"
```

## Evaluation

Two **label-free** metrics (no ground-truth answers needed), scored with an LLM judge:

- **Faithfulness** — is every claim in the answer supported by the retrieved passages?
- **Context precision** — are the retrieved passages actually relevant (rank-aware)?

| Question | Faithfulness | Context precision |
|---|:---:|:---:|
| NVIDIA AI risks (tech) | 1.00 | 0.00 |
| Bank credit risk (banks) | 1.00 | 0.70 |
| Tech vs banks comparison (all) | 1.00 | 1.00 |
| **Average** | **1.00** | **0.57** |

**Key finding:** generation is reliable (faithfulness 1.00 — the system even admits when the
retrieved passages don't contain the answer, instead of fabricating). The bottleneck is
**retrieval**: context precision is volatile and sensitive to query phrasing, and section-header
chunks (e.g. *"Item 1A. Risk Factors"*) can be retrieved instead of substantive text.

## Limitations & future work

- **Retrieval quality** is the main lever: drop header/short chunks, add re-ranking, query
  rewriting, or hybrid (BM25 + vector) search.
- Single fiscal year per company → no year-over-year trend analysis.
- Differing fiscal year-ends across companies (≈7-month spread).
- Small embedding model + free-tier LLM rate limits.

## Deployment

Containerised (`Dockerfile`, CPU-only torch to keep the image lean) and deployed to Google
Cloud Run. The prebuilt vector store (`chroma_db/`) is baked into the image (read-only). Access
is gated by a shared password (`APP_PASSWORD` env var); the Gemini key is passed as an env var
(never committed). Cost controls: scale-to-zero, `--max-instances`, and a billing budget alert.
