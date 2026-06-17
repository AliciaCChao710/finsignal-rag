# SMM694 RAG Prototype

Retrieval-Augmented Generation pipeline for SMM694 Applied NLP.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the fields
```

## Usage

1. Download raw documents from SEC to `data/raw/`


2. Ingest and process to chunks
   ```bash
   python src/ingest.py
   ```

3  TODO: Embed
   ```bash
   python src/embed.py
   ```

4. TODO: Launch the Streamlit app + query
   ```bash
   streamlit run app.py
   ```
5. TODO: Run RAGAS evaluation:
   ```bash
   python evaluate.py
   ```

## Project structure

```
smm694_rag_prototype/
├── data/
│   ├── raw/          # source documents (git-ignored)
│   └── processed/    # chunked JSON (git-ignored)
├── src/
│   ├── ingest.py     # parse documents → chunks
│   ├── embed.py      # embed chunks → vector store
│   ├── retrieve.py   # query → top-k chunks
│   ├── generate.py   # chunks + prompt → LLM answer
│   └── utils.py      # shared helpers
├── tests/
├── app.py            # Streamlit dashboard
├── evaluate.py       # RAGAS evaluation
└── requirements.txt
```
