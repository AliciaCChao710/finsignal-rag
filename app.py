"""Streamlit dashboard for the RAG pipeline.

Ask a question over the 10-K filings, optionally restrict to a sector, and see a
cited answer alongside the source passages it was grounded in.

Run:
    streamlit run app.py
"""

import hmac
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # make GEMINI_API_KEY available locally; Cloud Run injects it as an env var

# Make the modules in src/ importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from generate import generate_answer  # noqa: E402
from retrieve import get_collection  # noqa: E402
from retrieve_tfidf import retrieve as retrieve_tfidf  # noqa: E402  (sparse / TF-IDF baseline)
from evaluate import context_precision  # noqa: E402  (LLM-judge relevance score)

st.set_page_config(page_title="FinSignal RAG", page_icon="📊", layout="wide")


def check_password() -> bool:
    """Gate the app behind a shared password set via the APP_PASSWORD env var.

    If APP_PASSWORD is unset (e.g. local dev), the app is open. Otherwise the user
    must enter the matching password once per browser session.
    """
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True

    def _entered() -> None:
        st.session_state["auth_ok"] = hmac.compare_digest(
            st.session_state.get("pw", ""), expected
        )
        st.session_state.pop("pw", None)

    st.text_input("Password", type="password", key="pw", on_change=_entered)
    if st.session_state.get("auth_ok") is False:
        st.error("Incorrect password")
    return False


if not check_password():
    st.stop()


@st.cache_resource(show_spinner="Loading vector store and embedding model...")
def load_collection():
    """Load the ChromaDB collection once and reuse it across interactions."""
    return get_collection()


def search(collection, query: str, k: int, sector: str | None) -> list[dict]:
    """Query the cached collection and return hits (text / metadata / distance)."""
    where = {"sector": sector} if sector else None
    res = collection.query(query_texts=[query], n_results=k, where=where)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


st.title("📊 FinSignal — RAG over 10-K filings")
st.caption(
    "Retrieval-Augmented Generation over SEC 10-K annual reports "
    "(tech: GOOGL, MSFT, NVDA · banks: JPM, GS, BAC). "
    "Answers are grounded in retrieved passages and cite their sources."
)

with st.sidebar:
    st.header("Settings")
    sector_label = st.radio("Sector filter", ["All", "Tech", "Banks"], index=0)
    sector = {"All": None, "Tech": "tech", "Banks": "banks"}[sector_label]
    k = st.slider("Passages to retrieve (k)", min_value=3, max_value=10, value=6)
    method = st.radio("Retrieval method", ["Embedding (dense)", "TF-IDF (sparse)"], index=0)
    st.markdown(
        "**Sample questions**\n"
        "- Compare the main risks for tech vs banks\n"
        "- What does NVIDIA say about AI regulation?\n"
        "- How do banks describe credit risk?"
    )

query = st.text_input(
    "Your question",
    placeholder="e.g. Compare the main risks technology companies vs banks emphasise",
)

ask_col, cmp_col = st.columns([1, 2])
ask_clicked = ask_col.button("Ask", type="primary")
compare_clicked = cmp_col.button("Compare TF-IDF vs Embedding (scored)")

if ask_clicked and query.strip():
    with st.spinner("Retrieving relevant passages..."):
        if method.startswith("TF-IDF"):
            hits = retrieve_tfidf(query, k=k, sector=sector)
        else:
            collection = load_collection()
            hits = search(collection, query, k=k, sector=sector)

    with st.spinner("Generating answer with Gemini..."):
        response = generate_answer(query, hits)

    st.subheader("Answer")
    st.markdown(response)

    st.subheader(f"Sources ({len(hits)} passages · {method})")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        with st.expander(
            f"[{i}] {m['ticker']} ({m['sector']}) · filing {m['filing_date']} · "
            f"distance {h['distance']:.3f}"
        ):
            st.write(h["text"])

elif compare_clicked and query.strip():
    with st.spinner("Retrieving with both methods and scoring with the LLM judge..."):
        collection = load_collection()
        emb_hits = search(collection, query, k=k, sector=sector)
        cp_emb, _ = context_precision(query, [h["text"] for h in emb_hits])
        tfidf_hits = retrieve_tfidf(query, k=k, sector=sector)
        cp_tfidf, _ = context_precision(query, [h["text"] for h in tfidf_hits])

    st.subheader("Context precision — higher = more relevant retrieval")
    m1, m2 = st.columns(2)
    m1.metric("Embedding (dense · W3)", f"{cp_emb:.2f}")
    m2.metric("TF-IDF (sparse · W2)", f"{cp_tfidf:.2f}")
    st.caption(
        "Scored live by an LLM judge (non-deterministic): the numbers indicate "
        "direction, not a precise ranking, and vary slightly between runs."
    )

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown(f"**Embedding passages ({len(emb_hits)})**")
        for i, h in enumerate(emb_hits, 1):
            m = h["metadata"]
            with st.expander(f"[{i}] {m['ticker']} ({m['sector']}) · dist {h['distance']:.3f}"):
                st.write(h["text"])
    with pcol2:
        st.markdown(f"**TF-IDF passages ({len(tfidf_hits)})**")
        for i, h in enumerate(tfidf_hits, 1):
            m = h["metadata"]
            with st.expander(f"[{i}] {m['ticker']} ({m['sector']}) · dist {h['distance']:.3f}"):
                st.write(h["text"])

elif query == "":
    st.info("Enter a question above, then click **Ask** or **Compare**.")
