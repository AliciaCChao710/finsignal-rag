# syntax=docker/dockerfile:1
# Container image for deploying the FinSignal RAG Streamlit app to Google Cloud Run.
FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch FIRST (the default torch wheel bundles ~6GB of NVIDIA
# CUDA libraries we don't need — Cloud Run is CPU-only). This keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the runtime deps (torch is already satisfied above)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Pre-download the embedding model into the image so the container does not
# fetch it from Hugging Face on cold start (faster, works without HF egress).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the app code and the prebuilt vector store (chroma_db/ is read-only at runtime).
# data/ is excluded via .dockerignore (the 227MB raw filings are not needed here).
COPY . .

# Cloud Run sends traffic to $PORT (default 8080); Streamlit must listen there.
ENV PORT=8080
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
