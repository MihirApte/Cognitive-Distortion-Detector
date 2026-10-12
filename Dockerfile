# ── Dockerfile ─────────────────────────────────────────────────────────────────
# HuggingFace Spaces — single container serving both FastAPI backend
# and the static HTML frontend on port 7860.
#
# Build locally:
#   docker build -t cdd-app .
#   docker run -p 7860:7860 cdd-app
#
# Then open: http://localhost:7860
# ───────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install CPU-only PyTorch via direct wheel URL ──────────────────────────────
# Using --index-url breaks dependency resolution (pip can't reach PyPI for
# torch's own deps like typing-extensions). Installing from the wheel URL
# directly lets pip resolve all dependencies from PyPI as normal.
RUN pip install --no-cache-dir \
    "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-linux_x86_64.whl"

# ── Install remaining dependencies (excluding torch) ───────────────────────────
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt > requirements-filtered.txt && \
    pip install --no-cache-dir -r requirements-filtered.txt

# Build-time check: show installed versions and fail here (with the real error)
# if transformers cannot be imported, instead of crashing at app start.
RUN pip list 2>/dev/null | grep -iE "^(transformers|tokenizers|huggingface|torch|numpy|safetensors) " ; \
    python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; print('transformers import OK')"

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# HuggingFace Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
