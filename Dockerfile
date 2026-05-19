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

# ── Install CPU-only PyTorch first ─────────────────────────────────────────────
# Avoids downloading 2GB+ CUDA wheels on a CPU-only Space.
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining dependencies ─────────────────────────────────────────────
# torch is already satisfied above; pip skips reinstalling it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# HuggingFace Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
