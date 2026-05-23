"""
src/api/main.py
---------------
FastAPI backend for the Cognitive Distortion Detector.

Endpoints:
  GET  /          — serve the frontend (index.html)
  POST /analyse   — run inference + LIME + guidance on user text
  GET  /health    — quick liveness check

Usage (from project root):
    uvicorn src.api.main:app --reload --port 8000
    # then open http://localhost:8000

The DistilBERT model and LIME explainer are loaded once at startup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python src/api/main.py` without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

# Resolve frontend path relative to this file — works from any working directory
_FRONTEND = Path(__file__).parent.parent.parent / "frontend" / "index.html"

from src.explain import Explainer
from src.guidance import get_guidance
from src.config import DISTORTION_LABELS

# ── Constants ──────────────────────────────────────────────────────────────────
WORD_LIMIT   = 150
THRESHOLD    = 0.40    # label predicted positive if prob >= this
NUM_FEATURES = 6       # top LIME words per label
NUM_SAMPLES  = 300     # LIME perturbations — reduced for API latency

# Single-char tokens and subword artifacts to strip from LIME keyword output
# (DistilBERT tokenises "I'm" → ["I", "m"] — "m" is meaningless as a keyword)
_NOISE_TOKENS = {
    "m", "s", "t", "re", "ve", "ll", "d",       # subword artifacts from I'm, it's, etc.
    "and", "the", "a", "an", "of", "to", "in",  # stopwords that carry no distortion signal
    "is", "it", "be", "or", "at", "by", "but",
}

# ── Global model (loaded once at startup) ─────────────────────────────────────
_explainer: Explainer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _explainer
    print("Loading DistilBERT model …")
    _explainer = Explainer()
    print("Model ready.")
    yield
    _explainer = None
    print("Model released.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cognitive Distortion Detector API",
    description="Multi-label CBT distortion classifier with LIME explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tightened at deployment if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class AnalyseRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def check_word_limit(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty.")
        word_count = len(v.split())
        if word_count > WORD_LIMIT:
            raise ValueError(
                f"Input exceeds {WORD_LIMIT}-word limit "
                f"({word_count} words). Please shorten your text."
            )
        return v


class WordWeight(BaseModel):
    word: str
    weight: float


class GuidanceOut(BaseModel):
    explanation: str
    technique:   str
    challenge:   str
    reframe:     str
    lime_hint:   str


class DistortionResult(BaseModel):
    label:       str
    probability: float          # 0.0 – 1.0
    top_words:   list[WordWeight]
    guidance:    GuidanceOut


class AnalyseResponse(BaseModel):
    detected:    list[DistortionResult]   # only labels above threshold
    word_count:  int
    threshold:   float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Serve the single-page frontend."""
    return FileResponse(str(_FRONTEND), media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_explainer is not None,
    )


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest) -> AnalyseResponse:
    """
    Three-step pipeline:
      1. DistilBERT inference  → probabilities for all 10 labels
      2. LIME                  → top keywords per detected label
      3. Guidance lookup       → CBT technique + Socratic question + reframe
    """
    if _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    text = req.text
    word_count = len(text.split())

    # ── Step 1: Inference ──────────────────────────────────────────────────────
    probs = _explainer.predict_proba([text])[0]   # shape (10,)

    detected_indices = [i for i, p in enumerate(probs) if p >= THRESHOLD]

    if not detected_indices:
        # Return empty detected list — no distortions above threshold
        return AnalyseResponse(
            detected=[],
            word_count=word_count,
            threshold=THRESHOLD,
        )

    # ── Steps 2 & 3: LIME + Guidance per detected label ───────────────────────
    results: list[DistortionResult] = []

    for idx in detected_indices:
        label = DISTORTION_LABELS[idx]
        prob  = float(probs[idx])

        # LIME explanation
        lime_result = _explainer.explain(
            text,
            label_idx=idx,
            num_features=NUM_FEATURES,
            num_samples=NUM_SAMPLES,
        )
        top_words = [
            WordWeight(word=w, weight=round(float(wt), 4))
            for w, wt in lime_result["top_words"]
            if len(w) > 1 and w.lower() not in _NOISE_TOKENS
        ]

        # Guidance lookup
        g = get_guidance(label)
        guidance = GuidanceOut(
            explanation=g["explanation"],
            technique=g["technique"],
            challenge=g["challenge"],
            reframe=g["reframe"],
            lime_hint=g["lime_hint"],
        )

        results.append(DistortionResult(
            label=label,
            probability=round(prob, 4),
            top_words=top_words,
            guidance=guidance,
        ))

    # Sort by probability descending
    results.sort(key=lambda r: r.probability, reverse=True)

    return AnalyseResponse(
        detected=results,
        word_count=word_count,
        threshold=THRESHOLD,
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Local dev: http://localhost:8000  |  HuggingFace Spaces: port 7860 via Dockerfile
    uvicorn.run("