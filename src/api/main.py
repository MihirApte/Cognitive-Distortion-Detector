"""
src/api/main.py
---------------
FastAPI backend for the Cognitive Distortion Detector.

Endpoints:
  GET  /          - serve the frontend (index.html)
  POST /analyse   - run inference + LIME + guidance + safety on user text
  GET  /health    - quick liveness check

Safety pipeline (runs before any inference):
  1. check_crisis()  - routes crisis text to a fixed safe message (never LLM)
  2. check_misuse()  - blocks therapy/advice extraction attempts
  3. MentalRoBERTa inference + LIME + optional LLM explanation

Usage (from project root):
    uvicorn src.api.main:app --reload --port 8000
    # then open http://localhost:8000

LLM explanations are enabled when GROQ_API_KEY env variable is set.
If absent, the LLM explanation field is omitted gracefully.
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

# Resolve frontend path relative to this file - works from any working directory
_FRONTEND = Path(__file__).parent.parent.parent / "frontend" / "index.html"

from src.explain import Explainer
from src.guidance import get_guidance
from src.config import DISTORTION_LABELS
from src.safety import check_crisis, check_misuse, SAFE_RESPONSE, MISUSE_RESPONSE, DISCLAIMER

# ── Constants ──────────────────────────────────────────────────────────────────
WORD_LIMIT   = 200
THRESHOLD    = 0.40    # label predicted positive if prob >= this
NUM_FEATURES = 6       # top LIME words per label
NUM_SAMPLES  = 300     # LIME perturbations - reduced for API latency

# Single-char tokens and subword artifacts to strip from LIME keyword output
# (DistilBERT tokenises "I'm" → ["I", "m"] - "m" is meaningless as a keyword)
_NOISE_TOKENS = {
    "m", "s", "t", "re", "ve", "ll", "d",       # subword artifacts from I'm, it's, etc.
    "and", "the", "a", "an", "of", "to", "in",  # stopwords that carry no distortion signal
    "is", "it", "be", "or", "at", "by", "but",
}

# ── Global model and optional LLM explainer ───────────────────────────────────
_explainer: Explainer | None = None
_llm_explainer = None   # loaded only if GROQ_API_KEY is set


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _explainer, _llm_explainer
    print("Loading MentalRoBERTa model ...")
    _explainer = Explainer()
    print("Model ready.")

    # LLM explainer is optional - only loaded when API key is present
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            from src.llm_explainer import LLMExplainer
            _llm_explainer = LLMExplainer(api_key=groq_key)
            print("LLM explainer ready (Groq).")
        except Exception as e:
            print(f"LLM explainer not loaded: {e}")
    else:
        print("GROQ_API_KEY not set - LLM explanations disabled.")

    yield
    _explainer = None
    _llm_explainer = None
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
    label:            str
    probability:      float          # 0.0 - 1.0
    top_words:        list[WordWeight]
    guidance:         GuidanceOut
    llm_explanation:  str | None = None   # grounded explanation from LLM (None if disabled)
    llm_verified:     bool | None = None  # True if no hallucinated citations


class AnalyseResponse(BaseModel):
    detected:         list[DistortionResult]   # only labels above threshold
    word_count:       int
    threshold:        float
    is_crisis:        bool = False
    crisis_message:   str | None = None
    is_misuse:        bool = False
    misuse_message:   str | None = None
    disclaimer:       str = DISCLAIMER


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
    Five-step pipeline:
      0. Safety checks         - crisis routing + misuse detection (before any inference)
      1. MentalRoBERTa inference - probabilities for all 10 labels
      2. LIME                  - top keywords per detected label
      3. Guidance lookup       - CBT technique + Socratic question + reframe
      4. LLM explanation       - optional grounded plain-English explanation (Groq)
    """
    if _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    text = req.text
    word_count = len(text.split())

    # ── Step 0a: Crisis check (always runs first, always returns fixed message) ──
    crisis_result = check_crisis(text)
    if crisis_result["is_crisis"]:
        return AnalyseResponse(
            detected=[],
            word_count=word_count,
            threshold=THRESHOLD,
            is_crisis=True,
            crisis_message=SAFE_RESPONSE["message"],
            disclaimer=DISCLAIMER,
        )

    # ── Step 0b: Misuse / jailbreak check ─────────────────────────────────────
    misuse_result = check_misuse(text)
    if misuse_result["is_misuse"]:
        return AnalyseResponse(
            detected=[],
            word_count=word_count,
            threshold=THRESHOLD,
            is_misuse=True,
            misuse_message=MISUSE_RESPONSE["message"],
            disclaimer=DISCLAIMER,
        )

    # ── Step 1: Inference ──────────────────────────────────────────────────────
    probs = _explainer.predict_proba([text])[0]   # shape (10,)

    detected_indices = [i for i, p in enumerate(probs) if p >= THRESHOLD]

    if not detected_indices:
        # Return empty detected list - no distortions above threshold
        return AnalyseResponse(
            detected=[],
            word_count=word_count,
            threshold=THRESHOLD,
        )

    # ── Steps 2, 3 & 4: LIME + Guidance + LLM explanation per detected label ───
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
        lime_word_list = [ww.word for ww in top_words]

        # Guidance lookup
        g = get_guidance(label)
        guidance = GuidanceOut(
            explanation=g["explanation"],
            technique=g["technique"],
            challenge=g["challenge"],
            reframe=g["reframe"],
            lime_hint=g["lime_hint"],
        )

        # LLM explanation (optional - skipped if no API key)
        llm_explanation = None
        llm_verified = None
        if _llm_explainer is not None and lime_word_list:
            try:
                llm_result = _llm_explainer.explain(
                    text=text,
                    label=label,
                    lime_words=lime_word_list,
                )
                llm_explanation = llm_result["explanation"]
                llm_verified    = llm_result["verified"]
            except Exception:
                pass   # LLM failure is non-fatal; classifier result still returned

        results.append(DistortionResult(
            label=label,
            probability=round(prob, 4),
            top_words=top_words,
            guidance=guidance,
            llm_explanation=llm_explanation,
            llm_verified=llm_verified,
        ))

    # Sort by probability descending
    results.sort(key=lambda r: r.probability, reverse=True)

    return AnalyseResponse(
        detected=results,
        word_count=word_count,
        threshold=THRESHOLD,
        disclaimer=DISCLAIMER,
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Local dev: http://localhost:8000  |  HuggingFace Spaces: port 7860 via Dockerfile
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
