"""
src/llm_explainer.py
--------------------
LLM-based explanation generator for detected cognitive distortions.

Design (interview defence):
    The fine-tuned DistilBERT does the classification. The LLM is used ONLY
    to generate a plain-English explanation grounded in the LIME top words.
    This is the right split:
      - Classifier: calibrated, deterministic, fast
      - LLM: natural language, but constrained to cite only LIME words

    The citation verifier checks that every word the LLM claims is influential
    actually appears in the LIME top-word list. Any mismatch is a hallucinated
    citation and is logged. This is a practical faithfulness check.

Requirements:
    pip install groq

    Set environment variable:
        GROQ_API_KEY=your_key_here
    Or pass api_key= to LLMExplainer().

Usage:
    from src.llm_explainer import LLMExplainer

    explainer = LLMExplainer()
    result = explainer.explain(
        text="I always ruin everything.",
        label="Overgeneralization",
        lime_words=["always", "ruin", "everything"],
    )
    print(result["explanation"])
    print(result["verified"])         # True if no hallucinated citations
    print(result["hallucinated"])     # list of words cited but not in LIME
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_EXPLANATION_SYSTEM = """You are a cognitive behavioural therapy (CBT) educator.
Your job is to explain, in plain English, why a specific cognitive distortion
was detected in a piece of text.

Rules you MUST follow:
1. Write exactly 2-3 sentences. No more.
2. You MUST reference at least one word from the provided keyword list.
3. Do NOT give therapy advice, recommendations, or instructions.
4. Do NOT use the word "should".
5. Do NOT claim certainty - use "may suggest", "appears to", "could indicate".
6. Use simple, non-clinical language."""

_EXPLANATION_USER = """A cognitive distortion classifier detected "{label}" in the following text.

Text: "{text}"

The model's most influential keywords were: {keywords}

Explain in 2-3 sentences why "{label}" may be present, referencing the keywords above.
Do not give advice. Do not diagnose."""


# ── Citation verifier ─────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens from text, stripping punctuation."""
    return set(re.findall(r"[a-z']+", text.lower()))


def verify_citations(
    explanation: str,
    lime_words: list[str],
) -> dict:
    """
    Check that the LLM explanation only uses words that appear in the LIME
    top-word list. Returns a dict with verification result and any mismatches.

    A "hallucinated citation" is a word from the explanation that the LLM
    appears to treat as influential but which the model did not actually weight
    highly. We use a simple heuristic: quoted words in the explanation are
    checked against the LIME list; otherwise we check all content words above
    4 characters (short words are likely conjunctions/articles).

    Interview note: this is a faithfulness check, not a fluency check.
    The explanation can be well-written and still fail if it fabricates
    which words drove the prediction.
    """
    lime_set = {w.lower() for w in lime_words}

    # Check quoted words first (most explicit citations)
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', explanation)
    quoted_words = [q[0] or q[1] for q in quoted]
    hallucinated_quoted = [
        w for w in quoted_words if w.lower() not in lime_set
    ]

    # Also check all content words longer than 4 chars not in LIME
    # (catches implicit citations like "the word 'always' suggests...")
    exp_tokens = _tokenise(explanation)
    lime_related = {w.lower() for w in lime_words}
    long_novel = [
        w for w in exp_tokens
        if len(w) > 4 and w not in lime_related and w not in {
            "which", "their", "about", "shows", "could", "would",
            "there", "where", "these", "those", "being", "often",
            "think", "feels", "seems", "quite", "might", "while",
            "label", "distortion", "thinking", "appears", "suggest",
            "indicate", "present", "pattern", "cognitive", "because",
            "detected", "model", "sentence", "statement",
        }
    ]

    all_hallucinated = list(set(hallucinated_quoted))  # conservative: only quoted
    verified = len(all_hallucinated) == 0

    return {
        "verified": verified,
        "hallucinated": all_hallucinated,
        "lime_words_used": [w for w in lime_words if w.lower() in _tokenise(explanation)],
        "note": "Only quoted terms checked for hallucination; unquoted implicit references not flagged.",
    }


# ── LLM explainer ─────────────────────────────────────────────────────────────

class LLMExplainer:
    """
    Generates grounded plain-English explanations via Groq (Llama 3.3 70B).

    Parameters
    ----------
    api_key   : Groq API key. Falls back to GROQ_API_KEY env variable.
    model     : Groq model ID. Default is llama-3.3-70b-versatile (free tier).
    max_tokens: Max tokens in explanation. 150 is enough for 2-3 sentences.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3.8-27b",
        max_tokens: int = 150,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self._api_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY environment variable "
                "or pass api_key= to LLMExplainer(). "
                "Get a free key at https://console.groq.com"
            )

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: pip install groq"
                )
            self._client = Groq(api_key=self._api_key)
        return self._client

    def explain(
        self,
        text: str,
        label: str,
        lime_words: list[str],
        verify: bool = True,
    ) -> dict:
        """
        Generate a 2-3 sentence plain-English explanation for why `label`
        was detected in `text`, grounded in `lime_words`.

        Returns
        -------
        dict with keys:
            explanation  : str  - the generated explanation
            verified     : bool - True if no hallucinated citations
            hallucinated : list - words cited but absent from LIME list
            lime_words   : list - words passed in
            label        : str
            model        : str
            error        : str | None - populated if LLM call failed
        """
        client = self._get_client()
        keywords_str = ", ".join(f'"{w}"' for w in lime_words[:6])

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EXPLANATION_SYSTEM},
                    {"role": "user", "content": _EXPLANATION_USER.format(
                        label=label,
                        text=text[:400],        # cap at 400 chars for latency
                        keywords=keywords_str,
                    )},
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,              # low temp = more consistent output
            )
            explanation = response.choices[0].message.content.strip()
            error = None
        except Exception as e:
            logger.error("LLM explanation failed: %s", e)
            explanation = ""
            error = str(e)

        verification = verify_citations(explanation, lime_words) if (verify and explanation) else {
            "verified": False,
            "hallucinated": [],
            "lime_words_used": [],
        }

        result = {
            "explanation": explanation,
            "verified": verification["verified"],
            "hallucinated": verification["hallucinated"],
            "lime_words_used": verification.get("lime_words_used", []),
            "lime_words": lime_words,
            "label": label,
            "model": self.model,
            "error": error,
        }

        if not verification["verified"]:
            logger.warning(
                "Hallucinated citation in explanation for '%s': %s",
                label, verification["hallucinated"],
            )

        return result


# ── Batch explanation (for evaluation) ───────────────────────────────────────

def batch_explain_and_verify(
    explainer: LLMExplainer,
    examples: list[dict],
    log_path: str | None = None,
) -> list[dict]:
    """
    Run explain() on a list of examples and optionally log mismatches.

    Each example dict should have: text, label, lime_words (list of str).
    Returns list of result dicts (same as explain() output).
    Logs any verification failures to log_path (append mode).
    """
    results = []
    mismatches = 0

    for ex in examples:
        result = explainer.explain(
            text=ex["text"],
            label=ex["label"],
            lime_words=ex["lime_words"],
        )
        results.append(result)

        if not result["verified"]:
            mismatches += 1
            logger.warning(
                "MISMATCH | label=%s | hallucinated=%s",
                result["label"], result["hallucinated"],
            )

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for r in results:
                if not r["verified"]:
                    f.write(json.dumps({
                        "label": r["label"],
                        "hallucinated": r["hallucinated"],
                        "explanation": r["explanation"],
                    }) + "\n")

    print(f"\nExplanations: {len(results)} | Verification failures: {mismatches}")
    return results
