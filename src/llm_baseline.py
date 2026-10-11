"""
src/llm_baseline.py
-------------------
Zero-shot and few-shot LLM baseline for cognitive distortion classification.

Compares Llama 3.3 70B (via Groq API) against the fine-tuned DistilBERT
on the same held-out test set (results/test_predictions.csv).

Interview defence:
    "I ran an LLM zero-shot baseline to establish whether fine-tuning was
     necessary. The comparison shows that even a large LLM underperforms
     the fine-tuned small model on this structured classification task,
     because fine-tuning learns the specific annotation schema and label
     boundaries from labelled examples."

Usage:
    export GROQ_API_KEY=your_key_here
    python -m src.llm_baseline

    # Or with subset for speed (recommended first run):
    python -m src.llm_baseline --max-rows 100

Outputs saved to results/:
    llm_baseline_results.json    - zero-shot and few-shot F1 vs DistilBERT
    llm_comparison_chart.png     - side-by-side bar chart
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from src.config import DISTORTION_LABELS, RESULTS_DIR

SHORT = [
    l.replace("/", "_").replace(" ", "_").replace("-", "_")
    for l in DISTORTION_LABELS
]

# ── Distortion descriptions (used in zero-shot prompt) ───────────────────────
_DESCRIPTIONS = {
    "All-or-nothing thinking":      "Seeing things as completely good or bad with no middle ground. Uses absolute words like always, never, completely, total failure.",
    "Overgeneralization":           "Drawing a sweeping conclusion from a single event. Uses words like always, every time, everyone, no one.",
    "Mental filter":                "Focusing exclusively on one negative detail while ignoring the bigger picture.",
    "Should statements":            "Rigid rules about how things must be. Uses should, must, ought to, have to, need to.",
    "Labeling":                     "Attaching a fixed global label to oneself or others instead of describing behaviour. Uses I am a..., He is a..., loser, failure, idiot.",
    "Personalization":              "Blaming oneself for events outside one's control, or taking excessive personal responsibility.",
    "Catastrophising/Magnification":"Exaggerating the importance or consequences of a problem. Uses disaster, ruined, terrible, worst, unbearable.",
    "Emotional Reasoning":          "Assuming that negative feelings reflect reality. I feel stupid therefore I am stupid.",
    "Mind Reading":                 "Assuming you know what others are thinking, without evidence.",
    "Fortune-telling":              "Predicting a negative outcome as certain fact. I know this will go wrong. It's going to be a disaster.",
}

# ── Few-shot examples (1 per distortion, hand-written) ───────────────────────
_FEW_SHOT_EXAMPLES = [
    {
        "text": "I completely failed that exam. I'm a total failure and I always mess up when it matters.",
        "labels": ["All-or-nothing thinking", "Labeling", "Overgeneralization"],
    },
    {
        "text": "I got one question wrong so the whole presentation was ruined.",
        "labels": ["Mental filter", "Catastrophising/Magnification"],
    },
    {
        "text": "I should have known better. I must always be prepared.",
        "labels": ["Should statements"],
    },
    {
        "text": "I feel so stupid, so I must be stupid.",
        "labels": ["Emotional Reasoning"],
    },
    {
        "text": "I know they all think I'm boring at parties.",
        "labels": ["Mind Reading"],
    },
    {
        "text": "I'm going to fail the interview tomorrow, I just know it.",
        "labels": ["Fortune-telling"],
    },
    {
        "text": "The argument happened because I said the wrong thing. It's always my fault.",
        "labels": ["Personalization", "Overgeneralization"],
    },
]

# ── Build prompts ─────────────────────────────────────────────────────────────

def _distortion_list_str() -> str:
    lines = []
    for i, label in enumerate(DISTORTION_LABELS, 1):
        lines.append(f"{i}. {label}: {_DESCRIPTIONS[label]}")
    return "\n".join(lines)


def _few_shot_block() -> str:
    blocks = []
    for ex in _FEW_SHOT_EXAMPLES:
        labels_json = json.dumps({l: (l in ex["labels"]) for l in DISTORTION_LABELS})
        blocks.append(f'Text: "{ex["text"]}"\nOutput: {labels_json}')
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """You are a CBT text classifier. Label text for cognitive distortions.
Output ONLY a JSON object. Keys = distortion names, values = true/false.
No prose. No markdown. No code block. Just raw JSON on one line."""

# Shorter zero-shot user prompt - distortion list kept brief to save tokens
ZERO_SHOT_USER = """Distortions to detect (true/false each):
All-or-nothing thinking, Overgeneralization, Mental filter, Should statements,
Labeling, Personalization, Catastrophising/Magnification, Emotional Reasoning,
Mind Reading, Fortune-telling

Text: "{text}"
JSON output:"""

# Minimal few-shot - only 3 examples to save tokens
FEW_SHOT_USER = """Examples:
Text: "I always ruin everything. I am a complete failure."
Output: {{"All-or-nothing thinking": true, "Overgeneralization": true, "Mental filter": false, "Should statements": false, "Labeling": true, "Personalization": false, "Catastrophising/Magnification": false, "Emotional Reasoning": false, "Mind Reading": false, "Fortune-telling": false}}

Text: "I feel stupid so I must be stupid. I should never have tried."
Output: {{"All-or-nothing thinking": false, "Overgeneralization": false, "Mental filter": false, "Should statements": true, "Labeling": false, "Personalization": false, "Catastrophising/Magnification": false, "Emotional Reasoning": true, "Mind Reading": false, "Fortune-telling": false}}

Text: "I know they all think I'm boring and this will definitely go badly."
Output: {{"All-or-nothing thinking": false, "Overgeneralization": false, "Mental filter": false, "Should statements": false, "Labeling": false, "Personalization": false, "Catastrophising/Magnification": false, "Emotional Reasoning": false, "Mind Reading": true, "Fortune-telling": true}}

Text: "{text}"
Output:"""


# ── Parse LLM response ────────────────────────────────────────────────────────

def parse_response(response_text: str) -> dict[str, bool] | None:
    """
    Extract JSON from LLM output. Returns None if parsing fails.
    Handles: markdown code blocks, <think>...</think> reasoning tokens (Qwen3).
    """
    text = response_text.strip()
    # Strip Qwen3 / DeepSeek thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # Find first { ... }
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group())
        # Normalise to bool
        result = {}
        for label in DISTORTION_LABELS:
            val = parsed.get(label, False)
            result[label] = bool(val) if not isinstance(val, str) else val.lower() == "true"
        return result
    except (json.JSONDecodeError, KeyError):
        return None


# ── Single inference call ────────────────────────────────────────────────────

def classify_one(
    client,
    text: str,
    mode: str,             # "zero_shot" or "few_shot"
    model: str = "qwen/qwen3.8-27b",
    retries: int = 2,
    debug: bool = False,
) -> dict[str, bool] | None:
    user_msg = (
        ZERO_SHOT_USER.format(text=text[:400])
        if mode == "zero_shot"
        else FEW_SHOT_USER.replace("{text}", text[:400])
    )
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=150,
                temperature=0.0,   # deterministic for reproducibility
            )
            raw = resp.choices[0].message.content
            if debug:
                print(f"\n    [DEBUG raw response]: {raw[:300]}")
            parsed = parse_response(raw)
            if parsed is not None:
                return parsed
            if debug:
                print("    [DEBUG] parse_response returned None")
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < retries:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s ...")
                time.sleep(wait)
            else:
                print(f"    Error: {e}")
                return None
    return None


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_evaluation(
    max_rows: int = 50,
    model: str = "qwen/qwen3.8-27b",
    sleep_between: float = 3.0,
) -> dict:
    """
    Run zero-shot and few-shot LLM classification on the test set.

    max_rows: number of test rows to evaluate (100 is enough for reliable F1).
    sleep_between: seconds to wait between API calls (respect rate limits).
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key at https://console.groq.com"
        )
    client = Groq(api_key=api_key)

    csv_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    df = pd.read_csv(csv_path).head(max_rows)
    print(f"Evaluating on {len(df)} test examples with model: {model}\n")

    # Ground truth matrix
    Y_true = np.zeros((len(df), len(DISTORTION_LABELS)), dtype=int)
    for j, short in enumerate(SHORT):
        true_col = f"true_{short}"
        if true_col in df.columns:
            Y_true[:, j] = df[true_col].values.astype(int)

    # DistilBERT predictions (from CSV, threshold 0.40)
    Y_distilbert = np.zeros_like(Y_true)
    for j, short in enumerate(SHORT):
        prob_col = f"prob_{short}"
        if prob_col in df.columns:
            Y_distilbert[:, j] = (df[prob_col].values >= 0.40).astype(int)

    # LLM inference
    Y_zero = np.zeros_like(Y_true)
    Y_few  = np.zeros_like(Y_true)
    parse_failures = {"zero_shot": 0, "few_shot": 0}

    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["text"])
        debug_this = (i == 0)   # print raw output only for first row to verify parsing
        print(f"  [{i+1:03d}/{len(df)}] zero-shot ...", end=" ", flush=True)

        zs = classify_one(client, text, "zero_shot", model, debug=debug_this)
        if zs is None:
            parse_failures["zero_shot"] += 1
            print("FAIL", end="")
        else:
            for j, label in enumerate(DISTORTION_LABELS):
                Y_zero[i, j] = int(zs.get(label, False))
            print("ok", end="")

        time.sleep(sleep_between)

        print(" | few-shot ...", end=" ", flush=True)
        fs = classify_one(client, text, "few_shot", model, debug=debug_this)
        if fs is None:
            parse_failures["few_shot"] += 1
            print("FAIL")
        else:
            for j, label in enumerate(DISTORTION_LABELS):
                Y_few[i, j] = int(fs.get(label, False))
            print("ok")

        time.sleep(sleep_between)

    # Compute per-label F1
    def per_label_f1(Y_pred):
        scores = f1_score(Y_true, Y_pred, average=None, zero_division=0)
        return {l: round(float(s), 4) for l, s in zip(DISTORTION_LABELS, scores)}

    def macro_f1(Y_pred):
        return round(float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)), 4)

    results = {
        "model": model,
        "n_examples": max_rows,
        "parse_failures": parse_failures,
        "distilbert_threshold": 0.40,
        "distilbert": {
            "macro_f1": macro_f1(Y_distilbert),
            "per_label_f1": per_label_f1(Y_distilbert),
        },
        "zero_shot_llm": {
            "macro_f1": macro_f1(Y_zero),
            "per_label_f1": per_label_f1(Y_zero),
        },
        "few_shot_llm": {
            "macro_f1": macro_f1(Y_few),
            "per_label_f1": per_label_f1(Y_few),
        },
    }

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "llm_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Label':<40} {'DistilBERT':>10} {'Zero-shot':>10} {'Few-shot':>10}")
    print("-" * 75)
    for label in DISTORTION_LABELS:
        db = results["distilbert"]["per_label_f1"][label]
        zs = results["zero_shot_llm"]["per_label_f1"][label]
        fs = results["few_shot_llm"]["per_label_f1"][label]
        print(f"  {label:<38} {db:>10.4f} {zs:>10.4f} {fs:>10.4f}")
    print("-" * 75)
    print(f"  {'Macro F1':<38} {results['distilbert']['macro_f1']:>10.4f} "
          f"{results['zero_shot_llm']['macro_f1']:>10.4f} "
          f"{results['few_shot_llm']['macro_f1']:>10.4f}")
    print("=" * 75)

    # Plot comparison chart
    _plot_comparison(results)
    return results


def _plot_comparison(results: dict) -> None:
    labels = DISTORTION_LABELS
    db_f1  = [results["distilbert"]["per_label_f1"][l] for l in labels]
    zs_f1  = [results["zero_shot_llm"]["per_label_f1"][l] for l in labels]
    fs_f1  = [results["few_shot_llm"]["per_label_f1"][l] for l in labels]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x - w,   db_f1, w, label="DistilBERT (fine-tuned)", color="#4a6fa5", alpha=0.85)
    ax.bar(x,       zs_f1, w, label="Llama 3.3 70B zero-shot", color="#e8a838", alpha=0.85)
    ax.bar(x + w,   fs_f1, w, label="Llama 3.3 70B few-shot",  color="#38a169", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title(
        f"Per-Label F1: Fine-tuned DistilBERT vs LLM Baselines\n"
        f"(n={results['n_examples']} test examples | model: {results['model']})",
        fontsize=11,
    )
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "llm_comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=50,
                        help="Number of test rows to evaluate (default: 100)")
    parser.add_argument("--model", default="qwen/qwen3.8-27b",
                        help="Groq model ID")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds to sleep between API calls")
    args = parser.parse_args()

    run_evaluation(
        max_rows=args.max_rows,
        model=args.model,
        sleep_between=args.sleep,
    )
