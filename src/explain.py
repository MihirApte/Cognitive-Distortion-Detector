"""
src/explain.py
--------------
LIME explainability wrapper for the fine-tuned DistilBERT model (Phase 1).

Provides:
  - Explainer class: loads model, runs LIME, saves HTML + CSV outputs
  - run(): batch-explain hand-picked examples from test_predictions.csv

Usage:
    python -m src.explain

Outputs saved to: results/lime_explanations/
  - {idx}_{label_short}.html   - LIME HTML plot per (example, label)
  - lime_word_importance.csv   - top-5 words per label aggregated across examples
"""

import os
import re
import json
import html
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from lime.lime_text import LimeTextExplainer

from src.config import (
    DISTORTION_LABELS, NUM_LABELS,
    DATA_PROC_DIR, MODELS_DIR, RESULTS_DIR,
    DISTILBERT_MODEL_DIR,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH   = DISTILBERT_MODEL_DIR   # models/distilbert_cognitive_distortion
LIME_DIR     = os.path.join(RESULTS_DIR, "lime_explanations")
MAX_LEN      = 128
THRESHOLD    = 0.40
NUM_SAMPLES  = 500   # LIME perturbation samples - 500 is fast & stable enough
NUM_FEATURES = 10    # top N words shown per explanation

# Short names - must match evaluate.py's column naming convention exactly
SHORT = [
    l.replace("/", "_").replace(" ", "_").replace("-", "_")
    for l in DISTORTION_LABELS
]


class Explainer:
    """
    Wraps DistilBERT for multi-label LIME explanations (Phase 1).

    Example
    -------
    >>> exp = Explainer()
    >>> result = exp.explain("I always fail at everything.", label_idx=0)
    >>> result["top_words"]   # list of (word, weight) tuples
    """

    def __init__(self, model_path: str = MODEL_PATH, threshold: float = THRESHOLD):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Train DistilBERT first by running notebooks/02_train_distilbert.ipynb on Colab,\n"
                "then place the saved model in models/distilbert_cognitive_distortion/."
            )
        self.threshold = threshold
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading DistilBERT model from: {model_path}")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model     = DistilBertForSequenceClassification.from_pretrained(
            model_path, num_labels=NUM_LABELS
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

        # LimeTextExplainer - class_names used for display only
        self.lime_exp = LimeTextExplainer(
            class_names=DISTORTION_LABELS,
            random_state=42,
        )

    # ── LIME predict function ──────────────────────────────────────────────────
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Returns sigmoid probabilities for all 10 labels.
        Shape: (len(texts), 10)
        LIME calls this repeatedly with perturbed/masked text.
        """
        all_probs = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)   # (N, 10)

    # ── Single explanation ─────────────────────────────────────────────────────
    def explain(
        self,
        text: str,
        label_idx: int,
        num_features: int = NUM_FEATURES,
        num_samples:  int = NUM_SAMPLES,
    ) -> dict:
        """
        Run LIME on `text` for a single label index.

        Returns
        -------
        dict with keys:
          text        : original text
          label       : distortion label name
          label_idx   : int
          prob        : model probability for this label
          prediction  : 0 or 1 based on threshold
          top_words   : list of (word, weight) sorted by |weight| desc
          exp_object  : raw lime Explanation object
        """
        exp = self.lime_exp.explain_instance(
            text,
            self.predict_proba,
            labels=[label_idx],
            num_features=num_features,
            num_samples=num_samples,
        )

        prob       = self.predict_proba([text])[0, label_idx]
        prediction = int(prob >= self.threshold)
        top_words  = sorted(
            exp.as_list(label=label_idx),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "text"      : text,
            "label"     : DISTORTION_LABELS[label_idx],
            "label_idx" : label_idx,
            "prob"      : float(prob),
            "prediction": prediction,
            "top_words" : top_words,
            "exp_object": exp,
        }

    # ── Save HTML ──────────────────────────────────────────────────────────────
    def save_html(self, result: dict, out_path: str) -> None:
        """Save a self-contained HTML explanation for one (text, label) pair."""
        exp   = result["exp_object"]
        label = result["label_idx"]

        # LIME's built-in HTML
        raw_html = exp.as_html(labels=[label])

        # Inject a title banner
        label_name = result["label"]
        prob_pct   = f"{result['prob']*100:.1f}%"
        pred_str   = "POSITIVE" if result["prediction"] else "negative"
        color      = "#d9534f" if result["prediction"] else "#5cb85c"

        banner = f"""
        <div style="font-family:Arial,sans-serif; padding:12px 20px;
                    background:#f8f9fa; border-bottom:2px solid #dee2e6; margin-bottom:10px;">
          <h2 style="margin:0 0 4px 0; color:#343a40;">
            Label: <span style="color:{color};">{html.escape(label_name)}</span>
          </h2>
          <p style="margin:0; color:#6c757d; font-size:14px;">
            Model probability: <strong>{prob_pct}</strong> &nbsp;|&nbsp;
            Prediction: <strong style="color:{color};">{pred_str}</strong>
            &nbsp;(threshold&nbsp;=&nbsp;{self.threshold})
          </p>
          <p style="margin:8px 0 0 0; font-size:13px; color:#495057;">
            <em>{html.escape(result['text'][:200])}{"..." if len(result["text"]) > 200 else ""}</em>
          </p>
        </div>
        """
        final_html = raw_html.replace("<body>", f"<body>{banner}", 1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_html)

    # ── Explain + save ─────────────────────────────────────────────────────────
    def explain_and_save(
        self,
        text: str,
        label_idx: int,
        prefix: str = "ex",
        out_dir: str = LIME_DIR,
        num_features: int = NUM_FEATURES,
        num_samples:  int = NUM_SAMPLES,
    ) -> dict:
        """Run LIME, save HTML, return result dict."""
        result   = self.explain(text, label_idx, num_features, num_samples)
        short    = SHORT[label_idx]
        out_path = os.path.join(out_dir, f"{prefix}_{short}.html")
        self.save_html(result, out_path)
        print(
            f"  [{label_idx:02d}] {DISTORTION_LABELS[label_idx]:<35} "
            f"prob={result['prob']:.3f}  pred={'POS' if result['prediction'] else 'neg'}  "
            f"→ {os.path.basename(out_path)}"
        )
        return result


# ── Batch runner ───────────────────────────────────────────────────────────────
def _pick_examples(pred_df: pd.DataFrame) -> list[dict]:
    """
    Pick ~10 interesting examples from test_predictions.csv:
      - 1 clear true-positive per label (highest prob, correct prediction)
      - plus 2 hard false-positive examples (high prob, wrong label)
      - plus 1 multi-label example (2+ labels predicted positive)
    Returns list of dicts: {text, label_idx, example_type, row_idx}
    """
    picked = []
    seen_rows = set()

    short_map = {
        re.sub(r"[^a-z0-9]+", "_", l.lower()).strip("_"): i
        for i, l in enumerate(DISTORTION_LABELS)
    }

    # 1 clear TP per label - highest prob where pred=1 and true=1
    for i, label in enumerate(DISTORTION_LABELS):
        short   = SHORT[i]
        prob_col = f"prob_{short}"
        pred_col = f"pred_{short}"
        true_col = f"true_{short}"

        if prob_col not in pred_df.columns:
            continue

        tp_mask = (pred_df[pred_col] == 1) & (pred_df[true_col] == 1)
        if tp_mask.sum() == 0:
            continue

        row = pred_df[tp_mask].nlargest(1, prob_col).iloc[0]
        row_idx = pred_df[tp_mask].nlargest(1, prob_col).index[0]
        if row_idx not in seen_rows:
            picked.append({
                "text"        : row["text"],
                "label_idx"   : i,
                "example_type": "true_positive",
                "row_idx"     : row_idx,
            })
            seen_rows.add(row_idx)

    # 2 hard FP - predicted positive but actually negative, highest prob
    fp_candidates = []
    for i, label in enumerate(DISTORTION_LABELS):
        short    = SHORT[i]
        prob_col = f"prob_{short}"
        pred_col = f"pred_{short}"
        true_col = f"true_{short}"
        if prob_col not in pred_df.columns:
            continue
        fp_mask = (pred_df[pred_col] == 1) & (pred_df[true_col] == 0)
        if fp_mask.sum() == 0:
            continue
        row     = pred_df[fp_mask].nlargest(1, prob_col).iloc[0]
        row_idx = pred_df[fp_mask].nlargest(1, prob_col).index[0]
        fp_candidates.append((row[prob_col], row_idx, i, row["text"]))

    fp_candidates.sort(reverse=True)
    for _, row_idx, i, text in fp_candidates[:2]:
        if row_idx not in seen_rows:
            picked.append({
                "text"        : text,
                "label_idx"   : i,
                "example_type": "false_positive",
                "row_idx"     : row_idx,
            })
            seen_rows.add(row_idx)

    # 1 multi-label example - highest number of predicted positives
    prob_cols = [f"prob_{s}" for s in SHORT if f"prob_{s}" in pred_df.columns]
    pred_cols = [f"pred_{s}" for s in SHORT if f"pred_{s}" in pred_df.columns]
    n_preds   = pred_df[pred_cols].sum(axis=1)
    ml_idx    = n_preds.idxmax()
    if ml_idx not in seen_rows and n_preds[ml_idx] > 1:
        # explain the label with highest prob in that row
        best_label = int(
            pred_df.loc[ml_idx, prob_cols].values.argmax()
        )
        picked.append({
            "text"        : pred_df.loc[ml_idx, "text"],
            "label_idx"   : best_label,
            "example_type": "multi_label",
            "row_idx"     : ml_idx,
        })

    return picked


def run():
    os.makedirs(LIME_DIR, exist_ok=True)

    # ── Load predictions CSV ───────────────────────────────────────────────────
    preds_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(
            f"Not found: {preds_path}\nRun python -m src.evaluate first."
        )
    pred_df = pd.read_csv(preds_path)
    print(f"Loaded predictions: {len(pred_df)} rows")

    # ── Load model ─────────────────────────────────────────────────────────────
    exp = Explainer()

    # ── Pick examples ──────────────────────────────────────────────────────────
    examples = _pick_examples(pred_df)
    print(f"\nExplaining {len(examples)} examples …\n")

    # ── Run LIME on each ───────────────────────────────────────────────────────
    all_results = []
    for ex in examples:
        prefix = f"{ex['example_type']}_{ex['row_idx']}"
        print(f"[{ex['example_type'].upper()}] row={ex['row_idx']}")
        result = exp.explain_and_save(
            text      = ex["text"],
            label_idx = ex["label_idx"],
            prefix    = prefix,
            out_dir   = LIME_DIR,
        )
        result["example_type"] = ex["example_type"]
        result["row_idx"]      = ex["row_idx"]
        all_results.append(result)
        print()

    # ── Aggregate top words per label ─────────────────────────────────────────
    #    Collect all (word, weight) pairs per label, sum weights across examples
    from collections import defaultdict
    word_scores = defaultdict(lambda: defaultdict(float))   # label → word → total_weight

    for r in all_results:
        label = r["label"]
        for word, weight in r["top_words"]:
            word_scores[label][word] += weight

    rows = []
    for label, scores in word_scores.items():
        sorted_words = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        for rank, (word, weight) in enumerate(sorted_words[:5], 1):
            rows.append({
                "label"      : label,
                "rank"       : rank,
                "word"       : word,
                "importance" : round(weight, 4),
                "direction"  : "positive" if weight > 0 else "negative",
            })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(LIME_DIR, "lime_word_importance.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=" * 60)
    print(f"LIME complete. {len(all_results)} explanations saved to:")
    print(f"  {LIME_DIR}")
    print(f"\nWord importance summary: {summary_path}")
    print("\nTop positive words per label:")
    for label in DISTORTION_LABELS:
        sub = summary_df[
            (summary_df["label"] == label) & (summary_df["direction"] == "positive")
        ].head(3)
        if sub.empty:
            continue
        words = ", ".join(sub["word"].tolist())
        print(f"  {label:<35} → {words}")
    print("=" * 60)


if __name__ == "__main__":
    run()
