"""
src/evaluate.py
---------------
Run full test-set evaluation on the fine-tuned DistilBERT model.
Saves per-example predictions + probabilities to results/test_predictions.csv
for use in the error analysis notebook.

Usage:
    python -m src.evaluate
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from src.config import (
    DISTORTION_LABELS, NUM_LABELS,
    DATA_PROC_DIR, MODELS_DIR, RESULTS_DIR,
    BATCH_SIZE, SEED
)

MODEL_PATH = os.path.join(MODELS_DIR, "distilbert_cognitive_distortion")
MAX_LEN    = 128
THRESHOLD  = 0.2   # matches Colab training


class CognitiveDistortionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts     = df["text"].astype(str).tolist()
        self.labels    = df[DISTORTION_LABELS].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_baseline():
    path = os.path.join(RESULTS_DIR, "baseline_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def run():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Extract the Colab zip and place the folder there."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    test_df = pd.read_csv(os.path.join(DATA_PROC_DIR, "test.csv"))
    print(f"Test set : {len(test_df)} rows")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading  : {MODEL_PATH}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model     = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=NUM_LABELS
    )
    model = model.to(device)
    model.eval()

    dataset = CognitiveDistortionDataset(test_df, tokenizer, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_probs  = []
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.sigmoid(outputs.logits)
            preds   = (probs >= THRESHOLD).int()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs  = np.vstack(all_probs)
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    per_label_f1  = f1_score(all_labels, all_preds, average=None,    zero_division=0)
    per_label_p   = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_label_r   = recall_score(all_labels, all_preds, average=None,    zero_division=0)
    micro_f1      = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    macro_f1      = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    h_loss        = hamming_loss(all_labels, all_preds)

    # ── Print comparison table ────────────────────────────────────────────────
    baseline    = load_baseline()
    base_labels = baseline.get("per_label_f1", {})

    print("\n" + "=" * 72)
    print("DISTILBERT vs BASELINE — TEST SET EVALUATION")
    print("=" * 72)
    print(f"  {'Label':<35} {'Base F1':>7} {'P':>6} {'R':>6} {'F1':>6} {'Delta':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for label, p, r, f1 in zip(DISTORTION_LABELS, per_label_p, per_label_r, per_label_f1):
        base  = base_labels.get(label, float("nan"))
        delta = f1 - base if not np.isnan(base) else float("nan")
        sign  = "+" if delta >= 0 else ""
        d_str = f"{sign}{delta:.4f}" if not np.isnan(delta) else "   N/A"
        print(f"  {label:<35} {base:>7.4f} {p:>6.4f} {r:>6.4f} {f1:>6.4f} {d_str:>7}")

    print("=" * 72)
    print(f"  {'Micro F1':<35} {baseline.get('micro_f1', float('nan')):>7.4f} {'':>6} {'':>6} {micro_f1:>6.4f}")
    print(f"  {'Macro F1':<35} {baseline.get('macro_f1', float('nan')):>7.4f} {'':>6} {'':>6} {macro_f1:>6.4f}")
    print(f"  {'Hamming Loss':<35} {baseline.get('hamming_loss', float('nan')):>7.4f} {'':>6} {'':>6} {h_loss:>6.4f}")
    print("=" * 72)

    # ── Save results JSON ─────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "model"        : "DistilBERT fine-tuned",
        "threshold"    : THRESHOLD,
        "micro_f1"     : round(float(micro_f1), 4),
        "macro_f1"     : round(float(macro_f1), 4),
        "hamming_loss" : round(float(h_loss),   4),
        "per_label_f1" : {l: round(float(s), 4) for l, s in zip(DISTORTION_LABELS, per_label_f1)},
        "per_label_precision": {l: round(float(s), 4) for l, s in zip(DISTORTION_LABELS, per_label_p)},
        "per_label_recall"   : {l: round(float(s), 4) for l, s in zip(DISTORTION_LABELS, per_label_r)},
    }
    results_path = os.path.join(RESULTS_DIR, "distilbert_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # ── Save per-example predictions CSV (for error analysis notebook) ─────────
    pred_df = test_df[["text"]].copy()
    for i, label in enumerate(DISTORTION_LABELS):
        short = label.replace("/", "_").replace(" ", "_").replace("-", "_")
        pred_df[f"true_{short}"]  = all_labels[:, i]
        pred_df[f"pred_{short}"]  = all_preds[:, i]
        pred_df[f"prob_{short}"]  = all_probs[:, i].round(4)

    preds_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    pred_df.to_csv(preds_path, index=False)
    print(f"Saved: {preds_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    run()
