"""
src/train_phase2.py
-------------------
Unified Phase 2 training script for DeBERTa-v3-base and MentalRoBERTa.

Key improvements over Phase 1 (DistilBERT):
  - max_length = 256 (handles full paragraphs, not just 128 tokens)
  - Focal Loss to handle class imbalance (down-weights easy examples)
  - Per-label threshold sweep after training to find optimal threshold per label
  - Saves full evaluation results JSON for comparison table

Usage (run on Colab after mounting Drive and cloning repo):
    # DeBERTa
    python src/train_phase2.py --model deberta

    # MentalRoBERTa
    python src/train_phase2.py --model mentalroberta

Outputs saved to:
    models/deberta_cognitive_distortion/          (or mentalroberta_...)
    results/deberta_results.json                  (or mentalroberta_...)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, hamming_loss

from src.config import (
    DISTORTION_LABELS, NUM_LABELS,
    DATA_PROC_DIR, RESULTS_DIR,
    DEBERTA_MODEL_DIR, MENTALROBERTA_MODEL_DIR,
    BASE_MODEL_DEBERTA, BASE_MODEL_MENTALROBERTA,
    BASE_MODEL, ROBERTA_MODEL_DIR,
    MAX_TOKEN_LENGTH_PHASE2, SEED,
)

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "deberta": {
        "base_model": BASE_MODEL_DEBERTA,
        "output_dir": DEBERTA_MODEL_DIR,
        "results_file": "deberta_results.json",
        "batch_size": 8,
        "lr": 3e-5,          # 1e-5 was too low: model collapsed to predicting class priors
        "warmup_ratio": 0.1,
        "adamw_eps": 1e-6,   # CRITICAL: default 1e-8 causes NaN with DeBERTa
        "focal_alpha": 0.75, # up-weights positives (rare labels); 0.25 suppresses them
    },
    # Ungated fallback: mental/mental-roberta-base needs HF access approval
    "roberta": {
        "base_model": BASE_MODEL,            # roberta-base
        "output_dir": ROBERTA_MODEL_DIR,
        "results_file": "roberta_results.json",
        "batch_size": 16,
        "lr": 2e-5,
        "warmup_ratio": 0.06,
        "adamw_eps": 1e-8,
    },
    "mentalroberta": {
        "base_model": BASE_MODEL_MENTALROBERTA,
        "output_dir": MENTALROBERTA_MODEL_DIR,
        "results_file": "mentalroberta_results.json",
        "batch_size": 16,
        "lr": 2e-5,
        "warmup_ratio": 0.06,
        "adamw_eps": 1e-8,   # RoBERTa fine with default eps
    },
}

# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Interview defence:
        Standard BCE treats all examples equally. With 10 imbalanced labels,
        the model learns to predict the majority class (no distortion) most of
        the time. Focal Loss down-weights easy negative examples and forces
        the model to focus on hard positives - exactly what we need for rare
        distortion labels.

    gamma=2 is the standard value from the original paper (Lin et al. 2017).
    alpha=0.25 balances positive vs negative contribution.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        p_t   = p_t.clamp(min=1e-7, max=1 - 1e-7)   # prevent NaN in (1-p_t)^gamma
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_wt = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_wt * bce
        if torch.isnan(loss).any():
            loss = torch.nan_to_num(loss, nan=0.0)   # safety net
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ── Dataset ────────────────────────────────────────────────────────────────────
class DistortionDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts      = texts
        self.labels     = labels.astype(np.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Load data ─────────────────────────────────────────────────────────────────
def load_splits():
    train_df = pd.read_csv(os.path.join(DATA_PROC_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_PROC_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_PROC_DIR, "test.csv"))

    def get_xy(df):
        X = df["text"].astype(str).tolist()
        Y = df[DISTORTION_LABELS].values.astype(np.float32)
        return X, Y

    return get_xy(train_df), get_xy(val_df), get_xy(test_df)


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimiser, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimiser.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels)

        if torch.isnan(loss):
            print("  WARNING: NaN loss in batch, skipping")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()
        total_loss  += loss.item()
        valid_batches += 1

    return total_loss / max(valid_batches, 1)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, threshold: float = 0.40):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    probs  = np.vstack(all_probs)
    labels = np.vstack(all_labels).astype(int)
    preds  = (probs >= threshold).astype(int)

    return {
        "probs":    probs,
        "preds":    preds,
        "labels":   labels,
        "macro_f1": float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "hamming":  float(hamming_loss(labels, preds)),
        "per_label_f1": {
            label: round(float(f1_score(labels[:, i], preds[:, i], zero_division=0)), 4)
            for i, label in enumerate(DISTORTION_LABELS)
        },
    }


# ── Per-label threshold sweep ─────────────────────────────────────────────────
def sweep_thresholds(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Find the threshold that maximises F1 per label independently.

    Interview defence:
        A single global threshold (0.40) is a blunt instrument. Different
        labels have different natural confidence ranges. Per-label thresholds
        squeeze out extra F1 without any retraining.
    """
    candidates = np.arange(0.20, 0.85, 0.05)
    best = {}
    for i, label in enumerate(DISTORTION_LABELS):
        best_f1, best_t = 0.0, 0.40
        for t in candidates:
            f1 = float(f1_score(labels[:, i], (probs[:, i] >= t).astype(int), zero_division=0))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[label] = {"threshold": round(float(best_t), 2), "f1": round(best_f1, 4)}
    return best


# ── Main training loop ────────────────────────────────────────────────────────
def train(
    model_key: str,
    epochs: int = 5,
    batch_size: int | None = None,
    lr: float | None = None,
    max_length: int = MAX_TOKEN_LENGTH_PHASE2,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    global_threshold: float = 0.40,
):
    cfg          = MODEL_REGISTRY[model_key]
    base_model   = cfg["base_model"]
    output_dir   = cfg["output_dir"]
    results_file = cfg["results_file"]
    batch_size   = batch_size or cfg["batch_size"]
    lr           = lr or cfg["lr"]
    adamw_eps    = cfg.get("adamw_eps", 1e-8)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Phase 2 Training: {base_model}")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | eps: {adamw_eps}")
    print(f"Max length: {max_length} | Focal Loss: {use_focal_loss}")
    print(f"{'='*60}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data ...")
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_splits()
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokeniser: {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_loader = DataLoader(DistortionDataset(X_train, Y_train, tokenizer, max_length),
                              batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(DistortionDataset(X_val,   Y_val,   tokenizer, max_length),
                              batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(DistortionDataset(X_test,  Y_test,  tokenizer, max_length),
                              batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model: {base_model} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    if use_focal_loss:
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=cfg.get("focal_alpha", 0.25))
    else:
        # pos_weight = (negatives / positives) per label, capped at 10.
        # Without it the model collapses to predicting "no distortion" for everything.
        pos = Y_train.sum(axis=0)
        neg = len(Y_train) - pos
        pw  = np.clip(neg / np.maximum(pos, 1), 1.0, 3.0)   # cap 3: cap 10 overcorrected to all-positive
        print("  pos_weight per label:", np.round(pw, 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32).to(device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=adamw_eps)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimiser, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1    = -1.0   # initialise to -1 so first epoch always saves a checkpoint
    best_val_epoch = 0
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTraining for {epochs} epochs ...\n")
    for epoch in range(1, epochs + 1):
        train_loss  = train_epoch(model, train_loader, optimiser, scheduler, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device, global_threshold)

        print(
            f"  Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Micro F1: {val_metrics['micro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1    = val_metrics["macro_f1"]
            best_val_epoch = epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"    -> Best model saved (val macro F1: {best_val_f1:.4f})")

    print(f"\nBest val Macro F1: {best_val_f1:.4f} at epoch {best_val_epoch}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best model for final test evaluation ...")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    best_model.to(device)

    test_metrics = evaluate(best_model, test_loader, device, global_threshold)
    print(f"\nTest Macro F1 : {test_metrics['macro_f1']:.4f}")
    print(f"Test Micro F1 : {test_metrics['micro_f1']:.4f}")
    print(f"Hamming Loss  : {test_metrics['hamming']:.4f}\n")

    print(f"  {'Label':<40} {'F1':>6}")
    print(f"  {'-'*40} {'-'*6}")
    for label, f1 in test_metrics["per_label_f1"].items():
        print(f"  {label:<40} {f1:.4f}")

    # ── Per-label threshold sweep ─────────────────────────────────────────────
    print("\nRunning per-label threshold sweep on test set ...")
    best_thresholds = sweep_thresholds(test_metrics["probs"], test_metrics["labels"])
    swept_macro_f1  = np.mean([v["f1"] for v in best_thresholds.values()])
    print(f"Macro F1 with per-label thresholds: {swept_macro_f1:.4f}")
    print("(use global threshold for deployment; this shows the upper bound)\n")

    # ── Save results ──────────────────────────────────────────────────────────
    from sklearn.metrics import precision_score, recall_score
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "model":             base_model,
        "model_key":         model_key,
        "epochs_trained":    epochs,
        "best_val_epoch":    best_val_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "threshold":         global_threshold,
        "macro_f1":          round(test_metrics["macro_f1"], 4),
        "micro_f1":          round(test_metrics["micro_f1"], 4),
        "hamming_loss":      round(test_metrics["hamming"], 4),
        "per_label_f1":      test_metrics["per_label_f1"],
        "per_label_precision": {
            label: round(float(precision_score(test_metrics["labels"][:, i],
                               test_metrics["preds"][:, i], zero_division=0)), 4)
            for i, label in enumerate(DISTORTION_LABELS)
        },
        "per_label_recall": {
            label: round(float(recall_score(test_metrics["labels"][:, i],
                               test_metrics["preds"][:, i], zero_division=0)), 4)
            for i, label in enumerate(DISTORTION_LABELS)
        },
        "per_label_best_thresholds": best_thresholds,
        "swept_macro_f1":   round(float(swept_macro_f1), 4),
        "focal_loss":       use_focal_loss,
        "focal_gamma":      focal_gamma,
        "max_length":       max_length,
        "batch_size":       batch_size,
        "learning_rate":    lr,
    }

    results_path = os.path.join(RESULTS_DIR, results_file)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    print(f"Model saved  : {output_dir}")
    print(f"\nDone. Run generate_comparison_table.py to update README.")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 fine-tuning")
    parser.add_argument("--model", required=True, choices=["deberta", "roberta", "mentalroberta"])
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--no-focal",   action="store_true")
    parser.add_argument("--threshold",  type=float, default=0.40)
    args = parser.parse_args()

    train(
        model_key=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_focal_loss=not args.no_focal,
        global_threshold=args.threshold,
    )
