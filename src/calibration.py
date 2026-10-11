"""
src/calibration.py
------------------
Calibration analysis for the DistilBERT multi-label classifier.

Calibration answers the question: "When the model says 70% confident, is it
right 70% of the time?" A well-calibrated model earns trust - its probability
output is a reliable guide to prediction quality.

Metric - Expected Calibration Error (ECE):
    ECE = sum over bins of (|samples in bin| / |total samples|) * |accuracy - confidence|
    Lower is better. A perfect calibration gives ECE = 0.

Interview defence:
    "I chose ECE because it's the standard reliability metric for classifiers
     deployed in sensitive domains. It quantifies the gap between stated
     confidence and actual accuracy, which matters when a user might act on
     a 90% prediction differently than a 55% one."

Usage:
    python -m src.calibration
    # Reads results/test_predictions.csv
    # Saves results/calibration_curves.png and results/calibration_report.json
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DISTORTION_LABELS, RESULTS_DIR

# Column name normalisation - test_predictions.csv uses underscores
SHORT = [
    l.replace("/", "_").replace(" ", "_").replace("-", "_")
    for l in DISTORTION_LABELS
]


# ── Expected Calibration Error ────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute ECE over a flat array of (probability, true_label) pairs.

    Parameters
    ----------
    probs  : 1-D array of model sigmoid probabilities (0..1)
    labels : 1-D array of ground-truth binary labels (0 or 1)
    n_bins : number of equal-width bins

    Returns
    -------
    float - ECE score (lower = better calibrated)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(acc - conf)
    return float(ece)


def bin_stats(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (bin_centres, mean_accuracy_per_bin, fraction_in_bin) for plotting.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, fracs = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        centres.append((lo + hi) / 2)
        accs.append(labels[mask].mean() if mask.sum() > 0 else 0.0)
        fracs.append(mask.sum() / len(probs))
    return np.array(centres), np.array(accs), np.array(fracs)


# ── Load test predictions ──────────────────────────────────────────────────────

def load_predictions(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


# ── Per-label calibration ──────────────────────────────────────────────────────

def compute_per_label_ece(df: pd.DataFrame, n_bins: int = 10) -> dict:
    results = {}
    for label, short in zip(DISTORTION_LABELS, SHORT):
        prob_col = f"prob_{short}"
        true_col = f"true_{short}"
        if prob_col not in df.columns or true_col not in df.columns:
            continue
        probs  = df[prob_col].values.astype(float)
        labels = df[true_col].values.astype(int)
        ece    = expected_calibration_error(probs, labels, n_bins)
        results[label] = round(ece, 4)
    return results


# ── Reliability diagram ────────────────────────────────────────────────────────

def plot_reliability_diagram(
    df: pd.DataFrame,
    out_path: str,
    n_bins: int = 10,
) -> None:
    """
    Plot one reliability diagram per distortion label (2x5 grid).
    Each subplot shows predicted confidence vs. actual accuracy.
    The diagonal is perfect calibration.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for ax, (label, short) in zip(axes, zip(DISTORTION_LABELS, SHORT)):
        prob_col = f"prob_{short}"
        true_col = f"true_{short}"
        if prob_col not in df.columns:
            ax.set_visible(False)
            continue

        probs  = df[prob_col].values.astype(float)
        labels = df[true_col].values.astype(int)
        centres, accs, fracs = bin_stats(probs, labels, n_bins)
        ece = expected_calibration_error(probs, labels, n_bins)

        # Reliability bars
        ax.bar(centres, accs, width=1.0 / n_bins, alpha=0.7,
               color="#4a6fa5", edgecolor="white", label="Accuracy")
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nECE={ece:.3f}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Reliability Diagrams - DistilBERT Cognitive Distortion Classifier",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Overall calibration ───────────────────────────────────────────────────────

def compute_overall_ece(df: pd.DataFrame, n_bins: int = 10) -> float:
    """Pool all labels into one flat array, then compute ECE."""
    all_probs, all_labels = [], []
    for short in SHORT:
        prob_col = f"prob_{short}"
        true_col = f"true_{short}"
        if prob_col not in df.columns:
            continue
        all_probs.extend(df[prob_col].values.astype(float).tolist())
        all_labels.extend(df[true_col].values.astype(int).tolist())
    return expected_calibration_error(
        np.array(all_probs), np.array(all_labels), n_bins
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    csv_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Not found: {csv_path}\n"
            "Run python -m src.evaluate first to generate predictions."
        )

    print("Loading predictions ...")
    df = load_predictions(csv_path)
    print(f"  {len(df)} test examples, {len(df.columns)} columns")

    print("\nComputing per-label ECE ...")
    per_label_ece = compute_per_label_ece(df)
    for label, ece in per_label_ece.items():
        bar = "#" * int(ece * 40)
        print(f"  {label:<40} ECE={ece:.4f}  {bar}")

    overall_ece = compute_overall_ece(df)
    print(f"\n  Overall ECE (pooled across all labels): {overall_ece:.4f}")
    print("  (0 = perfect calibration, 0.5 = worst possible)")

    print("\nPlotting reliability diagrams ...")
    diagram_path = os.path.join(RESULTS_DIR, "calibration_curves.png")
    plot_reliability_diagram(df, diagram_path)

    report = {
        "overall_ece": round(overall_ece, 4),
        "per_label_ece": per_label_ece,
        "n_bins": 10,
        "n_test_examples": len(df),
        "note": (
            "ECE computed from sigmoid probabilities at threshold 0.40. "
            "Lower ECE = better calibration = confidence tracks accuracy."
        ),
    }

    report_path = os.path.join(RESULTS_DIR, "calibration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    return report


if __name__ == "__main__":
    run()
