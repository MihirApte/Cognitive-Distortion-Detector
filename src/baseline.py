import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score, hamming_loss, classification_report, confusion_matrix
)

from src.config import DISTORTION_LABELS, DATA_PROC_DIR, RESULTS_DIR, SEED


#  Load processed data 

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_PROC_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_PROC_DIR, "test.csv"))

    X_train = train_df["text"].astype(str).values
    X_test  = test_df["text"].astype(str).values

    Y_train = train_df[DISTORTION_LABELS].values.astype(int)
    Y_test  = test_df[DISTORTION_LABELS].values.astype(int)

    print(f"Train: {X_train.shape[0]} examples, {Y_train.shape[1]} labels")
    print(f"Test : {X_test.shape[0]} examples,  {Y_test.shape[1]} labels")
    return X_train, X_test, Y_train, Y_test


#  Build & train pipeline 

def train_baseline(X_train, Y_train):
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),     
        sublinear_tf=True,      
        strip_accents="unicode",
        analyzer="word",
        min_df=2                
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")

    clf = MultiOutputClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=SEED
        ),
        n_jobs=-1   
    )
    clf.fit(X_train_tfidf, Y_train)
    print("Training complete.")
    return vectorizer, clf


#  Evaluate 

def evaluate(vectorizer, clf, X_test, Y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    Y_pred       = clf.predict(X_test_tfidf)

    
    per_label_f1 = f1_score(Y_test, Y_pred, average=None, zero_division=0)

    
    micro_f1  = f1_score(Y_test, Y_pred, average="micro",   zero_division=0)
    macro_f1  = f1_score(Y_test, Y_pred, average="macro",   zero_division=0)
    hamming   = hamming_loss(Y_test, Y_pred)

    results = {
        "model"          : "TF-IDF + Logistic Regression (baseline)",
        "micro_f1"       : round(float(micro_f1),  4),
        "macro_f1"       : round(float(macro_f1),  4),
        "hamming_loss"   : round(float(hamming),    4),
        "per_label_f1"   : {
            label: round(float(score), 4)
            for label, score in zip(DISTORTION_LABELS, per_label_f1)
        }
    }
    return results, Y_pred


#  Print & save results 

def print_results(results):
    print("\n" + "=" * 60)
    print("BASELINE RESULTS - TF-IDF + Logistic Regression")
    print("=" * 60)
    print(f"  Micro F1      : {results['micro_f1']:.4f}")
    print(f"  Macro F1      : {results['macro_f1']:.4f}")
    print(f"  Hamming Loss  : {results['hamming_loss']:.4f}  (lower is better)")
    print()
    print(f"  {'Label':<35} {'F1':>6}")
    print(f"  {'-'*35} {'-'*6}")
    for label, f1 in results["per_label_f1"].items():
        flag = "  ⚠ low" if f1 < 0.40 else ""
        print(f"  {label:<35} {f1:>6.4f}{flag}")
    print("=" * 60)


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


#  Per-label F1 bar chart 

def plot_per_label_f1(results):
    labels = list(results["per_label_f1"].keys())
    scores = list(results["per_label_f1"].values())

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#d62728" if s < 0.40 else "#1f77b4" for s in scores]
    bars = ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=0.8)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.axhline(0.40, color="red", linestyle="--", linewidth=1.2, label="F1 = 0.40 threshold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline Per-Label F1 Score (TF-IDF + Logistic Regression)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Distortion Label")
    ax.set_ylabel("F1 Score")
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.legend()

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "baseline_per_label_f1.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {chart_path}")


#  Main 

if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, Y_train, Y_test = load_data()

    print("\nTraining baseline...")
    vectorizer, clf = train_baseline(X_train, Y_train)

    print("\nEvaluating...")
    results, Y_pred = evaluate(vectorizer, clf, X_test, Y_test)

    print_results(results)
    save_results(results)
    plot_per_label_f1(results)

    print("\nDone. Baseline complete.")
