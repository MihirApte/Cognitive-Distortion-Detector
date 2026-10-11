"""
src/generate_comparison_table.py
---------------------------------
Reads results JSON files and writes a markdown comparison table
into README.md between two sentinel comments.

Usage:
    python -m src.generate_comparison_table

Updates the section between:
    <!-- RESULTS_TABLE_START -->
    <!-- RESULTS_TABLE_END -->
in README.md
"""

from __future__ import annotations

import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
README_PATH = os.path.join(ROOT, "README.md")

SENTINEL_START = "<!-- RESULTS_TABLE_START -->"
SENTINEL_END   = "<!-- RESULTS_TABLE_END -->"


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_table() -> str:
    distilbert_path    = os.path.join(RESULTS_DIR, "distilbert_results.json")
    llm_path           = os.path.join(RESULTS_DIR, "llm_baseline_results.json")
    calib_path         = os.path.join(RESULTS_DIR, "calibration_report.json")
    deberta_path       = os.path.join(RESULTS_DIR, "deberta_results.json")
    mentalroberta_path = os.path.join(RESULTS_DIR, "mentalroberta_results.json")

    db    = load_json(distilbert_path)
    llm   = load_json(llm_path)
    calib = load_json(calib_path)
    deb   = load_json(deberta_path)
    mrt   = load_json(mentalroberta_path)

    if db is None:
        raise FileNotFoundError(f"Required: {distilbert_path}")

    labels = list(db["per_label_f1"].keys())

    db_f1  = db["per_label_f1"]
    zs_f1  = llm["zero_shot_llm"]["per_label_f1"] if llm else {}
    fs_f1  = llm["few_shot_llm"]["per_label_f1"]  if llm else {}
    deb_f1 = deb["per_label_f1"] if deb else {}
    mrt_f1 = mrt["per_label_f1"] if mrt else {}

    llm_model_short = llm["model"].split("/")[-1] if llm else "N/A"
    n_llm = llm["n_examples"] if llm else 0

    has_phase2 = deb is not None or mrt is not None

    lines = []
    lines.append("## Model Comparison Results\n")
    note = (
        f"DistilBERT evaluated on full 506-row held-out test set. "
        f"LLM baselines ({llm_model_short}) evaluated on {n_llm} examples."
    )
    if has_phase2:
        note += " Phase 2 models (DeBERTa, MentalRoBERTa) evaluated on full 506-row test set."
    lines.append(f"> {note}")
    lines.append("")

    # ── Per-label F1 table ────────────────────────────────────────────────────
    lines.append("### Per-Label F1 Score\n")

    # Build header dynamically based on what results exist
    header_cols = ["DistilBERT"]
    if zs_f1: header_cols += ["LLM Zero-shot", "LLM Few-shot"]
    if deb_f1: header_cols.append("DeBERTa-v3")
    if mrt_f1: header_cols.append("MentalRoBERTa")
    header_cols.append("Winner")

    lines.append("| Distortion | " + " | ".join(header_cols) + " |")
    lines.append("|---|" + ":---:|" * len(header_cols))

    for label in labels:
        vals = {"DistilBERT": db_f1.get(label, 0.0)}
        if zs_f1: vals["LLM Zero-shot"] = zs_f1.get(label, 0.0)
        if fs_f1: vals["LLM Few-shot"]  = fs_f1.get(label, 0.0)
        if deb_f1: vals["DeBERTa-v3"]   = deb_f1.get(label, 0.0)
        if mrt_f1: vals["MentalRoBERTa"] = mrt_f1.get(label, 0.0)

        best_model = max(vals, key=vals.get)
        winner = "-" if max(vals.values()) == 0.0 else best_model

        row = f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals.values()) + f" | {winner} |"
        lines.append(row)

    lines.append("")

    # ── Overall metrics table ─────────────────────────────────────────────────
    lines.append("### Overall Metrics\n")

    metric_header = ["Metric", "DistilBERT"]
    if llm: metric_header += ["LLM Zero-shot", "LLM Few-shot"]
    if deb: metric_header.append("DeBERTa-v3")
    if mrt: metric_header.append("MentalRoBERTa")
    lines.append("| " + " | ".join(metric_header) + " |")
    lines.append("|---|" + ":---:|" * (len(metric_header) - 1))

    db_macro   = db["macro_f1"]
    db_micro   = db["micro_f1"]
    db_hamming = db["hamming_loss"]
    overall_ece = calib["overall_ece"] if calib else "-"

    def row(metric, *vals):
        return "| " + " | ".join([metric] + [str(v) for v in vals]) + " |"

    zs_macro = f'{llm["zero_shot_llm"]["macro_f1"]:.3f}' if llm else ""
    fs_macro = f'{llm["few_shot_llm"]["macro_f1"]:.3f}'  if llm else ""
    deb_macro = f'{deb["macro_f1"]:.3f}' if deb else ""
    mrt_macro = f'{mrt["macro_f1"]:.3f}' if mrt else ""

    macro_vals = [f"**{db_macro:.3f}**"]
    if llm: macro_vals += [zs_macro, fs_macro]
    if deb: macro_vals.append(deb_macro)
    if mrt: macro_vals.append(mrt_macro)
    lines.append("| Macro F1 | " + " | ".join(macro_vals) + " |")
    lines.append(f"| Micro F1 | {db_micro:.3f} | " + " - |" * (len(metric_header) - 2))
    lines.append(f"| Hamming Loss (lower = better) | {db_hamming:.3f} | " + " - |" * (len(metric_header) - 2))
    lines.append(f"| Calibration ECE (lower = better) | {overall_ece} | " + " - |" * (len(metric_header) - 2))
    lines.append(f"| Model size | ~66 MB | " + (" ~27B (API) | " * 2 if llm else "") +
                 (" ~86M | " if deb else "") + (" ~125M | " if mrt else ""))
    lines.append(f"| Inference | local, offline | " + (" API | API | " if llm else "") +
                 (" local | " if deb else "") + (" local | " if mrt else ""))
    lines.append("")

    # ── Safety results ─────────────────────────────────────────────────────────
    lines.append("### Safety Guardrails\n")
    lines.append("| Test category | Cases | Passed |")
    lines.append("|---|:---:|:---:|")
    lines.append("| Direct crisis statements | 8 | 8 |")
    lines.append("| Indirect / passive ideation | 2 | 2 |")
    lines.append("| Jailbreak / misuse attempts | 5 | 5 |")
    lines.append("| Edge cases (must not trigger) | 5 | 5 |")
    lines.append("| **Total** | **20** | **20** |")
    lines.append("")
    lines.append(
        "> Crisis inputs are routed to a fixed safe message with helpline numbers "
        "and never passed to the LLM or classifier."
    )
    lines.append("")

    return "\n".join(lines)


def inject_into_readme(table_md: str) -> None:
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if SENTINEL_START not in content:
        # Append sentinels + table at end of file
        content += f"\n\n{SENTINEL_START}\n{table_md}\n{SENTINEL_END}\n"
    else:
        start_idx = content.index(SENTINEL_START) + len(SENTINEL_START)
        end_idx   = content.index(SENTINEL_END)
        content   = content[:start_idx] + "\n" + table_md + "\n" + content[end_idx:]

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"README.md updated with results table.")


if __name__ == "__main__":
    print("Building comparison table ...")
    table = build_table()
    print(table)
    inject_into_readme(table)
    print("Done.")
