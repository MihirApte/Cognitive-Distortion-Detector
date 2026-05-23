---
title: Cognitive Distortion Detector
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Cognitive Distortion Detector

A multi-label NLP classifier that identifies cognitive distortions in text, grounded in Cognitive Behavioural Therapy (CBT). Built as a portfolio project combining transformer fine-tuning, explainability (LIME), and a full-stack deployment on HuggingFace Spaces.

**[Try the live demo →](https://huggingface.co/spaces/MihirApte/cognitive-distortion-detector)**

---

## What It Does

Type a thought or sentence (up to 150 words). The app:

1. Runs it through a fine-tuned DistilBERT model to detect which of 10 cognitive distortions are present
2. Uses LIME to highlight which words drove each prediction
3. Shows a CBT-grounded therapy response for each detected distortion - what the distortion is, a relevant technique, a Socratic challenge question, and a suggested reframe

---

## The 10 Distortions

| # | Label | CBT Technique |
|---|-------|---------------|
| 1 | All-or-nothing thinking | Continuum Method |
| 2 | Overgeneralization | Evidence Examination |
| 3 | Mental filter | Positive Data Log |
| 4 | Should statements | Preference Reframing |
| 5 | Labeling | Behaviour–Identity Separation |
| 6 | Personalization | Responsibility Pie Chart |
| 7 | Catastrophising/Magnification | Decatastrophising |
| 8 | Emotional Reasoning | Emotion vs. Fact Check |
| 9 | Mind Reading | Behavioural Experiment |
| 10 | Fortune-telling | Probability Estimation |

---

## ⚠️ Phase 1 - Work in Progress

This is **Phase 1** of an ongoing project. The deployed model is functional but performance is limited by training data quality. False positives and label confusion can occur. A clear "Work in Progress" notice is shown in the app.

**Phase 2 is in active development:** DeBERTa-v3-base + Focal Loss + semantically distinct per-distortion training data. Expected to significantly reduce false positives and improve label separation.

---

## Model Performance

**Phase 1:** Fine-tuned DistilBERT (`distilbert-base-uncased`) on ~2,020 training examples. The API runs at **threshold = 0.40** (higher precision, lower recall compared to the evaluation below which used 0.20).

| Metric | Score (threshold=0.20) |
|--------|-------|
| Micro F1 | 0.237 |
| Macro F1 | 0.229 |
| Hamming Loss | 0.259 |

**Per-label F1:**

| Label | F1 |
|-------|----|
| All-or-nothing thinking | 0.182 |
| Overgeneralization | 0.267 |
| Mental filter | 0.173 |
| Should statements | 0.188 |
| Labeling | 0.304 |
| Personalization | 0.185 |
| Catastrophising/Magnification | 0.279 |
| Emotional Reasoning | 0.241 |
| Mind Reading | 0.264 |
| Fortune-telling | 0.210 |

**On the scores:** These are modest in absolute terms, and that's expected. The dataset (2,530 examples, 10 overlapping labels, sourced from therapist Q&A) is small, imbalanced, and inherently noisy - even human annotators disagree on many examples. The root cause of current limitations is template-based synthetic data using shared vocabulary across distortion classes. This is the primary target for Phase 2. Full error analysis is in `notebooks/03_error_analysis.ipynb`.

---

## Architecture

```
User input (text)
       │
       ▼
FastAPI /analyse endpoint
       │
       ├─ 1. DistilBERT inference → sigmoid probs for 10 labels
       │
       ├─ 2. LIME explainer → top 6 keywords per detected label
       │
       └─ 3. Static guidance lookup → CBT technique + reframe
                    │
                    ▼
             JSON response → rendered in browser
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Model | DistilBERT (`distilbert-base-uncased`), HuggingFace Transformers |
| Training | Google Colab (T4 GPU), 10 epochs |
| Explainability | LIME (`lime.lime_text`) |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | Vanilla HTML / CSS / JavaScript (single file) |
| Deployment | HuggingFace Spaces (Docker, CPU) |

---

## Dataset

[danthareja/cognitive-distortion](https://huggingface.co/datasets/danthareja/cognitive-distortion)

2,530 annotated patient questions from a therapist Q&A platform. Pre-split: 2,020 train / 506 test. Multi-label - one question can exhibit multiple distortions simultaneously.

Original source: Shreevastava & Foltz (2021), *Detecting Cognitive Distortions from Patient-Therapist Interactions*.

---

## Project Structure

```
NLP_CBT/
├── src/
│   ├── config.py          # Labels, paths, hyperparameters
│   ├── preprocess.py      # Data loading & tokenisation
│   ├── train.py           # DistilBERT fine-tuning loop
│   ├── evaluate.py        # Metrics + predictions CSV
│   ├── explain.py         # LIME explainability wrapper
│   ├── guidance.py        # Static CBT guidance lookup
│   └── api/
│       └── main.py        # FastAPI app (inference + frontend serving)
├── frontend/
│   └── index.html         # Single-file UI
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_lime_explanations.ipynb
├── models/
│   └── distilbert_cognitive_distortion/   # Saved weights (Git LFS)
├── results/               # Evaluation outputs, LIME HTMLs
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/MihirApte/cognitive-distortion-detector.git
cd cognitive-distortion-detector

# 2. Pull model weights via Git LFS
git lfs install
git lfs pull

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start the server
uvicorn src.api.main:app --reload --port 8000

# 6. Open in browser
# http://localhost:8000
```

---

## Run with Docker

```bash
docker build -t cdd-app .
docker run -p 7860:7860 cdd-app
# Open: http://localhost:7860
```

**Note on latency:** LIME runs ~300 perturbations through DistilBERT on CPU per detected label. Expect 10–20 seconds per request on the free HuggingFace Spaces tier. This is a known trade-off of running LIME on CPU.

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `01_eda.ipynb` | Label distribution, class imbalance, text length analysis |
| `02_training.ipynb` | DistilBERT fine-tuning on Colab T4, training curves |
| `03_error_analysis.ipynb` | Per-label F1, confusion patterns, failure cases |
| `04_lime_explanations.ipynb` | LIME word importance visualisations per distortion |
