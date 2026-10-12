---
title: Cognitive Distortion Detector
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Cognitive Distortion Detector

A multi-label NLP classifier that identifies cognitive distortions in text, grounded in Cognitive Behavioural Therapy (CBT). Built as a portfolio project combining transformer fine-tuning, explainability (LIME), an LLM explanation layer with hallucination checks, safety guardrails, and a full-stack deployment on HuggingFace Spaces.

**[Try the live demo →](https://huggingface.co/spaces/MihirApte/cognitive-distortion-detector)**

---

## What It Does

Type a thought or paragraph (up to 200 words). The app:

1. Screens the text for crisis or misuse signals (a fixed safe message is returned instead of any model output)
2. Runs it through a fine-tuned transformer to detect which of 10 cognitive distortions are present
3. Uses LIME to highlight which words drove each prediction
4. Optionally writes a short plain-English explanation grounded in those LIME words (checked by a citation verifier)
5. Shows a CBT-grounded response for each detected distortion: what it is, a relevant technique, a Socratic challenge question, and a suggested reframe

The top 4 distortions are shown first, with a "show more" button for the rest.

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

## Model Performance

Two fine-tuned transformers (DistilBERT and MentalRoBERTa) and two LLM baselines (zero-shot and few-shot) were evaluated on the same 10 labels. The deployed model runs at **threshold = 0.40**.

> Fine-tuned models: full 506-row held-out test set. LLM baselines (qwen3.8-27b via Groq): a 50-row subset of that test set, so their numbers are indicative only and not directly comparable. DistilBERT is shown at both 0.40 (the threshold used for MentalRoBERTa) and 0.20 (the threshold it was originally evaluated at). MentalRoBERTa was trained on 1,821 rows with 203 held out for choosing the best epoch; DistilBERT used about 2,020.

### Overall metrics

| Metric | DistilBERT (t=0.40) | DistilBERT (t=0.20) | MentalRoBERTa (t=0.40) | LLM zero-shot | LLM few-shot |
|---|:---:|:---:|:---:|:---:|:---:|
| Macro F1 | 0.172 | 0.229 | **0.268** | 0.211 | 0.239 |
| Micro F1 | 0.177 | 0.236 | **0.293** | - | - |
| Macro precision | 0.192 | 0.151 | **0.239** | - | - |
| Macro recall | 0.172 | **0.489** | 0.323 | - | - |
| Hamming loss (lower = better) | **0.121** | 0.259 | 0.136 | - | - |
| Labels with F1 ≥ 0.25 (of 10) | 2 | 4 | **6** | - | - |
| Parameters | ~66M | ~66M | ~125M | ~27B | ~27B |
| Inference | local, offline | local, offline | local, offline | API call | API call |
| Test rows | 506 | 506 | 506 | 50 | 50 |

### Per-label F1 (threshold 0.40 for both fine-tuned models)

| Distortion | DistilBERT | MentalRoBERTa | LLM zero-shot | LLM few-shot | Better fine-tuned model |
|---|:---:|:---:|:---:|:---:|:---:|
| All-or-nothing thinking | 0.109 | 0.063 | 0.000 | 0.000 | DistilBERT |
| Overgeneralization | 0.033 | 0.286 | 0.105 | 0.154 | MentalRoBERTa |
| Mental filter | 0.161 | 0.171 | 0.286 | 0.667 | MentalRoBERTa |
| Should statements | 0.185 | 0.279 | 0.333 | 0.000 | MentalRoBERTa |
| Labeling | 0.274 | 0.416 | 0.500 | 0.889 | MentalRoBERTa |
| Personalization | 0.146 | 0.234 | 0.500 | 0.500 | MentalRoBERTa |
| Catastrophising/Magnification | 0.274 | 0.356 | 0.133 | 0.000 | MentalRoBERTa |
| Emotional Reasoning | 0.219 | 0.184 | 0.000 | 0.000 | DistilBERT |
| Mind Reading | 0.143 | 0.414 | 0.250 | 0.182 | MentalRoBERTa |
| Fortune-telling | 0.174 | 0.275 | 0.000 | 0.000 | MentalRoBERTa |

**Reading the results:**
- MentalRoBERTa is better on 8 of 10 labels and about 0.10 macro F1 higher than DistilBERT at the same threshold. DistilBERT only catches up on recall by using a very low threshold, which also floods the output with false positives (precision 0.15).
- The LLMs do well on semantically distinctive labels such as Labeling and Mental filter, but score zero on others. With only 50 rows these per-label numbers are very noisy.
- Absolute scores are still modest. The dataset is small (2,530 examples), imbalanced, multi-label, and inherently noisy, since even human annotators disagree on many examples. This looks like a data limit more than a model limit, which is why Phase 3 targets the data.
- Per-label F1 for All-or-nothing thinking remains very low for every model.

Full error analysis is in `notebooks/03_error_analysis.ipynb`.

---

## Safety and Responsible AI

- **Crisis routing:** regex screening for suicidal ideation, self-harm and passive ideation runs before any model. Matches get a fixed safe message with helplines and never reach the classifier or the LLM.
- **Misuse detection:** jailbreak attempts and requests for therapy or medical advice are refused.
- **Adversarial tests:** 20 test cases (`tests/test_adversarial.py`), all passing.
- **Grounded explanations:** the optional LLM explanation may only cite words in the LIME top-word list; a verifier flags anything else.
- **Calibration:** DistilBERT overall ECE = 0.077 (see `results/calibration_report.json`). Not yet measured for MentalRoBERTa.
- **Not a diagnosis:** predictions are for non-clinical, educational use only. See `MODEL_CARD.md`.

---

## Architecture

```
User input (text, up to 200 words)
       │
       ▼
FastAPI /analyse endpoint
       │
       ├─ 0. Safety: crisis check → misuse check (fixed responses, no model)
       │
       ├─ 1. MentalRoBERTa inference → sigmoid probs for 10 labels
       │
       ├─ 2. LIME explainer → top keywords per detected label
       │
       ├─ 3. Static guidance lookup → CBT technique + reframe
       │
       └─ 4. (optional) LLM explanation grounded in LIME words + citation check
                    │
                    ▼
             JSON response → rendered in browser
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Model | MentalRoBERTa (`mental/mental-roberta-base`), HuggingFace Transformers, PyTorch |
| Baselines | DistilBERT, LLM zero-/few-shot via Groq API |
| Training | Google Colab (T4 GPU), weighted BCE loss, max length 256 |
| Explainability | LIME (`lime.lime_text`) |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | Vanilla HTML / CSS / JavaScript (single file) |
| Deployment | HuggingFace Spaces (Docker, CPU) |

---

## Dataset

[danthareja/cognitive-distortion](https://huggingface.co/datasets/danthareja/cognitive-distortion)

2,530 annotated patient questions from a therapist Q&A platform. Pre-split: 2,020 train / 506 test (MentalRoBERTa further splits the training set into 1,821 train / 203 validation). Multi-label: one question can exhibit multiple distortions simultaneously.

Original source: Shreevastava & Foltz (2021), *Detecting Cognitive Distortions from Patient-Therapist Interactions*.

---

## Project Structure

```
NLP_CBT/
├── src/
│   ├── config.py            # Labels, paths, hyperparameters
│   ├── preprocess.py        # Data loading & tokenisation
│   ├── baseline.py          # TF-IDF baseline
│   ├── evaluate.py          # Metrics + predictions CSV (DistilBERT)
│   ├── train_phase2.py      # MentalRoBERTa fine-tuning
│   ├── explain.py           # LIME explainability wrapper
│   ├── guidance.py          # Static CBT guidance lookup
│   ├── safety.py            # Crisis routing + misuse detection
│   ├── calibration.py       # ECE + reliability diagrams
│   ├── llm_baseline.py      # Zero-/few-shot LLM baseline
│   ├── llm_explainer.py     # LIME-grounded LLM explanations
│   └── api/
│       └── main.py          # FastAPI app (inference + frontend serving)
├── frontend/
│   └── index.html           # Single-file UI
├── notebooks/               # 01 EDA ... 06 MentalRoBERTa training
├── tests/
│   └── test_adversarial.py  # 20 safety tests
├── models/
│   ├── distilbert_cognitive_distortion/       # Baseline weights (Git LFS)
│   └── mentalroberta_cognitive_distortion/    # Deployed weights (Git LFS)
├── results/                 # Evaluation outputs, LIME HTMLs
├── MODEL_CARD.md
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

# 5. (optional) enable LLM explanations
export GROQ_API_KEY=your_key    # Windows PowerShell: $env:GROQ_API_KEY="your_key"

# 6. Start the server
uvicorn src.api.main:app --reload --port 8000

# 7. Open in browser
# http://localhost:8000
```

The fine-tuned weights load from the local `models/` folder, so no Hugging Face login is needed to run the app. A login is only needed to retrain, because the base model is gated.

---

## Run with Docker

```bash
docker build -t cdd-app .
docker run -p 7860:7860 cdd-app
# Open: http://localhost:7860
```

**Note on latency:** LIME runs ~300 perturbations through the model on CPU per detected label. Expect noticeably slower responses on the free HuggingFace Spaces tier, especially for long inputs. This is a known trade-off of running LIME on CPU.

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `01_eda.ipynb` | Label distribution, class imbalance, text length analysis |
| `02_train_distilbert.ipynb` | DistilBERT fine-tuning on Colab T4, training curves |
| `03_error_analysis.ipynb` | Per-label F1, confusion patterns, failure cases |
| `04_lime_explanations.ipynb` | LIME word importance visualisations per distortion |
| `05_synthetic_data.ipynb` | Synthetic data generation experiments |
| `06_mentalroberta_training.ipynb` | MentalRoBERTa fine-tuning on Colab (needs Hugging Face login) |

---

## Phase 3 (Planned): Better Data

Experiments showed the limit is data quality, not model choice, so Phase 3 will focus only on data and will not add new models:

- Replace short single-label synthetic sentences (about 15 words) with realistic multi-label paragraphs that match the real data (about 165 words)
- Add more examples for the weakest labels: All-or-nothing thinking, Mental filter, Should statements
- Add hard negatives (neutral or reasonable thinking that sounds similar to a distortion)
- Re-validate labels with a psychologist and re-evaluate on the untouched test set

---

## Licensing and Attribution

The deployed model is a fine-tuned version of [MentalRoBERTa](https://huggingface.co/mental/mental-roberta-base) (Ji et al.), released under [CC BY-NC 4.0](https://spdx.org/licenses/CC-BY-NC-4.0). The fine-tuned weights in this repository are therefore for **non-commercial, non-clinical use only**. Changes made: fine-tuned for 10-label cognitive distortion classification.
