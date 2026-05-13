# Cognitive Distortion Detector

An NLP-based multi-label classification system that identifies cognitive distortions in text, grounded in Cognitive Behavioural Therapy (CBT).

Built as a portfolio project using DistilBERT fine-tuned on real patient-therapist interaction data, with a FastAPI backend and HTML/CSS/JS frontend.

---

## Cognitive Distortions Detected (10 Labels)

| # | Label |
|---|-------|
| 1 | All-or-nothing thinking |
| 2 | Overgeneralization |
| 3 | Mental filter |
| 4 | Should statements |
| 5 | Labeling |
| 6 | Personalization |
| 7 | Catastrophising/Magnification |
| 8 | Emotional Reasoning |
| 9 | Mind Reading |
| 10 | Fortune-telling |

---

## Project Structure

```
Cognitive-Distortion-Detector/
├── data/
│   ├── raw/            # Original dataset CSVs
│   ├── processed/      # Tokenised / cleaned data
│   └── synthetic/      # Augmented examples (optional)
├── notebooks/          # EDA, training experiments
├── src/
│   ├── config.py       # Shared constants and label set
│   ├── preprocess.py   # Data loading & preprocessing
│   ├── train.py        # DistilBERT fine-tuning
│   ├── evaluate.py     # Metrics (F1, Hamming loss)
│   └── explain.py      # LIME explainability
├── api/
│   └── main.py         # FastAPI backend
├── frontend/
│   ├── templates/      # HTML pages
│   └── static/         # CSS & JS
├── models/             # Saved model weights (gitignored)
├── results/            # Evaluation outputs
├── forms/              # Google Form drafts
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Model**: DistilBERT (`distilbert-base-uncased`) - HuggingFace Transformers
- **Training**: Google Colab (T4 GPU)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML / CSS / JavaScript
- **Explainability**: LIME (per-label text explanations)
- **Deployment**: HuggingFace Spaces (Docker)

---

## Dataset

[Cognitive Distortion Detection Dataset](https://huggingface.co/datasets/danthareja/cognitive-distortion)  
2,530 annotated patient questions from a therapist Q&A platform.  
Pre-split: 2,020 train / 506 test.

Original source: Shreevastava & Foltz (2021) - *Detecting Cognitive Distortions from Patient-Therapist Interactions*

---

## Setup

```bash
# Clone the repo
git clone https://github.com/MihirApte/Cognitive-Distortion-Detector.git
cd Cognitive-Distortion-Detector

# Create virtual environment
python -m venv venv
source venv/bin/activate      

# Install dependencies
pip install -r requirements.txt
```

---

