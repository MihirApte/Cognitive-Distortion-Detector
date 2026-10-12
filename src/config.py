import os

#  Label Set (10 Cognitive Distortions - Beck's Taxonomy) 
DISTORTION_LABELS = [
    "All-or-nothing thinking",
    "Overgeneralization",
    "Mental filter",
    "Should statements",
    "Labeling",
    "Personalization",
    "Catastrophising/Magnification",
    "Emotional Reasoning",
    "Mind Reading",
    "Fortune-telling",
]

NUM_LABELS = len(DISTORTION_LABELS)   # 10

# Label to index mapping
LABEL2ID = {label: idx for idx, label in enumerate(DISTORTION_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(DISTORTION_LABELS)}


CSV_LABEL_MAP = {
    "Magnification"         : "Catastrophising/Magnification",
    "All-or-nothing thinking": "All-or-nothing thinking",
    "Overgeneralization"    : "Overgeneralization",
    "Mental filter"         : "Mental filter",
    "Should statements"     : "Should statements",
    "Labeling"              : "Labeling",
    "Personalization"       : "Personalization",
    "Emotional Reasoning"   : "Emotional Reasoning",
    "Mind Reading"          : "Mind Reading",
    "Fortune-telling"       : "Fortune-telling",
}

CSV_NO_DISTORTION = "No Distortion"

#  HuggingFace Dataset (kept for reference) 
HF_DATASET_NAME = "danthareja/cognitive-distortion"

#  Models
BASE_MODEL_DISTILBERT    = "distilbert-base-uncased"       # Phase 1 model
BASE_MODEL_MENTALROBERTA = "mental/mental-roberta-base"    # Phase 2 model (CC BY-NC 4.0, gated)
MAX_TOKEN_LENGTH      = 512
MAX_TOKEN_LENGTH_PHASE2 = 256   # used for MentalRoBERTa training

#  Training Hyperparameters 
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 5
WARMUP_STEPS  = 100
WEIGHT_DECAY  = 0.01
THRESHOLD     = 0.5   

#  Paths 
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR  = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
DATA_SYN_DIR  = os.path.join(ROOT_DIR, "data", "synthetic")
MODELS_DIR              = os.path.join(ROOT_DIR, "models")
DISTILBERT_MODEL_DIR    = os.path.join(ROOT_DIR, "models", "distilbert_cognitive_distortion")
MENTALROBERTA_MODEL_DIR = os.path.join(ROOT_DIR, "models", "mentalroberta_cognitive_distortion")
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")

#  Random Seed 
SEED = 42
