import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DISTORTION_LABELS, LABEL2ID, NUM_LABELS,
    CSV_LABEL_MAP, CSV_NO_DISTORTION,
    DATA_RAW_DIR, DATA_PROC_DIR, SEED
)



def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")

    
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "patient question" in cl:
            rename_map[col] = "text"
        elif "dominant" in cl:
            rename_map[col] = "dominant_distortion"
        elif "secondary" in cl:
            rename_map[col] = "secondary_distortion"
        elif "distorted part" in cl:
            rename_map[col] = "distorted_part"
    df = df.rename(columns=rename_map)

    df["text"] = df["text"].astype(str).str.strip()
    df["dominant_distortion"] = df["dominant_distortion"].astype(str).str.strip()

    if "secondary_distortion" in df.columns:
        df["secondary_distortion"] = df["secondary_distortion"].astype(str).str.strip()
    else:
        df["secondary_distortion"] = ""

    return df


def normalise_label(raw: str):
    """Map a CSV label string to our canonical label, or return None."""
    raw = str(raw).strip()
    if raw == CSV_NO_DISTORTION or raw.lower() == "no distortion":
        return None                      
    return CSV_LABEL_MAP.get(raw, raw)   


def build_label_vector(dominant: str, secondary: str) -> list:

    vec = [0] * NUM_LABELS

    dom = normalise_label(dominant)
    if dom and dom in LABEL2ID:
        vec[LABEL2ID[dom]] = 1

    if secondary and secondary not in ("nan", "", "None", "NaN"):
        sec = normalise_label(secondary)
        if sec and sec in LABEL2ID:
            vec[LABEL2ID[sec]] = 1

    return vec


def run(csv_filename: str = "Annotated_data.csv") -> None:
    csv_path = os.path.join(DATA_RAW_DIR, csv_filename)
    os.makedirs(DATA_PROC_DIR, exist_ok=True)

    print(f"Loading: {csv_path}")
    df = load_raw(csv_path)
    print(f"Rows loaded: {len(df)}")

    
    label_vecs = df.apply(
        lambda row: build_label_vector(
            row["dominant_distortion"], row["secondary_distortion"]
        ),
        axis=1
    )
    label_df = pd.DataFrame(label_vecs.tolist(), columns=DISTORTION_LABELS)
    df = pd.concat(
        [df[["text", "dominant_distortion", "secondary_distortion"]], label_df],
        axis=1
    )

    
    distortion_rows    = df[df[DISTORTION_LABELS].sum(axis=1) > 0]
    no_distortion_rows = df[df[DISTORTION_LABELS].sum(axis=1) == 0]
    multi_label_rows   = df[df[DISTORTION_LABELS].sum(axis=1) == 2]
    print(f"Rows with >=1 distortion label   : {len(distortion_rows)}")
    print(f"Rows with no distortion (all 0s) : {len(no_distortion_rows)}")
    print(f"Multi-label rows (2 labels)      : {len(multi_label_rows)}")

    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["dominant_distortion"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"\nTrain size : {len(train_df)}")
    print(f"Test size  : {len(test_df)}")

    train_distortion_only = train_df[train_df[DISTORTION_LABELS].sum(axis=1) > 0]
    dom_labels = (
        train_distortion_only["dominant_distortion"]
        .map(normalise_label)
        .dropna()
    )

    present_labels = np.array(sorted(dom_labels.unique()))
    raw_weights    = compute_class_weight(
        class_weight="balanced",
        classes=present_labels,
        y=dom_labels.values
    )

    
    class_weights = np.ones(NUM_LABELS, dtype=np.float32)
    for label, weight in zip(present_labels, raw_weights):
        if label in LABEL2ID:
            class_weights[LABEL2ID[label]] = weight

    print("\nClass weights aligned to DISTORTION_LABELS order:")
    for label, w in zip(DISTORTION_LABELS, class_weights):
        print(f"  {label:<35} {w:.4f}")

   
    train_path   = os.path.join(DATA_PROC_DIR, "train.csv")
    test_path    = os.path.join(DATA_PROC_DIR, "test.csv")
    weights_path = os.path.join(DATA_PROC_DIR, "class_weights.npy")

    train_df.to_csv(train_path,   index=False)
    test_df.to_csv(test_path,    index=False)
    np.save(weights_path, class_weights)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {weights_path}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    run()
