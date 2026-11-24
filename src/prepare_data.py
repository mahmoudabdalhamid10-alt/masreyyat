"""Prepare Masreyyat datasets from the raw downloads in data/.

Inputs (see DATA_PLAN.md for provenance and terms):
  - data/hard/data/balanced/balanced-reviews.txt   (HARD, UTF-16 TSV)
  - data/arsas.parquet                             (ArSAS tweets)

Outputs:
  - data/processed/hard_binary.csv     review,label  (label: neg/pos)
  - data/processed/arsas_binary.csv    text,label    (neg/pos only)
  - data/processed/arsas_full.csv      text,label    (neg/neu/pos/mixed)
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)


def prepare_hard():
    path = os.path.join(ROOT, "data", "hard", "data", "balanced", "balanced-reviews.txt")
    df = pd.read_csv(path, sep="\t", encoding="utf-16")
    df = df.dropna(subset=["review", "rating"])
    df = df[df["review"].astype(str).str.strip().str.len() > 0]
    df["label"] = df["rating"].map(lambda r: "neg" if int(r) <= 2 else "pos")
    out = df[["review", "label"]].rename(columns={"review": "text"})
    out.to_csv(os.path.join(OUT, "hard_binary.csv"), index=False)
    print(f"HARD binary: {len(out)} rows, {out['label'].value_counts().to_dict()}")


def prepare_arsas():
    path = os.path.join(ROOT, "data", "arsas.parquet")
    df = pd.read_parquet(path)
    label_map = {0: "neg", 1: "neu", 2: "pos", 3: "mixed"}
    df["label"] = df["label"].map(label_map)
    full = df[["Tweet_text", "label"]].rename(columns={"Tweet_text": "text"})
    full = full[full["text"].astype(str).str.strip().str.len() > 0]
    full.to_csv(os.path.join(OUT, "arsas_full.csv"), index=False)
    binary = full[full["label"].isin(["neg", "pos"])]
    binary.to_csv(os.path.join(OUT, "arsas_binary.csv"), index=False)
    print(f"ArSAS full: {len(full)} rows, {full['label'].value_counts().to_dict()}")
    print(f"ArSAS binary: {len(binary)} rows, {binary['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    prepare_hard()
    prepare_arsas()
