"""Train the Masreyyat v0.1 baseline and evaluate it honestly.

Model: character n-gram (3-5, word-bounded) TF-IDF + Logistic Regression.
Chosen because char n-grams are robust to Arabic orthographic variation
and dialect spelling, and the whole pipeline trains in minutes on a laptop.

Evaluations (all numbers land in results/metrics.json):
  1. In-domain:  stratified 90/10 split of HARD hotel reviews.
  2. Transfer:   the model is applied *unchanged* to ArSAS Egyptian-dialect
                 tweets (binary subset). Expect a big drop - that gap is the
                 point of the project, so we publish it, not hide it.
"""
import json
import os
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from arabic import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

RANDOM_SEED = 42


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=300_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])


def evaluate(model, texts, labels):
    preds = model.predict(texts)
    return {
        "n": len(labels),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "per_class": classification_report(labels, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=["neg", "pos"]).tolist(),
    }


def main():
    hard = pd.read_csv(os.path.join(PROC, "hard_binary.csv"))
    arsas = pd.read_csv(os.path.join(PROC, "arsas_binary.csv"))

    print("normalizing (this takes a minute on 100k+ reviews)...")
    hard["text"] = hard["text"].map(normalize)
    arsas["text"] = arsas["text"].map(normalize)

    train_df, test_df = train_test_split(
        hard, test_size=0.1, stratify=hard["label"], random_state=RANDOM_SEED
    )
    print(f"train: {len(train_df)}, in-domain test: {len(test_df)}, transfer test: {len(arsas)}")

    model = build_pipeline()
    t0 = time.time()
    model.fit(train_df["text"], train_df["label"])
    train_seconds = round(time.time() - t0, 1)
    print(f"trained in {train_seconds}s")

    metrics = {
        "model": "char_wb 3-5gram TF-IDF + LogisticRegression(C=2.0, class_weight=balanced)",
        "seed": RANDOM_SEED,
        "train_seconds": train_seconds,
        "train_data": f"HARD balanced hotel reviews, 90% stratified split (n={len(train_df)})",
        "in_domain_hard_test": evaluate(model, test_df["text"], test_df["label"]),
        "transfer_arsas_egyptian_tweets": evaluate(model, arsas["text"], arsas["label"]),
    }

    with open(os.path.join(RESULTS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    joblib.dump(model, os.path.join(MODELS, "baseline_charlogreg.joblib"))

    print("\n=== in-domain (HARD test) ===")
    print("accuracy:", metrics["in_domain_hard_test"]["accuracy"],
          "macro F1:", metrics["in_domain_hard_test"]["macro_f1"])
    print("\n=== transfer (ArSAS Egyptian tweets, unseen domain) ===")
    print("accuracy:", metrics["transfer_arsas_egyptian_tweets"]["accuracy"],
          "macro F1:", metrics["transfer_arsas_egyptian_tweets"]["macro_f1"])
    print("\nmetrics written to results/metrics.json")


if __name__ == "__main__":
    main()
