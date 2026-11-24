# Masreyyat (مصريات)

Egyptian Arabic NLP, measured honestly. v0.1 is a sentiment baseline trained on
Arabic hotel reviews and then stress-tested on Egyptian-dialect tweets it has
never seen. The gap between those two numbers is the whole point of the
project: Egyptian dialect is under-served, and this repo tracks how each
improvement closes that gap.

## Results (v0.1 baseline)

Model: character 3-5-gram TF-IDF + logistic regression. Trains in ~15 seconds
on a laptop. Full details in `results/metrics.json`.

| Evaluation | Data | n | Accuracy | Macro F1 |
|---|---|---|---|---|
| In-domain | HARD hotel reviews (10% held out) | 10,570 | **0.944** | **0.944** |
| Transfer (zero-shot) | ArSAS Egyptian tweets | 11,784 | **0.540** | **0.531** |

The transfer row is not a failure to hide. It is the measurement this project
exists to improve. The confusion matrix shows the failure mode clearly: on
tweets the model over-predicts positive (pos recall 0.91) and misses most
real negatives (neg recall 0.32). The polite, formulaic praise of hotel
reviews taught it signals that do not exist in Egyptian Twitter language.

## Why this matters

Most Arabic NLP models are trained on Modern Standard Arabic or review sites.
Egyptian Arabic, the most widely understood Arabic dialect, differs in
vocabulary, spelling, and grammar. A model that looks great on reviews
(94%) can be nearly useless on real Egyptian text (54%). Masreyyat measures
that honestly and improves it step by step.

## Data

- **HARD** (Hotel Arabic Reviews Dataset, balanced subset): 105,698 reviews,
  binary sentiment, perfectly balanced (52,849 negative / 52,849 positive).
  Elnagar et al. Used for training and in-domain evaluation.
- **ArSAS** (Arabic Speech-act and Sentiment corpus of tweets): 19,897 tweets.
  Binary subset (11,784: 7,384 neg / 4,400 pos) used as an unseen transfer
  test. Tweets cover Egyptian topics and heavy Egyptian dialect.
- Provenance, citation requirements, and the plan for collecting
  Egyptian-dialect data (including YouTube comments via the official Data
  API) are in `DATA_PLAN.md`.

## Reproduce

```bash
pip install -r requirements.txt
# 1. get the data (see DATA_PLAN.md for links)
git clone https://github.com/elnagara/HARD-Arabic-Dataset data/hard
unzip data/hard/data/balanced-reviews.zip -d data/hard/data/balanced
# download ArSAS parquet from huggingface.co/datasets/arbml/ArSAS into data/
# 2. prepare
python src/prepare_data.py
# 3. train + evaluate (writes results/metrics.json, models/baseline_charlogreg.joblib)
cd src && python train_baseline.py
```

## Repo layout

```
src/arabic.py           shared Arabic normalization (diacritics, alef/yaa, tatweel)
src/prepare_data.py     raw downloads -> data/processed/*.csv
src/train_baseline.py   trains the baseline, evaluates both settings, saves model
results/metrics.json    every number reported above, with confusion matrices
DATA_PLAN.md            what data, from where, under what terms, and what's next
```

## Roadmap

- [ ] v0.2: fine-tune a small subset of HARD-style labels on ArSAS tweets
      (few-shot domain adaptation) and measure the transfer gain
- [ ] v0.3: collect Egyptian YouTube comments via the official Data API
      (respecting quota and ToS, see DATA_PLAN.md), label a seed set
- [ ] v0.4: compare the char-ngram baseline against a fine-tuned
      Arabic transformer (e.g. AraBERT/MARBERT) on the same splits
- [ ] Egyptian dialect identification (Egyptian vs other Arabic) as a
      second task

## Honest limitations

- HARD ratings 1-2 are treated as negative and 4-5 as positive; the balanced
  release contains no 3-star reviews, so neutral review language is untested.
- ArSAS has no dialect column; the Egyptian character of the test set comes
  from its topics and visible dialect usage, not a metadata filter.
- No hyperparameter search was done for v0.1; C=2.0 was picked once and kept.
