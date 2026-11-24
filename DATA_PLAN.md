# Masreyyat data plan

What data this project uses, where it comes from, under what terms, and how
future collection will respect source terms.

## Currently used

### HARD - Hotel Arabic Reviews Dataset (training + in-domain test)
- Source: https://github.com/elnagara/HARD-Arabic-Dataset
- Paper: Elnagar, Khalifa, Einea, "Hotel Arabic-Reviews Dataset Construction
  for Sentiment Analysis Applications" (2018).
- Size used: balanced subset, 105,698 reviews (ratings 1-2 = neg, 4-5 = pos).
- Terms: published for academic/research use with citation requested; no
  explicit license file in the repo. OK for a research portfolio project with
  citation. If this project is ever used commercially, contact the authors
  first.
- Citation is included in the README.

### ArSAS - Arabic Speech-act and Sentiment corpus of tweets (transfer test)
- Source: https://huggingface.co/datasets/arbml/ArSAS
- Paper: Elmadany et al., "ArSAS: An Arabic Speech-Act and Sentiment Corpus
  of Tweets" (LREC 2018).
- Size used: 19,897 tweets total; 11,784 in the binary neg/pos subset.
- Terms: academic dataset, citation requested. Tweet text redistribution
  follows the dataset authors' own release; we do not re-publish the raw
  tweets in this repo, only IDs/labels derived at prepare time.
- Note: tweets were collected via the Twitter API under its terms; use for
  research/ML training evaluation only.

## Planned: Egyptian YouTube comments (v0.3)

Goal: a fresh, dialect-heavy Egyptian corpus that is not review or news text.

Rules:
- Collect ONLY via the official YouTube Data API v3 (`commentThreads.list`),
  no scraping. This keeps collection inside YouTube's Terms of Service.
- Respect the daily quota (10,000 units/day on a free key;
  `commentThreads.list` costs 1 unit per request, 100 comments max per
  request, so ~1M comments/day theoretical max, far more than needed).
- Public comments only. No private data, no author data beyond the public
  display name; store comment ID + text + video ID, drop author fields.
- Target channels: large Egyptian creators across comedy, football, cooking,
  and commentary, so the corpus is not one community's slang.
- Label a seed set manually (~2,000 comments, sentiment neg/neu/pos), then
  use the v0.1 model for pre-labeling with human review (active learning).
- Store the labeling guideline in the repo so labels are reproducible.
- Keep raw API responses out of git; commit only the labeled seed set and
  the collection script.

## Deliberately avoided

- Scraping Facebook groups or pages (against Meta ToS).
- Downloading "Egyptian dialect dumps" of unknown provenance from file
  shares; several circulating dumps are mixes of copyrighted forums scraped
  without permission.
- Re-identifying or profiling comment authors.
