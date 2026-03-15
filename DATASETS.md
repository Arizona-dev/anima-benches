# Datasets

This repository does not bundle benchmark datasets by default.

## LoCoMo

- Upstream dataset:
  `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`
- `locomo/run.py`, `locomo/run_think.py`, and `locomo/compare_servers.py`
  auto-download the dataset when the configured path is missing.

## LongMemEval

- Upstream dataset:
  `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json`
- `longmemeval/run.py` auto-downloads the dataset when the configured path is
  missing.

## Credentials

The harnesses expect API credentials through environment variables or an
optional local `.env` file that is not tracked in Git.
