# Data (train / test)

The #12 train set is `s9` — an internal **32,412-question** filtered subset of a larger
math corpus. It is **not redistributable** in this package. This folder documents its shape
and gives you a public stand-in so the pipeline runs end-to-end.

## What `s9` is

- A math question-answer set derived by filtering a big math corpus down to a **hard,
  answerable, chain-of-thought-friendly** subset. The filtering is roughly: keeep questions
  whose ground-truth answer is a single value that `math-verify` agrees on, drop trivial /
  degenerate entries, balance difficulty.
- `s9` is the *training* set for #12. The eval set is AIME 2024/2025/2026.

## Since you can't get the exact set

Use the **public** DAPO-style math mix as a close stand-in, the same one the entry script
defaults to:

```bash
export DATA_DIR=$PWD/datasets
# dapo-math-17k + deepmath + deepscaler  (all public on HuggingFace / corresponding repos)
export TRAIN_FILE="$DATA_DIR/dapo-math-17k.jsonl $DATA_DIR/deepmath.jsonl $DATA_DIR/deepscaler.jsonl"
export TEST_FILE="aime2026 $DATA_DIR/aime-2026.jsonl aime2025 $DATA_DIR/aime-2025.jsonl aime2024 $DATA_DIR/aime-2024.jsonl"
```

Each file is a `.jsonl` with a `prompt` (and optional `label`) field per line — the entry
uses `--input-key prompt --label-key label`.

## AIME eval data

AIME 2024/2025/2026 are public benchmarks; convert each to a `.jsonl` of
`{"prompt": ..., "label": "<answer>"}` and reference by name + path in `TEST_FILE`:
`aime2026 <path>/aime-2026.jsonl`.

## The `s9` filtering logic (for reference)

If you want to reproduce *the idea* of `s9` rather than an exact set: take a broad math
corpus (`deepmath`, `deepscaler`, `dapo-math-17k`, `numina`), keep only questions passing a
`math-verify` ground-truth check, drop the bottom/top difficulty tails, and cap at ~30k.
The exact 32412 selection and the exact filter are model-best internal and not released.
