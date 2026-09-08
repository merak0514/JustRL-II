# Data

## Training set — `openbmb/UltraData-RL-2609`, config `Math`

JustRL2 trains on the **Math** slice of
[UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609):
**32,412** competition- and textbook-style problems, each with a single extractable
answer verified against `ground_truth`. `python justrl2/prepare_data.py` downloads it
and writes `<data-dir>/UltraData-RL-Math-2609.jsonl`.

The Hub schema is five fields:

| field | meaning |
|---|---|
| `uuid` | `Math_00001`-style id |
| `query` | the problem statement |
| `ground_truth` | the reference answer |
| `source` | provenance tag |
| `domain` | `Math` |

`prepare_data.py` adds `prompt` (= `query`) and `label` (= `ground_truth`) because that
is what miles reads (`--input-key prompt --label-key label`); the original columns are
kept as metadata. `prompt` goes through the model's chat template as a single user turn,
and responses are graded by miles' `math` reward (rule-based normalisation with
math-verify as a fallback).

### Why the difficulty distribution matters here

The dataset card documents that difficulty is calibrated **against the RL initialization
checkpoint** — the same `openbmb/JustRL-II-base-model` this recipe starts from: items it
already solves at pass rate 1 are dropped (no gradient), the learnable band is kept, and
pass-rate-0 items with a confirmed-valid label are kept and left to online dynamic
sampling. That is why `DYNAMIC_SAMPLING=1` is part of the recipe: it discards
zero-variance groups at rollout time, so the remaining budget goes to problems that still
produce a learning signal. Labels are never modified by that filtering.

If you swap in your own corpus, the property to preserve is that one: drop what the
starting checkpoint already solves every time, keep the mixed band.

## Evaluation — AIME 2024 / 2025 / 2026

The eval sets are public competition benchmarks and are **not** part of
UltraData-RL-2609, so `prepare_data.py` does not download them by default. Provide three
jsonl files with the same `prompt` / `label` fields:

```
<data-dir>/aime-2024.jsonl
<data-dir>/aime-2025.jsonl
<data-dir>/aime-2026.jsonl
```

and `train.sh` picks them up (`TEST_FILE="aime2024 … aime2025 … aime2026 …"`). Several
AIME sets are mirrored on the Hub; if the one you use is laid out as splits or configs of
a single repo, `prepare_data.py --eval-repo <repo> --eval-splits aime2024,aime2025,aime2026`
converts them for you. The reported numbers use 30 problems per year, 16 samples per
problem, T=1.0, top-p 0.95, 126976-token budget (`justrl2/eval.py`).

## Using your own data

Any jsonl with `prompt` and `label` works:

```bash
export TRAIN_FILE="/path/a.jsonl /path/b.jsonl"     # space-separated, mixed together
export TEST_FILE="mytest /path/test.jsonl"           # name path [name path ...]
```

Labels must be gradable by the math verifier — a number, expression, or short answer.
