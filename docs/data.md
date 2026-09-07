# Data

Both datasets are on the Hugging Face Hub; `python justrl2/prepare_data.py` downloads them
and writes the jsonl files `justrl2/train.sh` reads.

## Training set — `openbmb/JustRL2-math-s9`

32,412 math problems with a single verifiable final answer ("s9"), drawn from public math
corpora (DAPO-Math-17k, DeepMath, DeepScaleR and similar). The set is **label-audited, not
difficulty-filtered**: every problem's reference answer was checked by two strong models
solving it independently, and problems with ambiguous, multiple-choice or disputed answers
were dropped. Under the base model at 8 samples per problem about 63 % of s9 is solved
8/8, 30 % is mixed and 7 % is solved 0/8; the mixed band (~12k problems) is where the
learning signal is, and `DYNAMIC_SAMPLING=1` selects it at rollout time by discarding
zero-variance groups.

Columns: `prompt` (the problem, plain text), `label` (final answer string, as it should
appear inside `\boxed{}`), `source`, `id`. `train.sh` passes `prompt` through the model's
chat template as a single user turn and grades responses against `label` with miles'
`math` reward (rule-based normalisation with math-verify as a fallback).

## Evaluation — `openbmb/JustRL2-aime-eval`

Splits `aime2024`, `aime2025`, `aime2026` (30 problems each), same columns. The paper
reports the mean of AIME 2025 and 2026 at n = 16 samples per problem.

## Using your own data

Any jsonl with `prompt` and `label` works:

```bash
export TRAIN_FILE="/path/a.jsonl /path/b.jsonl"          # space-separated, all mixed
export TEST_FILE="mytest /path/test.jsonl"                # name path [name path ...]
```

Labels must be gradable by the math verifier (a number, expression or short answer).
Filtering out problems the base model gets 8/8 right matters more than the exact corpus:
with `DYNAMIC_SAMPLING=1` those groups are discarded at rollout time anyway, so leaving
them in only wastes generation budget.
