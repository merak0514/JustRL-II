#!/usr/bin/env python3
"""Download the JustRL2 datasets from the Hugging Face Hub and write the jsonl files
that `justrl2/train.sh` reads.

    python justrl2/prepare_data.py [--data-dir datasets]

Produces:
    <data-dir>/justrl2-math-s9.jsonl      training set   (openbmb/JustRL2-math-s9)
    <data-dir>/aime-2024.jsonl            eval           (openbmb/JustRL2-aime-eval, split aime2024)
    <data-dir>/aime-2025.jsonl                                                       aime2025
    <data-dir>/aime-2026.jsonl                                                       aime2026

Each line is {"prompt": <str or chat messages>, "label": <answer str>, ...extra columns}.
miles reads `prompt` (wrapped as a single user turn and passed through the chat template)
and `label` (the ground-truth answer, graded by the `math` reward with math-verify as
fallback). Any other columns are kept as metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_REPO = "openbmb/JustRL2-math-s9"
EVAL_REPO = "openbmb/JustRL2-aime-eval"
EVAL_SPLITS = {"aime2024": "aime-2024.jsonl", "aime2025": "aime-2025.jsonl", "aime2026": "aime-2026.jsonl"}


def _dump(ds, path: Path, prompt_key: str, label_key: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in ds:
            row = dict(row)
            if prompt_key != "prompt":
                row["prompt"] = row.pop(prompt_key)
            if label_key != "label":
                row["label"] = row.pop(label_key)
            if row["label"] is None or str(row["label"]).strip() == "":
                continue
            row["label"] = str(row["label"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--train-repo", default=TRAIN_REPO)
    ap.add_argument("--eval-repo", default=EVAL_REPO)
    ap.add_argument("--prompt-key", default="prompt", help="column holding the question")
    ap.add_argument("--label-key", default="label", help="column holding the answer")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    out = Path(args.data_dir)
    if not args.skip_train:
        ds = load_dataset(args.train_repo, split="train")
        n = _dump(ds, out / "justrl2-math-s9.jsonl", args.prompt_key, args.label_key)
        print(f"train: {n} rows -> {out / 'justrl2-math-s9.jsonl'}")
    if not args.skip_eval:
        for split, fname in EVAL_SPLITS.items():
            ds = load_dataset(args.eval_repo, split=split)
            n = _dump(ds, out / fname, args.prompt_key, args.label_key)
            print(f"{split}: {n} rows -> {out / fname}")

    print(
        "\nNow export for train.sh (or leave the defaults, which point here):\n"
        f"  export DATA_DIR={out.resolve()}"
    )


if __name__ == "__main__":
    main()
