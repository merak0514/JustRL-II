#!/usr/bin/env python3
"""Download the JustRL2 training data from the Hugging Face Hub and write the jsonl
that `justrl2/train.sh` reads.

    python justrl2/prepare_data.py [--data-dir datasets]

Training set: the **Math** slice of `openbmb/UltraData-RL-2609` (32,412 verifiable
math problems) — the corpus the JustRL2 runs were trained on. It is written to
`<data-dir>/UltraData-RL-Math-2609.jsonl`.

The Hub schema is `{uuid, query, ground_truth, source, domain}`; miles expects
`prompt` (the question, wrapped as a single user turn through the chat template) and
`label` (the reference answer, graded by the `math` reward with math-verify as a
fallback). The mapping is done here, and the original columns are preserved.

Evaluation sets (AIME 2024 / 2025 / 2026) are public benchmarks that are not part of
this dataset — see `--eval-repo` below and docs/data.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_REPO = "openbmb/UltraData-RL-2609"
TRAIN_CONFIG = "Math"
TRAIN_OUT = "UltraData-RL-Math-2609.jsonl"


def _dump(ds, path: Path, prompt_key: str, label_key: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    with path.open("w") as f:
        for row in ds:
            row = dict(row)
            prompt = row.get(prompt_key)
            label = row.get(label_key)
            if not prompt or label is None or str(label).strip() == "":
                skipped += 1
                continue
            # keep the original columns (uuid/source/domain) as metadata
            row["prompt"] = prompt
            row["label"] = str(label)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if skipped:
        print(f"  ({skipped} rows skipped: empty query or ground_truth)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--train-repo", default=TRAIN_REPO)
    ap.add_argument("--train-config", default=TRAIN_CONFIG, help="dataset config (Math/Code/Knowledge/Long-Context)")
    ap.add_argument("--prompt-key", default="query", help="Hub column holding the question")
    ap.add_argument("--label-key", default="ground_truth", help="Hub column holding the answer")
    ap.add_argument("--out-name", default=TRAIN_OUT)
    ap.add_argument(
        "--eval-repo",
        default=None,
        help="optional HF dataset with AIME-style eval problems; needs --eval-splits. "
        "Not set by default: supply your own aime-20XX.jsonl (see docs/data.md).",
    )
    ap.add_argument(
        "--eval-splits",
        default="aime2024,aime2025,aime2026",
        help="comma-separated splits (or configs) to pull from --eval-repo",
    )
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    out = Path(args.data_dir)
    if not args.skip_train:
        ds = load_dataset(args.train_repo, args.train_config, split="train")
        n = _dump(ds, out / args.out_name, args.prompt_key, args.label_key)
        print(f"train: {n} rows -> {out / args.out_name}")

    if args.eval_repo:
        for split in [s.strip() for s in args.eval_splits.split(",") if s.strip()]:
            try:
                ds = load_dataset(args.eval_repo, split=split)
            except Exception:
                ds = load_dataset(args.eval_repo, split, split="train")
            fname = f"{split.replace('aime', 'aime-')}.jsonl"
            n = _dump(ds, out / fname, args.prompt_key, args.label_key)
            print(f"{split}: {n} rows -> {out / fname}")
    else:
        print(
            "\neval: not downloaded (no --eval-repo). train.sh expects AIME jsonl files\n"
            f"      at {out}/aime-2024.jsonl, aime-2025.jsonl, aime-2026.jsonl with the\n"
            "      same prompt/label fields — see docs/data.md."
        )

    print(f"\nexport DATA_DIR={out.resolve()}    # train.sh defaults derive from this")


if __name__ == "__main__":
    main()
