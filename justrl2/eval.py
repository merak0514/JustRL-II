#!/usr/bin/env python3
"""Offline AIME evaluation of a JustRL2 checkpoint (the HF exports under
<SAVE_DIR>/hf/iter_XXXXXXX, or the base model).

    python justrl2/eval.py --model runs/<EXP_TAG>/hf/iter_0000299 \
        --data datasets/aime-2025.jsonl [--data datasets/aime-2026.jsonl ...] \
        --n 16 --temperature 1.0 --top-p 0.95 --max-tokens 126976

This reproduces the eval setting used for the paper numbers (16 samples per problem,
T=1.0, top-p 0.95, 126976-token budget) with sglang's offline engine and the same
`math` grader miles uses in training (rule-based + math-verify fallback). Reports
mean accuracy over samples (acc@n), pass@n, mean response length and the truncation
rate per file. Needs the sglang submodule on PYTHONPATH (see third_party/README.md)
and one GPU (or more with --tp).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir (hf/iter_XXXXXXX or base)")
    ap.add_argument("--data", action="append", required=True, help="jsonl with prompt/label; repeatable")
    ap.add_argument("--n", type=int, default=16, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=126976)
    ap.add_argument("--context-len", type=int, default=131072)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--out", default=None, help="write per-sample records to this jsonl")
    args = ap.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    from miles.rollout.rm_hub.math_utils import grade_answer_union

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = sgl.Engine(
        model_path=args.model,
        trust_remote_code=True,
        tp_size=args.tp,
        context_length=args.context_len,
        mem_fraction_static=args.mem_fraction,
        json_model_override_args=json.dumps({"max_position_embeddings": args.context_len}),
        log_level="warning",
    )
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_tokens}

    out_f = open(args.out, "w") if args.out else None
    for data_path in args.data:
        rows = load_jsonl(Path(data_path))
        prompts = []
        for row in rows:
            p = row["prompt"]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        prompts_rep = [p for p in prompts for _ in range(args.n)]
        outputs = engine.generate(prompts_rep, sampling)

        correct, lengths, truncated = [], [], []
        per_problem = [[] for _ in rows]
        for i, o in enumerate(outputs):
            q = i // args.n
            text = o["text"]
            ok = bool(grade_answer_union(text, str(rows[q]["label"])))
            n_tok = o["meta_info"]["completion_tokens"]
            trunc = o["meta_info"].get("finish_reason", {}).get("type") == "length"
            correct.append(ok)
            lengths.append(n_tok)
            truncated.append(trunc)
            per_problem[q].append(ok)
            if out_f:
                out_f.write(json.dumps({"file": data_path, "idx": q, "sample": i % args.n, "correct": ok,
                                        "tokens": n_tok, "truncated": trunc, "response": text},
                                       ensure_ascii=False) + "\n")
        acc = statistics.mean(correct)
        pass_n = statistics.mean(any(v) for v in per_problem)
        print(f"{Path(data_path).name}: {len(rows)} problems x {args.n}  "
              f"acc@{args.n}={acc:.4f}  pass@{args.n}={pass_n:.4f}  "
              f"mean_len={statistics.mean(lengths):.0f}  truncated={statistics.mean(truncated):.4f}")
    if out_f:
        out_f.close()
    engine.shutdown()


if __name__ == "__main__":
    main()
