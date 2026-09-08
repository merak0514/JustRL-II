<div align="center">

# JustRL2

**PPO for 128k-context math reasoning with a head-less critic, a mean-reward-seeded value
head and a length-adaptive GAE λ — MiniCPM-2B recipe, code, data and weights.**

[中文](README-cn.md) · [Blog post](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic) · [博客（中文）](https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn)

</div>

- **Blog**: [JustRL-II: Scaling Small LLMs to 128k Reasoning with a Critic](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic) — the full write-up, experiments and ablations this code implements.
- **Method**: [`docs/method.md`](docs/method.md) — what each piece does and where it lives.
- **Reproduce**: [`docs/reproduce.md`](docs/reproduce.md) — topology, launch, resume, eval.
- **Data**: [`docs/data.md`](docs/data.md) — the UltraData-RL-Math-2609 training set and AIME eval sets.

## Layout

```
justrl2/                   the recipe — everything JustRL2-specific
  configs/*.env            all knobs with defaults (release + 1-node debug)
  train.sh                 launcher: .env -> miles arguments -> Ray job
  prepare_data.py          Hugging Face -> jsonl
  prepare_model.sh         Hugging Face -> HF ckpt + Megatron torch_dist ckpt
  eval.py                  offline AIME eval of an HF export
  model_args/, setup/      MiniCPM-2B Megatron args; env / ray / setup helpers
miles/, train.py           the framework (Miles fork; see third_party/README.md)
third_party/               Megatron-LM / SGLang: community image, patches for stock Megatron
Dockerfile                 FROM radixark/miles:dev (Megatron-LM + SGLang + TE preinstalled)
tools/                     HF <-> torch_dist converters
examples/value_head_demo.py   CPU toy: bias 0 vs 0.52 value-head init
tests/                     GAE λ, value-head init, chunked GAE
```

## Quick start

```bash
# 0. environment: the community Miles image has Megatron-LM, SGLang, TE, Ray preinstalled
docker build -t justrl2 . && docker run --gpus all --ipc=host --network=host -it justrl2
#    (bare-metal alternative: third_party/README.md)

# 1. weights and data (Hugging Face)
bash   justrl2/prepare_model.sh          # -> ./models
python justrl2/prepare_data.py           # -> ./datasets

# 2. train (16 nodes x 8 GPU for the reference run; run on every node)
bash justrl2/train.sh justrl2/configs/minicpm-2b-math-128k.env

# 3. evaluate an export
python justrl2/eval.py --model runs/justrl2_minicpm_2b_math128k/hf/iter_0000299 \
    --data datasets/aime-2025.jsonl --data datasets/aime-2026.jsonl --n 16
```

All hyper-parameters are in [`justrl2/configs/minicpm-2b-math-128k.env`](justrl2/configs/minicpm-2b-math-128k.env);
any of them can be overridden from the shell (`GAE_LAMBDA_K=0.4 bash justrl2/train.sh …`).

## The three numbers that matter

| knob                     | value | why                                                                                          |
| ------------------------ | ----- | -------------------------------------------------------------------------------------------- |
| `GAE_LAMBDA_K`           | 0.5   | λ_i = k^(1/L_i): constant terminal-credit fraction k at the first token regardless of length |
| `CRITIC_VALUE_BIAS_INIT` | 0.52  | value head starts at the mean reward; removes the ~25-step warmup transient                  |
| `NUM_CRITIC_ONLY_STEPS`  | 30    | critic converges before the first policy update                                              |

## Without a GPU stack

Megatron-LM and SGLang are needed to *train*. To study or unit-test the recipe itself:

```bash
python examples/value_head_demo.py          # torch only
python -m pytest tests/test_gae_lambda_k.py tests/test_critic_value_bias_init.py tests/test_chunked_gae.py
```

## Citation

If you use this code, data or the recipe, please cite the blog post:

```bibtex
@misc{justrl2_2026,
  title  = {JustRL-II: Scaling Small LLMs to 128k Reasoning with a Critic},
  author = {Pan, Haoxuan and others},
  year   = {2026},
  howpublished = {\url{https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic}},
  note   = {Chinese version: \url{https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn}}
}
```

## License

Apache-2.0 (framework and recipe code). Model weights and datasets carry their own
licenses on their Hugging Face pages.
