<div align="center">

# JustRL2

**面向 128k 长上下文数学推理的 PPO：无 LM 头的 critic、按平均奖励初始化的 value head、长度自适应的 GAE λ —— MiniCPM-2B 配方、代码、数据与权重。**

[English](README.md) · [博客](https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn) · [Blog (EN)](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic)

</div>

JustRL2 用 PPO 在数学任务上训练 MiniCPM-2B。它和常规 PPO 的差别集中在 critic 上：critic 是一个独立模型，输出只有一个标量 value head，没有语言模型头；value head 的初始值设为期望平均奖励，加载底模权重之后再重新初始化一次，避免被 LM 头的权重污染。advantage 用逐样本的 GAE λ = k^(1/L)：λ 随回复长度自适应，使终端奖励传回首 token 的比例对任何长度都恒为 k，长回复不再因为 λ 的累积衰减而收不到信号。其余部分是标准 PPO，配 DAPO 式超长惩罚、dynamic sampling 与 partial rollout，运行在 [Miles](https://github.com/radixark/miles) 框架上。

- **博客**：[JustRL-II：用 critic 把小模型推到 128k 推理](https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn)（[English](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic)）—— 完整的方法、实验与消融，本仓库是它的实现。
- **方法**：[`docs/method.md`](docs/method.md) —— 每个组件做什么、代码在哪。
- **复现**：[`docs/reproduce.md`](docs/reproduce.md) —— 拓扑、启动、续训、评测。
- **数据**：[`docs/data.md`](docs/data.md) —— s9 训练集与 AIME 评测集。

## 目录结构

```
justrl2/                   配方层 —— 所有 JustRL2 专属内容
  configs/*.env            全部超参及默认值（发布配置 + 单机调试配置）
  train.sh                 启动器：.env -> miles 参数 -> Ray job
  prepare_data.py          Hugging Face -> jsonl
  prepare_model.sh         Hugging Face -> HF 权重 + Megatron torch_dist 权重
  eval.py                  对 HF 导出权重做离线 AIME 评测
  model_args/, setup/      MiniCPM-2B 的 Megatron 参数；env / ray / 安装辅助脚本
miles/, train.py           框架层（Miles fork；见 third_party/README.md）
third_party/               三个子模块（Megatron-LM / mbridge / sglang）的 pin 与 patch
tools/                     HF <-> torch_dist 转换脚本
examples/value_head_demo.py   CPU 玩具实验：value head bias 初始化 0 vs 0.52
tests/                     GAE λ、value head 初始化、chunked GAE
```

## 快速开始

```bash
# 0. 环境：Linux、CUDA、Python >= 3.10，至少一台 H100 级别节点
pip install -r requirements.txt
#    另需三个子模块（Megatron-LM、mbridge、sglang）-> third_party/README.md

# 1. 权重与数据（Hugging Face）
bash   justrl2/prepare_model.sh          # -> ./models
python justrl2/prepare_data.py           # -> ./datasets

# 2. 训练（参考实验为 16 节点 x 8 卡；每个节点都执行）
bash justrl2/train.sh justrl2/configs/minicpm-2b-math-128k.env

# 3. 评测某个导出的 checkpoint
python justrl2/eval.py --model runs/justrl2_minicpm_2b_math128k/hf/iter_0000299 \
    --data datasets/aime-2025.jsonl --data datasets/aime-2026.jsonl --n 16
```

所有超参都在 [`justrl2/configs/minicpm-2b-math-128k.env`](justrl2/configs/minicpm-2b-math-128k.env)；任何一项都可以从 shell 覆盖（`GAE_LAMBDA_K=0.4 bash justrl2/train.sh …`），不必改文件。

## 最关键的三个数

| 参数 | 值 | 作用 |
|---|---|---|
| `GAE_LAMBDA_K` | 0.5 | λ_i = k^(1/L_i)：无论回复多长，首 token 拿到的终端 credit 比例恒为 k |
| `CRITIC_VALUE_BIAS_INIT` | 0.52 | value head 从平均奖励起步；消除约 25 步的 warmup 过渡期 |
| `NUM_CRITIC_ONLY_STEPS` | 30 | critic 先收敛，再开始更新 policy |

## 没有子模块时能做什么

训练需要 Megatron / SGLang 的 fork。只研究或单测配方本身：

```bash
python examples/value_head_demo.py          # 只依赖 torch
python -m pytest tests/test_gae_lambda_k.py tests/test_critic_value_bias_init.py tests/test_chunked_gae.py
```

## 引用

使用本代码、数据或配方，请引用博客：

```bibtex
@misc{justrl2_2026,
  title  = {JustRL-II: Scaling Small LLMs to 128k Reasoning with a Critic},
  author = {Pan, Haoxuan and others},
  year   = {2026},
  howpublished = {\url{https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic}},
  note   = {Chinese version: \url{https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn}}
}
```

## 许可

Apache-2.0（框架与配方代码）。模型权重与数据集以各自 Hugging Face 页面上的许可为准。
