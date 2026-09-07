# Copyright 2026 Miles contributors
#
# RLVR-IFEval rule-based reward for GRPO. Matches verl's
# ``rlvrifeval_compute_score`` (``verl/utils/reward_score/ifeval_task_verify.py``):
# ``extract_non_reasoning_content`` from OpenCompass, then ``IFEvalVerifierOld``
# from ``open_instruct.ground_truth_utils``.
#
# Requires: pinned ``opencompass`` submodule on ``PYTHONPATH`` (or editable install),
# ``mmengine`` (pulled by ``opencompass.utils``), and ``open_instruct`` on ``PYTHONPATH``.

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_extract_non_reasoning_content = None
_if_evaluator = None


def _lazy_init_rlvr_ifeval_scoring() -> None:
    """Load OpenCompass ``extract_non_reasoning_content`` and ``IFEvalVerifierOld`` once.

    Semantics match verl ``rlvrifeval_compute_score``; Miles does not import verl.
    """
    global _extract_non_reasoning_content, _if_evaluator
    if _if_evaluator is not None:
        return
    try:
        from opencompass.utils.text_postprocessors import extract_non_reasoning_content
        from open_instruct.ground_truth_utils import IFEvalVerifierOld
    except ImportError as e:
        raise ImportError(
            "rm type 'rlvr_ifeval' requires verl-aligned deps: "
            "``pip install mmengine`` (or your OpenCompass env), "
            "``opencompass`` on PYTHONPATH (submodule at <repo>/opencompass/), "
            "and parent of ``open_instruct`` on PYTHONPATH. Original error: "
            f"{e}"
        ) from e
    _extract_non_reasoning_content = extract_non_reasoning_content
    _if_evaluator = IFEvalVerifierOld()


def compute_rlvr_ifeval_reward(
    response: str,
    label: str | None,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Return a scalar reward in [0, 1] for RLVR-IFEval style constraints.

    ``label`` must be the JSON string stored in ``reward_model.ground_truth`` (func_name + args).
    ``metadata`` is reserved (e.g. future fusion_think); currently ignored.
    """
    del metadata
    if label is None:
        return 0.0
    if not isinstance(label, str):
        label = str(label)
    try:
        _lazy_init_rlvr_ifeval_scoring()
        pred = _extract_non_reasoning_content(response or "")
        result = _if_evaluator(
            prediction=pred,
            label=label,
            tokenized_prediction=None,
        )
        print(f"wsctest=====>>>>>>result: {float(result.score)}")
        return float(result.score)
    except Exception:
        logger.exception(
            "rlvr_ifeval scoring failed (response_len=%s label_len=%s)",
            len(response or ""),
            len(label),
        )
        return 0.0
