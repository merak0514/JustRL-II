import logging
import os
import re

from .f1 import normalize_answer

logger = logging.getLogger(__name__)

# Whitespace tokenization makes Chinese (no spaces) a single token; any mismatch → LCS=0. Use chars for CJK text.
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

# Same markers as deepscaler / gpqa: score only the final answer, not chain-of-thought.
_THINK_END = "</think>"
_RESPONSE_MARKER = "###Response"


def extract_solution_text_for_rouge(response: str | None) -> str | None:
    """
    Remove reasoning blocks before ROUGE-L, consistent with deepscaler reward logic.

    - If ``</think>`` is present (Qwen-style think end), use the segment after the last one.
    - Else if ``###Response`` is present, use the text after the first marker.
    - Else use the full string (plain outputs without think wrappers still score normally).
    """
    if response is None:
        return None
    text = str(response)
    if _THINK_END in text:
        text = text.rsplit(_THINK_END, 1)[-1]
    elif _RESPONSE_MARKER in text:
        text = text.split(_RESPONSE_MARKER, 1)[1]
    return text.strip()


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence of token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def _rouge_l_tokens(norm: str) -> list[str]:
    """Word tokens for Latin text; character tokens when CJK is present (typical for Chinese QA)."""
    if _CJK_RE.search(norm):
        return list(norm.replace(" ", ""))
    return norm.split()


def rouge_l_f1(prediction: str | None, ground_truth) -> tuple[float, float, float]:
    """
    ROUGE-L F-measure from LCS after ``normalize_answer`` (same as f1 reward).

    Tokenization: whitespace-split for Latin-only text; **character sequence** when CJK is
    present, so Chinese strings are not treated as a single mismatched word.

    Returns (f1, recall_l, precision_l). Recall = LCS/ref_len, precision = LCS/pred_len.
    """
    if prediction is None:
        return 0.0, 0.0, 0.0

    norm_pred = normalize_answer(str(prediction))
    norm_ref = normalize_answer(str(ground_truth))

    pred_tokens = _rouge_l_tokens(norm_pred)
    ref_tokens = _rouge_l_tokens(norm_ref)

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0, 0.0, 0.0

    recall = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # Reward runs in Ray/SGLang workers: plain print often does not appear on the job-submit terminal.
    if os.environ.get("MILES_DEBUG_ROUGE_L", "").lower() in ("1", "true", "yes"):
        logger.info("rouge_l_f1 recall=%s precision=%s f1=%s", recall, precision, f1)

    return f1, recall, precision


def compute_rouge_l_reward(response: str | None, label) -> float:
    """Scalar reward: ROUGE-L F1 in [0, 1], after stripping think / ###Response like deepscaler."""
    body = extract_solution_text_for_rouge(response)
    if not body:
        return 0.0
    return rouge_l_f1(body, label)[0]
