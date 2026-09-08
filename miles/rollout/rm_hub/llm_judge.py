"""LLM-as-Judge reward model (Arena Hard / GenRM style).

Ports the GenRM Arena Hard evaluation logic from verl to miles. Key points:
- Arena Hard style system+user prompt template (---SYSTEM--- / ---USER--- separators)
- Position debiasing via two queries with A/B swapped, weighted scoring (strong verdicts count 3x)
- Both the model response and the judge response go through extract_non_reasoning_content
  to strip <think> blocks
- Judge calls have a timeout and up to 3 retries with exponential backoff; retryable HTTP
  statuses (400/408/429/500/502/503/504, ...) back off and retry
- If the retries still fail, ``llm_judge_reward`` / ``self_judge_reward`` drops the sample
  (``remove_sample=True``) instead of aborting the whole rollout
- Judge API parameters (temperature, max_tokens, ...) are passed through with
  --judge-kwargs 'key=value,...'

Sends the model's response and a reference response to a judge LLM via an
OpenAI-compatible /v1/chat/completions endpoint.  Supports two modes:

* **llm_judge** -- calls an *external* API (e.g. Kimi) specified by ``--rm-url``.
* **self_judge** -- calls the *local* SGLang router that already serves the
  model being trained, so the model judges its own outputs.

Both modes use double-query position debiasing (A/B swapped) with weighted
scoring ported from verl's GenRM Arena Hard implementation.

* **llm_rubric_judge** / **self_rubric_judge** -- single-call rubric scoring:
  fill a template with ``question_prompt``, ``answer`` (reference), ``score_point``,
  ``predictions`` (model output), ``max_score``; parse JSON ``point_scores`` /
  ``total_score``; reward is ``total_score / max_score`` in ``[0, 1]``.
  Rubric fields are read from ``sample.metadata`` (see :func:`_rubric_metadata_lookup`).
"""

import asyncio
import json
import logging
import re
from functools import lru_cache
from typing import Tuple

import aiohttp

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict regex patterns (verl Arena Hard style)
# ---------------------------------------------------------------------------
VERDICT_PATTERNS = [
    re.compile(r"\[\[([AB<>=]+)\]\]"),
    re.compile(r"- \[([AB<>=]+)\]"),
]

# Weighted scoring: strong verdicts (>>, <<) count 3x.
LABEL_TO_SCORE = {
    "A>>B": [1] * 3,
    "A>B": [1],
    "A=B": [0.5],
    "A<B": [0],
    "A<<B": [0] * 3,
    "B>A": [0],
    "B>>A": [0] * 3,
    "B=A": [0.5],
    "B<A": [1],
    "B<<A": [1] * 3,
}

STRONG_VERDICT_WEIGHT = 3


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_template(path: str) -> Tuple[str | None, str]:
    """Load a judge prompt template and split into (system_prompt, user_template).

    The file may contain ``---SYSTEM---`` and ``---USER---`` section markers.
    If both markers are present the text between them is the system prompt and
    the text after ``---USER---`` is the user template.  Otherwise the entire
    file is treated as a user-only template (system_prompt=None).
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    if "---SYSTEM---" in raw and "---USER---" in raw:
        parts = raw.split("---USER---", 1)
        system_part = parts[0].replace("---SYSTEM---", "", 1).strip()
        user_part = parts[1].strip()
        return system_part, user_part

    return None, raw.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_non_reasoning_content(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from text."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _format_prompt_as_text(prompt) -> str:
    """Convert a prompt (str or list of message dicts) into plain text."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"[{role}]: {content}")
        return "\n".join(parts)
    return str(prompt)


def parse_judge_kwargs(raw: str | None) -> dict:
    """Parse ``key=value,key=value`` string into a dict with numeric coercion."""
    if not raw:
        return {}
    result = {}
    for item in raw.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        result[k] = v
    return result


def _get_score(judgment_text: str) -> str | None:
    """Extract the last verdict token from judge output (verl-compatible).

    Tries ``[[A>>B]]`` bracket format first, then ``- [A>>B]`` list format.
    Returns the last match uppercased, or None.
    """
    text_upper = judgment_text.upper()
    for pattern in VERDICT_PATTERNS:
        matches = [m for m in pattern.findall(text_upper) if m]
        if matches:
            return matches[-1].strip()
    return None


def _compute_reward(score_round1: str | None, score_round2: str | None) -> float:
    """Compute weighted reward from two rounds of pairwise judgment.

    Round 1: A=model, B=reference.
    Round 2: A=reference, B=model (position-swapped).

    Uses verl's weighted averaging where strong verdicts count 3x.
    """
    scores_r2 = LABEL_TO_SCORE.get(score_round2, [])
    scores_r1 = LABEL_TO_SCORE.get(score_round1, [])
    reward_list = list(scores_r2) + [1 - s for s in scores_r1]
    if not reward_list:
        return 0.0
    return sum(reward_list) / len(reward_list)


# ---------------------------------------------------------------------------
# Judge API call
# ---------------------------------------------------------------------------
_JUDGE_TIMEOUT_SECS = 300
_JUDGE_MAX_RETRIES = 3
_JUDGE_RETRY_BACKOFF = 2.0

# Transient gateway / upstream statuses (incl. flaky 400 from some OpenAI-compatible proxies).
_RETRIABLE_JUDGE_HTTP_STATUSES = frozenset({400, 408, 429, 500, 502, 503, 504})


async def _call_judge(
    rm_url: str,
    model: str,
    user_message: str,
    system_message: str | None = None,
    api_key: str | None = None,
    proxy: str | None = None,
    **extra_kwargs,
) -> str:
    """Call the judge LLM via OpenAI-compatible chat completions API.

    Retries up to ``_JUDGE_MAX_RETRIES`` times on timeout or transient
    HTTP errors (see ``_RETRIABLE_JUDGE_HTTP_STATUSES``).  Each retry waits with exponential backoff.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": extra_kwargs.pop("temperature", 0.6),
        "max_tokens": extra_kwargs.pop("max_tokens", 8192),
    }
    payload.update(extra_kwargs)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(total=_JUDGE_TIMEOUT_SECS)
    last_exc: Exception | None = None
    for attempt in range(1, _JUDGE_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(rm_url, json=payload, headers=headers, proxy=proxy) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        if resp.status in _RETRIABLE_JUDGE_HTTP_STATUSES:
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=body[:500],
                            )
                        logger.error("Judge API %s returned %s: %s", rm_url, resp.status, body[:500])
                        resp.raise_for_status()
                    data = await resp.json()
            return data["choices"][0]["message"]["content"]
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as exc:
            last_exc = exc
            logger.warning("Judge call timeout (attempt %d/%d)", attempt, _JUDGE_MAX_RETRIES)
        except aiohttp.ClientResponseError as exc:
            if exc.status not in _RETRIABLE_JUDGE_HTTP_STATUSES:
                raise
            last_exc = exc
            logger.warning(
                "Judge transient HTTP %s (attempt %d/%d): %s",
                exc.status,
                attempt,
                _JUDGE_MAX_RETRIES,
                (exc.message or "")[:200],
            )

        if attempt < _JUDGE_MAX_RETRIES:
            await asyncio.sleep(_JUDGE_RETRY_BACKOFF * attempt)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
async def llm_judge_reward(args, sample: Sample) -> float:
    """Compute reward by querying an external judge LLM with position debiasing.

    Sends two requests with A/B swapped, parses ``[[verdict]]`` from each,
    and computes a weighted reward.
    """
    system_prompt, user_template = _load_template(args.judge_prompt_template)
    prompt_text = _format_prompt_as_text(sample.prompt)
    model_response = extract_non_reasoning_content(sample.response)
    reference = extract_non_reasoning_content(sample.label)
    judge_kwargs = parse_judge_kwargs(getattr(args, "judge_kwargs", None))

    msg_ab = user_template.format(prompt=prompt_text, response_a=model_response, response_b=reference)
    msg_ba = user_template.format(prompt=prompt_text, response_a=reference, response_b=model_response)

    judge_model = getattr(args, "judge_model", "default")
    api_key = getattr(args, "judge_api_key", None)
    proxy = getattr(args, "judge_proxy", None)

    try:
        raw_ab, raw_ba = await asyncio.gather(
            _call_judge(
                args.rm_url,
                judge_model,
                msg_ab,
                system_message=system_prompt,
                api_key=api_key,
                proxy=proxy,
                **judge_kwargs,
            ),
            _call_judge(
                args.rm_url,
                judge_model,
                msg_ba,
                system_message=system_prompt,
                api_key=api_key,
                proxy=proxy,
                **judge_kwargs,
            ),
        )
    except aiohttp.ClientResponseError as e:
        if e.status in (401, 403):
            raise
        logger.warning(
            "llm_judge HTTP error after retries, skipping sample: status=%s body=%s",
            e.status,
            (e.message or "")[:300],
        )
        sample.remove_sample = True
        return 0.0
    except aiohttp.ClientError as e:
        logger.warning("llm_judge connection error, skipping sample: %s", e)
        sample.remove_sample = True
        return 0.0
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.warning("llm_judge unexpected response, skipping sample: %s", e)
        sample.remove_sample = True
        return 0.0

    result_ab = extract_non_reasoning_content(raw_ab)
    result_ba = extract_non_reasoning_content(raw_ba)
    score_ab = _get_score(result_ab)
    score_ba = _get_score(result_ba)

    parse_failed = False
    if score_ab is None:
        parse_failed = True
        logger.warning(
            "Failed to parse verdict from judge (round1 A=model) head=%s ... tail=%s",
            result_ab[:200], result_ab[-200:],
        )
    if score_ba is None:
        parse_failed = True
        logger.warning(
            "Failed to parse verdict from judge (round2 A=reference) head=%s ... tail=%s",
            result_ba[:200], result_ba[-200:],
        )

    if parse_failed:
        sample.remove_sample = True
        return 0.0

    return _compute_reward(score_ab, score_ba)


async def self_judge_reward(args, sample: Sample) -> float:
    """Compute reward using the training model itself as judge.

    Same position-debiased comparison as :func:`llm_judge_reward`, but the
    judge endpoint is the local SGLang router instead of an external API.
    """
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
    system_prompt, user_template = _load_template(args.judge_prompt_template)
    prompt_text = _format_prompt_as_text(sample.prompt)
    model_response = extract_non_reasoning_content(sample.response)
    reference = extract_non_reasoning_content(sample.label)
    judge_kwargs = parse_judge_kwargs(getattr(args, "judge_kwargs", None))

    msg_ab = user_template.format(prompt=prompt_text, response_a=model_response, response_b=reference)
    msg_ba = user_template.format(prompt=prompt_text, response_a=reference, response_b=model_response)

    try:
        raw_ab, raw_ba = await asyncio.gather(
            _call_judge(url, "default", msg_ab, system_message=system_prompt, **judge_kwargs),
            _call_judge(url, "default", msg_ba, system_message=system_prompt, **judge_kwargs),
        )
    except aiohttp.ClientResponseError as e:
        if e.status in (401, 403):
            raise
        logger.warning(
            "self_judge HTTP error after retries, skipping sample: status=%s body=%s",
            e.status,
            (e.message or "")[:300],
        )
        sample.remove_sample = True
        return 0.0
    except aiohttp.ClientError as e:
        logger.warning("self_judge connection error, skipping sample: %s", e)
        sample.remove_sample = True
        return 0.0
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.warning("self_judge unexpected response, skipping sample: %s", e)
        sample.remove_sample = True
        return 0.0

    result_ab = extract_non_reasoning_content(raw_ab)
    result_ba = extract_non_reasoning_content(raw_ba)
    score_ab = _get_score(result_ab)
    score_ba = _get_score(result_ba)

    parse_failed = False
    if score_ab is None:
        parse_failed = True
        logger.warning(
            "Failed to parse verdict from self-judge (round1 A=model) head=%s ... tail=%s",
            result_ab[:200], result_ab[-200:],
        )
    if score_ba is None:
        parse_failed = True
        logger.warning(
            "Failed to parse verdict from self-judge (round2 A=reference) head=%s ... tail=%s",
            result_ba[:200], result_ba[-200:],
        )

    if parse_failed:
        sample.remove_sample = True
        return 0.0

    return _compute_reward(score_ab, score_ba)


# ---------------------------------------------------------------------------
# Rubric judge (score-point JSON)
# ---------------------------------------------------------------------------


def _rubric_metadata_lookup(metadata: dict | None) -> dict:
    """Resolve rubric fields from dataset-loaded ``sample.metadata``.

    Typical jsonl uses a top-level ``extra_info`` object. Use ``--metadata-key
    extra_info`` so that object becomes ``sample.metadata``. If rows nest
    rubric fields under ``metadata.extra_info``, unwrap that dict.
    """
    if not isinstance(metadata, dict):
        return {}
    if "score_point" in metadata or "max_score" in metadata:
        return metadata
    inner = metadata.get("extra_info")
    if isinstance(inner, dict):
        return inner
    return metadata


def _coerce_positive_int(val) -> int | None:
    if val is None:
        return None
    if hasattr(val, "item") and callable(getattr(val, "item", None)):
        try:
            val = val.item()
        except Exception:
            pass
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val if val > 0 else None
    if isinstance(val, float):
        i = int(val)
        return i if i > 0 else None
    if isinstance(val, str) and val.strip().isdigit():
        i = int(val.strip())
        return i if i > 0 else None
    try:
        i = int(val)
        return i if i > 0 else None
    except (TypeError, ValueError):
        return None


def _extract_json_object_from_text(text: str) -> str | None:
    """Pull a single JSON object from judge output (fenced block or first balanced ``{...}``)."""
    text = extract_non_reasoning_content(text)
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else text.strip()
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : i + 1]
    return None


def _parse_rubric_judge_json(judge_text: str) -> dict | None:
    raw = _extract_json_object_from_text(judge_text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _rubric_reward_from_payload(data: dict, max_score: int) -> tuple[float, str | None]:
    """Return ``(reward, error_tag)``. ``error_tag`` is set when parsing/validation fails."""
    point_scores = data.get("point_scores")
    if not isinstance(point_scores, list):
        return 0.0, "missing_point_scores"
    total = data.get("total_score")
    if total is None:
        try:
            total = sum(int(p.get("score", 0)) for p in point_scores if isinstance(p, dict))
        except (TypeError, ValueError):
            return 0.0, "bad_point_scores"
    else:
        try:
            total = int(total)
        except (TypeError, ValueError):
            return 0.0, "bad_total_score"
    denom = max_score
    if denom <= 0:
        denom = len(point_scores)
    if denom <= 0:
        return 0.0, "max_score_zero"
    reward = float(total) / float(denom)
    return max(0.0, min(1.0, reward)), None


def _format_rubric_user_message(template: str, sample: Sample, meta: dict) -> str:
    question_prompt = _format_prompt_as_text(sample.prompt)
    answer = sample.label if sample.label is not None else ""
    score_point = str(meta.get("score_point") or "")
    predictions = extract_non_reasoning_content(sample.response)
    max_score = _coerce_positive_int(meta.get("max_score"))
    max_score_fmt = max_score if max_score is not None else 0
    return template.format(
        question_prompt=question_prompt,
        answer=answer,
        score_point=score_point,
        predictions=predictions,
        max_score=max_score_fmt,
    )


async def _rubric_judge_once(
    args,
    sample: Sample,
    *,
    rm_url: str,
    judge_model: str,
) -> float:
    template_path = getattr(args, "judge_prompt_template", None)
    if not template_path:
        logger.warning("rubric judge: --judge-prompt-template is not set, skipping sample")
        sample.remove_sample = True
        return 0.0

    system_prompt, user_template = _load_template(template_path)
    meta = _rubric_metadata_lookup(sample.metadata)
    score_point = meta.get("score_point")
    if score_point is None or (isinstance(score_point, str) and not score_point.strip()):
        logger.warning("rubric judge: missing metadata score_point, skipping sample")
        sample.remove_sample = True
        return 0.0

    max_score = _coerce_positive_int(meta.get("max_score"))
    try:
        user_message = _format_rubric_user_message(user_template, sample, meta)
        print(f"[debug]wsctest====>>>>>>>>user_message: {user_message}")
    except (KeyError, ValueError) as e:
        logger.warning("rubric judge: template format failed: %s", e)
        sample.remove_sample = True
        return 0.0

    judge_kwargs = parse_judge_kwargs(getattr(args, "judge_kwargs", None))
    api_key = getattr(args, "judge_api_key", None)
    proxy = getattr(args, "judge_proxy", None)

    try:
        raw = await _call_judge(
            rm_url,
            judge_model,
            user_message,
            system_message=system_prompt,
            api_key=api_key,
            proxy=proxy,
            **judge_kwargs,
        )
    except aiohttp.ClientResponseError as e:
        if e.status in (401, 403):
            raise
        logger.warning(
            "rubric judge HTTP error after retries, skipping sample: status=%s body=%s",
            e.status,
            (e.message or "")[:300],
        )
        sample.remove_sample = True
        return 0.0
    except aiohttp.ClientError as e:
        logger.warning("rubric judge connection error, skipping sample: %s", e)
        sample.remove_sample = True
        return 0.0
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.warning("rubric judge unexpected response, skipping sample: %s", e)
        sample.remove_sample = True
        return 0.0

    parsed = _parse_rubric_judge_json(raw)
    if not parsed:
        logger.warning(
            "rubric judge: failed to parse JSON head=%s ... tail=%s",
            raw[:200],
            raw[-200:],
        )
        sample.remove_sample = True
        return 0.0

    reward, err = _rubric_reward_from_payload(parsed, max_score or 0)
    if err:
        logger.warning("rubric judge: invalid payload (%s) head=%s", err, raw[:200])
        sample.remove_sample = True
        return 0.0
    return reward


async def llm_rubric_judge_reward(args, sample: Sample) -> float:
    """External API rubric judge: one completion, JSON total_score / max_score → reward."""
    judge_model = getattr(args, "judge_model", "default")
    return await _rubric_judge_once(args, sample, rm_url=args.rm_url, judge_model=judge_model)


async def self_rubric_judge_reward(args, sample: Sample) -> float:
    """Local SGLang rubric judge (same contract as :func:`llm_rubric_judge_reward`)."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
    return await _rubric_judge_once(args, sample, rm_url=url, judge_model="default")
