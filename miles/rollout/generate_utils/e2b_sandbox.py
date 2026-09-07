"""
Async E2B sandbox management for tool-integrated reasoning.

Provides per-sample stateful sandbox lifecycle: create a sandbox at the start of
a sample's generation, execute code across multiple turns (state persists), and
kill the sandbox when the sample finishes.

Configuration is read from environment variables:
    E2B_API_KEY, E2B_DOMAIN, E2B_TEMPLATE, http_proxy

Concurrency is controlled via an asyncio.Semaphore whose size is set by the
``--generate-e2b-max-concurrency`` CLI argument.
"""

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore(max_concurrency: int) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max_concurrency)
    return _semaphore


async def create_sandbox(
    sandbox_timeout: int = 3600,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Create an E2B async sandbox using env-var configuration.

    Returns an ``AsyncSandbox`` instance.  The caller is responsible for
    calling :func:`kill_sandbox` when done.

    Retries up to *max_retries* times with exponential backoff to handle
    transient failures (proxy limits, rate limits, connection pool exhaustion).
    """
    from e2b_code_interpreter import AsyncSandbox

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            sbx = await AsyncSandbox.create(
                template=os.getenv("E2B_TEMPLATE", "tir-code-interpreter"),
                api_key=os.getenv("E2B_API_KEY"),
                domain=os.getenv("E2B_DOMAIN"),
                proxy=os.getenv("http_proxy"),
                timeout=sandbox_timeout,
            )
            logger.info("E2B sandbox created (id=%s) in %.2fs", sbx.sandbox_id, time.time() - t0)
            return sbx
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "E2B sandbox creation failed (attempt %d/%d, %s: %s), retrying in %.1fs",
                    attempt, max_retries, type(exc).__name__, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "E2B sandbox creation failed after %d attempts. Last error: %s: %s",
                    max_retries, type(last_exc).__name__, last_exc,
                )
    raise last_exc  # type: ignore[misc]


async def execute_code(sandbox: Any, code: str, timeout: int = 600) -> str:
    """Run *code* inside *sandbox* and return a human-readable result string.

    Captures stdout, stderr, error, and results from the execution.  On any
    exception the error message is returned as a string (the sandbox stays
    alive for subsequent calls).
    """
    try:
        result = await sandbox.run_code(code, timeout=timeout)
    except Exception as exc:
        logger.warning("E2B code execution failed: %s", exc)
        return f"Error: code execution failed: {exc}"

    parts: list[str] = []

    if result.logs.stdout:
        stdout_text = "".join(line.strip("\n") + "\n" for line in result.logs.stdout)
        parts.append(stdout_text.rstrip("\n"))

    if result.logs.stderr:
        stderr_text = "".join(line.strip("\n") + "\n" for line in result.logs.stderr)
        parts.append(f"[stderr]\n{stderr_text.rstrip(chr(10))}")

    if result.error:
        parts.append(f"Error: {result.error.name}: {result.error.value}")
        if result.error.traceback:
            parts.append(result.error.traceback)

    if result.results:
        for r in result.results:
            if r.text is not None:
                parts.append(r.text)

    return "\n".join(parts) if parts else "(no output)"


async def kill_sandbox(sandbox: Any) -> None:
    """Kill *sandbox*, swallowing errors so callers can use this in ``finally``."""
    try:
        await sandbox.kill()
        logger.info("E2B sandbox killed (id=%s)", sandbox.sandbox_id)
    except Exception as exc:
        logger.warning("Failed to kill E2B sandbox: %s", exc)


async def execute_tool_in_sandbox(
    sandbox: Any,
    name: str,
    params: dict,
    code_timeout: int = 600,
) -> str:
    """Execute a single tool call inside an E2B sandbox.

    Currently only ``code_interpreter`` is supported.  Returns the result as a
    plain string suitable for inclusion in a tool-response message.
    """
    if name != "code_interpreter":
        return f"Error: unsupported tool '{name}'"

    code = params.get("code", "")
    if not code.strip():
        return "Error: no code provided"

    return await execute_code(sandbox, code, timeout=code_timeout)
