"""
Tests for the E2B sandbox utilities.

These tests require live E2B access (env vars E2B_API_KEY, E2B_DOMAIN, etc.).
They are skipped automatically when the environment is not configured.
"""

import os

import pytest

from miles.utils.async_utils import run

_E2B_CONFIGURED = bool(os.getenv("E2B_API_KEY"))

pytestmark = pytest.mark.skipif(not _E2B_CONFIGURED, reason="E2B env vars not set")


@pytest.fixture
def sandbox():
    from miles.rollout.generate_utils.e2b_sandbox import create_sandbox, kill_sandbox

    sbx = run(create_sandbox(sandbox_timeout=300))
    yield sbx
    run(kill_sandbox(sbx))


class TestE2BSandbox:
    def test_create_and_kill(self):
        from miles.rollout.generate_utils.e2b_sandbox import create_sandbox, kill_sandbox

        sbx = run(create_sandbox(sandbox_timeout=300))
        assert sbx is not None
        assert hasattr(sbx, "sandbox_id")
        run(kill_sandbox(sbx))

    def test_execute_simple_code(self, sandbox):
        from miles.rollout.generate_utils.e2b_sandbox import execute_code

        result = run(execute_code(sandbox, "print(2 + 2)", timeout=30))
        assert "4" in result

    def test_execute_stateful_across_calls(self, sandbox):
        """Variables persist across execute_code calls within the same sandbox."""
        from miles.rollout.generate_utils.e2b_sandbox import execute_code

        run(execute_code(sandbox, "x = 42", timeout=30))
        result = run(execute_code(sandbox, "print(x * 2)", timeout=30))
        assert "84" in result

    def test_execute_with_import(self, sandbox):
        from miles.rollout.generate_utils.e2b_sandbox import execute_code

        result = run(execute_code(sandbox, "import math; print(math.factorial(10))", timeout=30))
        assert "3628800" in result

    def test_execute_error_returns_string(self, sandbox):
        from miles.rollout.generate_utils.e2b_sandbox import execute_code

        result = run(execute_code(sandbox, "1/0", timeout=30))
        assert "Error" in result or "ZeroDivisionError" in result

    def test_execute_empty_code(self, sandbox):
        from miles.rollout.generate_utils.e2b_sandbox import execute_tool_in_sandbox

        result = run(execute_tool_in_sandbox(sandbox, "code_interpreter", {"code": ""}))
        assert "Error" in result or "no code" in result.lower()

    def test_unsupported_tool(self, sandbox):
        from miles.rollout.generate_utils.e2b_sandbox import execute_tool_in_sandbox

        result = run(execute_tool_in_sandbox(sandbox, "unknown_tool", {}))
        assert "unsupported" in result.lower()

    def test_execute_tool_in_sandbox_happy_path(self, sandbox):
        from miles.rollout.generate_utils.e2b_sandbox import execute_tool_in_sandbox

        result = run(execute_tool_in_sandbox(
            sandbox, "code_interpreter", {"code": "print('hello from e2b')"},
        ))
        assert "hello from e2b" in result

    def test_multiline_output(self, sandbox):
        from miles.rollout.generate_utils.e2b_sandbox import execute_code

        code = "for i in range(3): print(f'line {i}')"
        result = run(execute_code(sandbox, code, timeout=30))
        assert "line 0" in result
        assert "line 1" in result
        assert "line 2" in result
