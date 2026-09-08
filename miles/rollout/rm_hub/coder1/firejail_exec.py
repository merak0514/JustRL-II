import os
import signal
import subprocess
import shlex
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _DEFAULT_TIMEOUT_SECONDS, _ERROR_MSG_PREFIX, coder1_debug_log, wrap_code_with_imports
from .executor import get_current_batch_ctx

CLI_ARG_SIZE_LIMIT = 1024 * 3

_MAX_STDIN_LOG_CHARS = 4096
_MAX_CMD_LOG_CHARS = 2048
_MAX_OUTPUT_LOG_CHARS = 8192

_RLIMIT_AS_MB = int(os.environ.get("CODER1_RLIMIT_AS_MB", "512"))
_RLIMIT_AS_BYTES = _RLIMIT_AS_MB * 1024 * 1024


def _shell_join(argv) -> str:
    try:
        return shlex.join(argv)
    except AttributeError:
        return " ".join(shlex.quote(str(x)) for x in argv)


def _log_full_cmd(command, *, cwd: str = None) -> None:
    cmd_str = _shell_join(command)
    if len(cmd_str) > _MAX_CMD_LOG_CHARS:
        cmd_str_truncated = cmd_str[:_MAX_CMD_LOG_CHARS] + f"... (truncated, total={len(cmd_str)})"
    else:
        cmd_str_truncated = cmd_str
    msg = f"[firejail.cmd_full] {('cwd=' + cwd + ' ') if cwd else ''}{cmd_str_truncated}"
    coder1_debug_log(msg)
    try:
        coder1_debug_log(msg, path=os.path.join(os.getcwd(), "coder1.log"))
    except Exception:
        pass


def _log_stdin_and_repro(command, *, stdin: str = None, cwd: str = None) -> None:
    _log_full_cmd(command, cwd=cwd)

    if stdin is None:
        msg = "[firejail.stdin] <none>"
        coder1_debug_log(msg)
        try:
            coder1_debug_log(msg, path=os.path.join(os.getcwd(), "coder1.log"))
        except Exception:
            pass
        return

    s = str(stdin)
    head = s[:_MAX_STDIN_LOG_CHARS]
    truncated = len(s) > _MAX_STDIN_LOG_CHARS

    msg = f"[firejail.stdin] stdin_len={len(s)} truncated={truncated} stdin_head={repr(head)}"
    coder1_debug_log(msg)
    try:
        coder1_debug_log(msg, path=os.path.join(os.getcwd(), "coder1.log"))
    except Exception:
        pass

    cmd_str = _shell_join(command)
    if len(cmd_str) > _MAX_CMD_LOG_CHARS:
        cmd_str = cmd_str[:_MAX_CMD_LOG_CHARS] + "..."
    stdin_q = shlex.quote(head if truncated else s)
    repro = f"[firejail.repro] printf %s {stdin_q} | {cmd_str}"
    if truncated:
        repro += "  # NOTE: stdin was truncated in this log"
    coder1_debug_log(repro)
    try:
        coder1_debug_log(repro, path=os.path.join(os.getcwd(), "coder1.log"))
    except Exception:
        pass


def _killpg_safe(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass


def _run_subprocess(command, *, input_data=None, env=None, cwd=None, timeout=None):
    """
    Popen + communicate wrapper with BatchContext tracking.
    start_new_session=True puts the child in its own process group so
    cancel_all() can kill the whole group with killpg.
    """
    batch_ctx = get_current_batch_ctx()
    if batch_ctx and batch_ctx.cancelled:
        return subprocess.CompletedProcess(
            command, returncode=-1, stdout=b"", stderr=b"Cancelled",
        )

    if _RLIMIT_AS_BYTES > 0:
        command = ["prlimit", f"--as={_RLIMIT_AS_BYTES}:{_RLIMIT_AS_BYTES}", "--"] + command

    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_data is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )

    if batch_ctx:
        batch_ctx.register_proc(proc)

    try:
        stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _killpg_safe(proc)
        proc.wait()
        raise
    finally:
        if batch_ctx:
            batch_ctx.unregister_proc(proc)


def code_exec_firejail(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    code = wrap_code_with_imports(code)

    _local_log = os.path.join(os.getcwd(), "coder1.log")
    enter_msg = f"[firejail.enter] code_len={len(code)} stdin_len={0 if stdin is None else len(str(stdin))} timeout={timeout} pytest={pytest is not None}"
    coder1_debug_log(enter_msg)
    try:
        coder1_debug_log(enter_msg, path=_local_log)
    except Exception:
        pass
    code_head = code[:100] if len(code) > 100 else code
    code_tail = code[-100:] if len(code) > 100 else ""
    if code_tail:
        code_msg = f"[firejail.code] ===== BEGIN CODE (truncated, total={len(code)}) =====\n{code_head}\n... (truncated) ...\n{code_tail}\n===== END CODE ====="
    else:
        code_msg = f"[firejail.code] ===== BEGIN CODE =====\n{code_head}\n===== END CODE ====="
    coder1_debug_log(code_msg)
    try:
        coder1_debug_log(code_msg, path=_local_log)
    except Exception:
        pass

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=2",
        "--rlimit-nofile=2",
        "--rlimit-fsize=2m",
        "--rlimit-as=8m",
        f"--timeout=00:00:{timeout}",
    ]

    if pytest:
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["python3", "-m", "pytest", tmpdir])
            coder1_debug_log(f"[firejail.cmd] {' '.join(command[:8])} ...")
            _log_full_cmd(command, cwd=tmpdir)
            try:
                result = _run_subprocess(
                    command, env=env, cwd=tmpdir, timeout=timeout + 2,
                )
            except subprocess.TimeoutExpired:
                coder1_debug_log(f"[firejail.timeout] subprocess timeout after {timeout + 2}s")
                return False, _ERROR_MSG_PREFIX + f"Execution timeout after {timeout + 2}s"
    else:
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command.extend(["python3", "-c", code])
            coder1_debug_log("[firejail.cmd] firejail ... python3 -c <code>")
            _log_stdin_and_repro(command, stdin=stdin)
            try:
                result = _run_subprocess(
                    command,
                    input_data=stdin.encode() if stdin else None,
                    env=env,
                    timeout=timeout + 2,
                )
            except subprocess.TimeoutExpired:
                coder1_debug_log(f"[firejail.timeout] subprocess timeout after {timeout + 2}s")
                return False, _ERROR_MSG_PREFIX + f"Execution timeout after {timeout + 2}s"
        else:
            with NamedTemporaryFile() as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command.insert(4, f"--whitelist={tmp.name}")
                command.extend(["python3", tmp.name])
                coder1_debug_log(f"[firejail.cmd] firejail ... python3 {tmp.name}")
                _log_stdin_and_repro(command, stdin=stdin)
                try:
                    result = _run_subprocess(
                        command,
                        input_data=stdin.encode() if stdin else None,
                        env=env,
                        timeout=timeout + 2,
                    )
                except subprocess.TimeoutExpired:
                    coder1_debug_log(f"[firejail.timeout] subprocess timeout after {timeout + 2}s")
                    return False, _ERROR_MSG_PREFIX + f"Execution timeout after {timeout + 2}s"

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()
    coder1_debug_log(
        f"[firejail.ret] returncode={result.returncode} stdout_len={len(stdout)} stderr_len={len(stderr)} stderr_head={repr(stderr[:200])}"
    )

    if result.returncode == 0:
        return True, stdout

    stdout_truncated = stdout if len(stdout) <= _MAX_OUTPUT_LOG_CHARS else stdout[:_MAX_OUTPUT_LOG_CHARS] + f"\n... (truncated, total={len(stdout)} chars)"
    stderr_truncated = stderr if len(stderr) <= _MAX_OUTPUT_LOG_CHARS else stderr[:_MAX_OUTPUT_LOG_CHARS] + f"\n... (truncated, total={len(stderr)} chars)"
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout_truncated}\n\nSTDERR:\n{stderr_truncated}"
