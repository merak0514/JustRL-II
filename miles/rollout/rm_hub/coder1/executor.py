"""
线程池 + 批次上下文模块

全局 ThreadPoolExecutor 单例提供自然并发限流（固定线程数 = 固定最大 firejail 并发数）。
BatchContext 跟踪每批评测的活跃子进程，fail-fast 时主动 kill 残留 firejail 进程。

环境变量：
  CODER1_MAX_WORKERS  线程池大小（默认 8）
"""
import os
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

from .utils import coder1_debug_log

EXECUTOR_MAX_WORKERS = int(os.environ.get("CODER1_MAX_WORKERS", "8"))

_executor_lock = threading.Lock()
_global_executor: Optional[ThreadPoolExecutor] = None
_executor_pid: Optional[int] = None

_tls = threading.local()


def get_global_executor() -> ThreadPoolExecutor:
    global _global_executor, _executor_pid
    with _executor_lock:
        current_pid = os.getpid()
        if (
            _global_executor is None
            or _global_executor._shutdown
            or _executor_pid != current_pid
        ):
            coder1_debug_log(
                f"[executor] Creating global executor with {EXECUTOR_MAX_WORKERS} workers (pid={current_pid})"
            )
            _global_executor = ThreadPoolExecutor(
                max_workers=EXECUTOR_MAX_WORKERS,
                thread_name_prefix="coder1_exec",
            )
            _executor_pid = current_pid
    return _global_executor


class BatchContext:
    """
    每次 _compute_score 调用的上下文，跟踪该批次活跃的子进程。
    fail-fast 时调用 cancel_all()：
      1. 设置 cancel_event → 阻止尚未启动的任务调用 code_exec
      2. kill 所有已注册的活跃子进程 → 被 communicate() 阻塞的线程立即释放
    """

    def __init__(self):
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._active_procs: Dict[int, subprocess.Popen] = {}

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @staticmethod
    def _killpg_safe(proc: subprocess.Popen) -> bool:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
                return True
            except Exception:
                return False

    def register_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            if self.cancel_event.is_set():
                self._killpg_safe(proc)
                return
            self._active_procs[id(proc)] = proc

    def unregister_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._active_procs.pop(id(proc), None)

    def cancel_all(self) -> None:
        self.cancel_event.set()
        with self._lock:
            killed = 0
            for proc in self._active_procs.values():
                if self._killpg_safe(proc):
                    killed += 1
            if killed:
                coder1_debug_log(
                    f"[batch] cancel_all: killed {killed} process groups"
                )
            self._active_procs.clear()


def set_current_batch_ctx(ctx: Optional[BatchContext]) -> None:
    _tls.batch_ctx = ctx


def get_current_batch_ctx() -> Optional[BatchContext]:
    return getattr(_tls, "batch_ctx", None)
