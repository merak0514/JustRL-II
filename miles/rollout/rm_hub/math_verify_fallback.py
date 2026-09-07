"""math-verify fallback for the math-line graders (hard-timeout, fail-closed).

Why
---
The string graders on the math line (mathd normalization + the sympy string
path used by ``grade_answer_verl`` / deepscaler / dapo-minerva) return false
negatives on many mathematically equivalent answer forms: reordered terms
inside a ``\\frac`` numerator (``\\frac{-\\sqrt{3}+3\\sqrt{7}}{2}`` vs
``\\frac{3\\sqrt{7}-\\sqrt{3}}{2}``), split/merged fractions, ``24\\pi\\sqrt{2}``
vs ``24\\sqrt{2}\\pi`` factor order, unordered lists, etc.  math-verify grades
those correctly, and rejected 15/15 adversarial near-miss probes (rounded
decimals vs exact values, sign flips, open/closed intervals, ordered-pair
swaps) in our battery — so it is safe to use as a *fallback only*: it runs
after the original grader says 0 and can flip that to 1, never the reverse.

How to use
----------
Nothing to configure in code — the three math reward paths (``deepscaler``,
``math``, ``dapo``) call this automatically when the original grade is 0.

Dependency: ``pip install math-verify==0.9.0`` (pin the version).  If the
training container cannot reach an index, vendor the pure-python packages
(``math_verify``, ``latex2sympy2_extended``, ``antlr4``) to a shared
directory and set ``MILES_MATH_VERIFY_PYTHONPATH=<dir>`` — the worker reads
it directly, so launch scripts that overwrite ``PYTHONPATH`` don't matter.
If math_verify is missing entirely, grading silently degrades to the
original behavior (fail-closed), it never crashes training.

Env knobs::

    MILES_MATH_VERIFY_FALLBACK=0     kill switch (default: enabled)
    MILES_MATH_VERIFY_TIMEOUT=3.0    per-call wall-clock budget, seconds
    MILES_MATH_VERIFY_MAXLEN=2000    max chars of pred/label fed to sympy
    MILES_MATH_VERIFY_WORKERS=2     persistent worker subprocesses

Isolation design
----------------
sympy can occasionally hang, and math-verify's built-in SIGALRM timeout only
works in a process's main thread.  Each call is therefore served by a small
pool of persistent ``subprocess.Popen`` workers running a self-contained
script that imports ONLY math_verify (never the miles package — no
torch/CUDA in workers, no fork-after-CUDA hazards).  Communication is
JSON-lines over pipes with a select()-based wall-clock timeout; on timeout
the worker is killed and respawned and the grade is False.  Any parse error,
worker crash, or timeout grades False.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time

_ENABLED = os.environ.get("MILES_MATH_VERIFY_FALLBACK", "1").lower() not in ("0", "false")
_TIMEOUT = float(os.environ.get("MILES_MATH_VERIFY_TIMEOUT", "3"))
_MAXLEN = int(os.environ.get("MILES_MATH_VERIFY_MAXLEN", "2000"))
_WORKERS = int(os.environ.get("MILES_MATH_VERIFY_WORKERS", "2"))

# Self-contained worker: imports only math_verify. Protocol lines are JSON
# objects prefixed with MVGRADE so stray library prints cannot be mistaken
# for a response.
_WORKER_SRC = r"""
import json, sys
from math_verify import parse, verify
for line in sys.stdin:
    try:
        req = json.loads(line)
        gold = parse("\\boxed{" + req["gt"] + "}")
        ans = parse("\\boxed{" + req["pred"] + "}")
        out = bool(verify(gold, ans))
    except Exception:
        out = False
    sys.stdout.write("MVGRADE" + json.dumps({"r": out}) + "\n")
    sys.stdout.flush()
"""


class _Worker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None

    def _ensure(self) -> subprocess.Popen:
        if self.proc is None or self.proc.poll() is not None:
            # launch scripts overwrite PYTHONPATH wholesale, which would break
            # a vendored math_verify; give the worker its own knob that
            # survives that.
            env = os.environ.copy()
            extra = env.get("MILES_MATH_VERIFY_PYTHONPATH")
            if extra:
                env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "-c", _WORKER_SRC],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
            )
        return self.proc

    def _kill(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None

    def grade(self, payload: dict) -> bool:
        """One request/response exchange under the wall-clock budget."""
        with self.lock:
            try:
                proc = self._ensure()
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                deadline = time.monotonic() + _TIMEOUT
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._kill()
                        return False
                    ready, _, _ = select.select([proc.stdout], [], [], remaining)
                    if not ready:
                        self._kill()
                        return False
                    line = proc.stdout.readline()
                    if not line:  # worker died
                        self._kill()
                        return False
                    if line.startswith("MVGRADE"):
                        return bool(json.loads(line[len("MVGRADE"):])["r"])
                    # stray library output: ignore and keep reading
            except Exception:
                self._kill()
                return False


_workers = [_Worker() for _ in range(max(1, _WORKERS))]
_rr = [0]
_rr_lock = threading.Lock()


def math_verify_grade(pred, gt) -> bool:
    """True iff math-verify judges ``pred`` equivalent to ``gt`` in time.

    Both arguments are bare answer strings (no ``\\boxed``).  Fail-closed.
    """
    if not _ENABLED or pred is None or gt is None:
        return False
    pred, gt = str(pred).strip(), str(gt).strip()
    if not pred or not gt or len(pred) > _MAXLEN or len(gt) > _MAXLEN:
        return False
    # prefer an idle worker; otherwise round-robin (blocks on that worker)
    for w in _workers:
        if not w.lock.locked():
            return w.grade({"pred": pred, "gt": gt})
    with _rr_lock:
        _rr[0] = (_rr[0] + 1) % len(_workers)
        w = _workers[_rr[0]]
    return w.grade({"pred": pred, "gt": gt})
