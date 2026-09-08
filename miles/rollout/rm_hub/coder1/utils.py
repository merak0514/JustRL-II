import os
import time
from typing import Optional

import requests

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 6

BASE_IMPORTS = """\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""


def wrap_code_with_imports(code: str) -> str:
    return BASE_IMPORTS + "\n" + code


def coder1_debug_enabled() -> bool:
    v = os.environ.get("CODER1_DEBUG", "")
    return v.lower() in {"1", "true", "yes", "y", "on"}


def coder1_debug_log(msg: str, *, path: Optional[str] = None) -> None:
    if not coder1_debug_enabled():
        return
    log_path = path or os.environ.get("CODER1_DEBUG_LOG", "/tmp/coder1_exec_debug.log")
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"{ts} pid={os.getpid()} {msg}\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except Exception:
        pass


def check_executor_alive(executor):
    try:
        return requests.get(executor + "/").status_code in [200, 404]
    except Exception:
        return False


def _normalize_line_whitespace(s: str) -> str:
    return '\n'.join(line.rstrip() for line in s.split('\n'))


def semantic_compare_output(output_str: str, expected_str: str) -> bool:
    """
    Compare two output strings semantically (follows the opencompass implementation).

    Idea:
        A program's stdout is just a string. Parse expected first to learn the
        expected type:
        - expected is a str → compare output as a plain string (no eval)
        - expected is an int/float/list/... → eval output before comparing
    """
    import json
    import numpy as np

    output_str_clean = _normalize_line_whitespace(output_str.strip())
    expected_str_clean = _normalize_line_whitespace(str(expected_str).strip())

    expected_parsed = False
    try:
        expected_obj = json.loads(expected_str_clean)
        expected_parsed = True
    except Exception:
        try:
            expected_obj = eval(expected_str_clean)
            expected_parsed = True
        except Exception:
            expected_obj = expected_str_clean

    if expected_parsed and isinstance(expected_obj, str):
        return output_str_clean == expected_obj

    output_parsed = False
    try:
        output_obj = eval(output_str_clean)
        output_parsed = True
    except Exception:
        try:
            output_obj = json.loads(output_str_clean)
            output_parsed = True
        except Exception:
            output_obj = output_str_clean

    if output_parsed and expected_parsed:
        if output_obj == expected_obj:
            return True
        return _float_compare(output_obj, expected_obj)
    else:
        if _multiline_compare(output_str_clean, expected_str_clean):
            return True
        return output_str_clean == expected_str_clean


def _is_int_like(val) -> bool:
    if isinstance(val, int):
        return True
    if isinstance(val, str) and val.isdigit():
        return True
    return False


def _multiline_compare(output_str: str, expected_str: str) -> bool:
    import numpy as np

    output_lines = output_str.split('\n')
    expected_lines = expected_str.split('\n')

    if len(output_lines) != len(expected_lines):
        return False

    for out_line, exp_line in zip(output_lines, expected_lines):
        out_line = out_line.strip()
        exp_line = exp_line.strip()

        if out_line == exp_line:
            continue

        try:
            out_val = float(out_line)
            exp_val = float(exp_line)
            if not np.allclose(out_val, exp_val):
                return False
        except (ValueError, TypeError):
            return False

    return True


def _float_compare(output_obj, expected_obj) -> bool:
    import numpy as np

    if isinstance(output_obj, list) and isinstance(expected_obj, list):
        if len(output_obj) != len(expected_obj):
            return False

        try:
            all_ints = all(
                _is_int_like(e1) and _is_int_like(e2)
                for e1, e2 in zip(output_obj, expected_obj)
            )

            if not all_ints:
                output_float = [float(e) for e in output_obj]
                expected_float = [float(e) for e in expected_obj]
                return np.allclose(output_float, expected_float)
        except (ValueError, TypeError):
            pass

        try:
            if isinstance(output_obj[0], list) and isinstance(expected_obj[0], list):
                if len(output_obj[0]) != len(expected_obj[0]):
                    return False

                all_ints = all(
                    _is_int_like(e1) and _is_int_like(e2)
                    for e1, e2 in zip(output_obj[0], expected_obj[0])
                )

                if not all_ints:
                    output_float = [float(e) for e in output_obj[0]]
                    expected_float = [float(e) for e in expected_obj[0]]
                    return np.allclose(output_float, expected_float)
        except (ValueError, TypeError, IndexError):
            pass

    elif isinstance(output_obj, (int, float)) and isinstance(expected_obj, (int, float)):
        if isinstance(output_obj, float) or isinstance(expected_obj, float):
            return np.allclose(float(output_obj), float(expected_obj))

    elif isinstance(output_obj, str) and isinstance(expected_obj, str):
        try:
            output_float = float(output_obj)
            expected_float = float(expected_obj)
            return np.allclose(output_float, expected_float)
        except (ValueError, TypeError):
            pass

    return False
