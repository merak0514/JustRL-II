import logging
import os
import subprocess
import uuid
import re
import zmq
import json
import math
import random
import threading

logger = logging.getLogger(__name__)

_ZMQ_TIMEOUT_MS = int(os.environ.get('SANDBOX_TIMEOUT_MS', str(5 * 60 * 1000)))

_endpoint_counter = 0
_endpoint_lock = threading.Lock()

def _parse_sandbox_endpoints():
    """Parse multi-sandbox endpoints from SANDBOX_ENDPOINTS env var.
    Format: 'tcp://host1:port1,tcp://host2:port2'
    Falls back to SANDBOX_ADDR:SANDBOX_PORT if not set.
    """
    endpoints_str = os.environ.get('SANDBOX_ENDPOINTS', '')
    if endpoints_str:
        return [ep.strip() for ep in endpoints_str.split(',') if ep.strip()]
    return []

def _pick_endpoint(endpoints):
    """Round-robin pick from multiple endpoints."""
    global _endpoint_counter
    if not endpoints:
        return None
    with _endpoint_lock:
        idx = _endpoint_counter % len(endpoints)
        _endpoint_counter += 1
    return endpoints[idx]

# def check_torch_nn_usage(code_string: str) -> int:
#     # 查看是否有torch调用，如果有，返回0；如果没有，返回1。

#     whitelist = {
#         'nn.Parameter',
#         'torch.ones',
#         'torch.zeros',
#         'torch.empty',
#         'nn.Module',
#         'torch.Tensor',
#         'torch.tensor' 
#     }

#     all_found_usages = set()
#     is_nn_triggered = False

#     lines = code_string.splitlines()

#     for line in lines:
#         clean_line = line.strip()

#         if clean_line.startswith(('import ', 'from ')):
#             continue

#         current_line_matches = re.findall(r'(torch|nn|F)\.(\w+)', clean_line)

#         if not current_line_matches:
#             continue

#         for module, func in current_line_matches:
#             if module == 'nn':
#                 is_nn_triggered = True
            
#             all_found_usages.add(f"{module}.{func}")

#     if not is_nn_triggered:
#         return 1
    
#     if all_found_usages.issubset(whitelist):
#         return 1
#     else:
#         return 0



def check_torch_nn_usage(code_string: str) -> int:
    whitelist = {
        'nn.Parameter',
        'torch.ones',
        'torch.zeros', # 修正：应该是 torch.zeros 而不是 torch.zero
        'torch.empty',
        'nn.Module',
        'torch.Tensor',
        'torch.tensor',
        'nn.init',
    }

    all_found_usages = set()
    is_nn_triggered = False

    lines = code_string.splitlines()

    for line in lines:
        clean_line = line.strip()

        if clean_line.startswith(('import ', 'from ')):
            continue

        current_line_matches = re.findall(r'(torch|nn|F)\.(\w+)', clean_line)

        if not current_line_matches:
            continue

        for module, func in current_line_matches:
            if module == 'nn':
                is_nn_triggered = True
            
            all_found_usages.add(f"{module}.{func}")

    if not is_nn_triggered:
        return 1
    
    if all_found_usages.issubset(whitelist):
        return 1
    else:
        return 0

def _zmq_send_single(request: dict, endpoint: str, timeout_ms: int) -> tuple:
    """Send a single ZMQ request to one endpoint."""
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)

    try:
        socket.connect(endpoint)
        socket.send(json.dumps(request).encode())
        response_data = socket.recv()
        result = json.loads(response_data.decode())

        status = result.get('status')
        if status == 'success' and result.get('result', {}).get('success'):
            speedup = result['result'].get('speedup')
            message = f"Speedup: {speedup}x" if speedup else "Success with no speedup data"
            return (True, message, result)
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            return (False, error_msg, result)
    except zmq.Again:
        return (False, f"ZMQ timeout ({timeout_ms}ms) on {endpoint}", {})
    except zmq.ZMQError as e:
        return (False, f"ZMQ error on {endpoint}: {e}", {})
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return (False, f"Response decode error: {e}", {})
    finally:
        socket.close()
        context.term()


def zmq_send_request(request: dict, addr: str = "tcp://172.24.30.208", port: int = 5555, timeout_ms: int | None = None) -> tuple:
    """发送ZMQ请求并返回 (success, message, result_dict)。

    支持多沙箱实例：设置 SANDBOX_ENDPOINTS='tcp://h1:5555,tcp://h2:5556'
    会轮询分发请求，超时后尝试下一个实例。
    """
    if timeout_ms is None:
        timeout_ms = _ZMQ_TIMEOUT_MS

    endpoints = _parse_sandbox_endpoints()

    if not endpoints:
        endpoint = f"{addr}:{port}"
        return _zmq_send_single(request, endpoint, timeout_ms)

    picked = _pick_endpoint(endpoints)
    result = _zmq_send_single(request, picked, timeout_ms)

    if not result[0] and not result[2] and len(endpoints) > 1:
        for ep in endpoints:
            if ep == picked:
                continue
            logger.info(f"Failover: retrying on {ep}")
            result = _zmq_send_single(request, ep, timeout_ms)
            if result[0] or result[2]:
                break

    return result


def extract_python_code_1(text):
    pattern = r'```python.*?\n(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    return code_blocks

def compute_score(solution_str, ground_truth):
    TRITON_HOME=os.environ.get('TRITON_HOME', '/home/wangzefan/data/triton_home')
    os.makedirs(os.path.join(TRITON_HOME, 'torch_'), exist_ok=True)
    os.makedirs(os.path.join(TRITON_HOME, 'triton_'), exist_ok=True)
    os.makedirs(os.path.join(TRITON_HOME, 'content_'), exist_ok=True)
    uuid_str = 'a'+uuid.uuid4().hex

    content_path = os.path.join(TRITON_HOME, 'content_', uuid_str+'.txt')
    triton_path = os.path.join(TRITON_HOME, 'triton_', uuid_str+'.py')
    torch_path = os.path.join(TRITON_HOME, 'torch_', uuid_str+'.py')
    tmp_paths = [content_path, triton_path, torch_path]

    try:
        with open(content_path, 'w') as f:
            f.write(solution_str)

        if 'TEST_TRITON' in os.environ:
            triton_code_str = ground_truth['torch_function']
        else:
            triton_code_str = solution_str.split('</think>')[-1]
            triton_code_blocks = extract_python_code_1(triton_code_str)
            if len(triton_code_blocks) > 0:
                triton_code_str = triton_code_blocks[-1]
            else:
                triton_code_str = ""

        format_res = check_torch_nn_usage(triton_code_str)

        with open(triton_path, 'w') as f:
            f.write(triton_code_str)

        torch_code_str = ground_truth['torch_function']
        with open(torch_path, 'w') as f:
            f.write(torch_code_str)

        result = zmq_send_request(
            request={"torch_code": torch_code_str, "triton_code": triton_code_str,
                     "torch_class": "Model", "triton_class": "ModelNew"},
            addr=os.environ.get('SANDBOX_ADDR', "tcp://172.24.30.208"),
            port=os.environ.get('SANDBOX_PORT', 5555),
        )

        if not result[0] and not result[2]:
            logger.warning("Sandbox unreachable: %s", result[1])

        sandbox_status = result[2].get('status', '')
        sandbox_success = result[2].get('result', {}).get('success', False)
        speedup = result[2].get('result', {}).get('speedup')

        reward_mode = os.environ.get('TRITON_REWARD_MODE', 'binary')

        if reward_mode == 'continuous':
            # --- 备选方案：5 级连续奖励 (0.0 / 0.1 / 0.2 / 0.3 / 0.5-1.0) ---
            # 启用方式: export TRITON_REWARD_MODE=continuous
            if result[0] and sandbox_success:
                if speedup is None or speedup <= 0:
                    speedup = 1.0
                base_reward = float(os.environ.get('TRITON_REWARD_BASE', '0.5'))
                bonus_max = float(os.environ.get('TRITON_REWARD_BONUS_MAX', '0.5'))
                speedup_coef = float(os.environ.get('TRITON_REWARD_SPEEDUP_COEF', '0.25'))
                speedup_bonus = min(bonus_max, max(0.0, math.log2(max(speedup, 1.0)) * speedup_coef))
                return {
                    "score": round(base_reward + speedup_bonus, 4),
                    "speedup": speedup,
                    "pred": solution_str,
                }
            elif sandbox_status == 'success' and not sandbox_success:
                return {"score": 0.3, "speedup": 0, "pred": solution_str}
            elif triton_code_str.strip() and format_res and "@triton.jit" in triton_code_str:
                return {"score": 0.2, "speedup": 0, "pred": solution_str}
            elif triton_code_str.strip():
                return {"score": 0.1, "speedup": 0, "pred": solution_str}
            else:
                return {"score": 0.0, "speedup": 0, "pred": solution_str}
        else:
            # --- 默认方案：二值奖励 (0 / 1) ---
            if result[0] and sandbox_success:
                return {"score": 1.0, "speedup": speedup or 0, "pred": solution_str}
            else:
                return {"score": 0.0, "speedup": 0, "pred": solution_str}
    except OSError as e:
        # IO failure (e.g. errno=28 No space left, errno=13 Permission denied,
        # errno=122 Disk quota exceeded). Without this guard a single transient
        # write failure on the shared FS aborts the whole Ray actor and kills
        # training. Treat as a "failed sample" instead so training can keep going.
        logger.error(
            "compute_score IO error: errno=%s path=%s msg=%s",
            getattr(e, "errno", "?"),
            getattr(e, "filename", "?"),
            e,
        )
        return {
            "score": 0.0,
            "speedup": 0,
            "pred": solution_str,
            "error": f"io_errno_{getattr(e, 'errno', 'unknown')}",
        }
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    # # 运行
    # os.environ['PYTHONPATH']=TRITON_HOME
    # # os.environ['CUDA_VISIBLE_DEVICES']=str(random.randint(0,7))
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    # # conda_env='verl083'
    # proc = subprocess.run(
    #     [os.environ.get('PYTHON_HOME','/home/wangzefan/anaconda3/envs/verl083/bin/python'), os.path.join(TRITON_HOME, uuid_str+'.py')],
    #     capture_output=True,  # 捕获 stdout 和 stderr
    #     text=True             # 以字符串形式返回，而非 bytes
    # )
    # if proc.stdout.find('Test passed')>-1:
    #     return 1.
    # else:
    #     print(f'fail! {uuid_str}')
    #     return 0.

if __name__ == "__main__":
    # Test data - PyTorch and Triton code examples (from test_simple.py)
    # with open("/home/yinxinyu/data/T2C/dataset/L2&L3_train_dataset_1113.json", "r") as f:
    #     dataset = json.load(f)
    # print(dataset[0].keys())

    
    # TORCH_CODE = extract_python_code_1(dataset[0]["instruction"])[-1]
    # CUDA_CODE = extract_python_code_1(dataset[0]["output"])[0]

    # with open("/home/yinxinyu/data/T2C/dataset/cuda_cor/39_Conv2d_AdaptiveAvgPool2d.py", "r") as f:
    #     CUDA_CODE = f.read()

    # with open("/home/yinxinyu/data/T2C/dataset/torch_cor/39_Conv2d_AdaptiveAvgPool2d.py", "r") as f:
    #     TORCH_CODE = f.read()

    # print(TORCH_CODE)
    # print(CUDA_CODE)
#     TORCH_CODE = """
# import torch
# import torch.nn as nn
# from torch.nn import functional as F

# class VectorAdd(torch.nn.Module):
#     def forward(self, x, y):
#         return x + y

# def get_init_inputs():
#     return [[], {}]
    
# def get_inputs():
#     return [torch.randn(1024), torch.randn(1024)]
# """

#     TRITON_CODE = """
# import triton
# import triton.language as tl
# import torch
# import torch.nn as nn

# @triton.jit
# def vector_add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < n_elements
#     x = tl.load(x_ptr + offsets, mask=mask)
#     y = tl.load(y_ptr + offsets, mask=mask)
#     z = x + y
#     tl.store(z_ptr + offsets, z, mask=mask)

# class VectorAddTriton(torch.nn.Module):
#     def forward(self, x, y):
#         output = torch.empty_like(x)
#         n_elements = output.numel()
#         grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
#         vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
#         return output
# """

    info = {
        "torch_code": "",
        "triton_code": ""
    }

    result = zmq_send_request(request=info, port=5555)
    print(result)

    print(result[2]['status'])
    print(result[0])