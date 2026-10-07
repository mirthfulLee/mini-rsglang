"""Identical raw-text HTTP workloads on the same A100 device group, running servers sequentially."""

import argparse
import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import threading
import time

import numpy as np
import requests
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model", default="/models/store/Qwen/Qwen3-0.6B")
p.add_argument("--requests", type=int, default=16)
p.add_argument("--concurrency", type=int, default=8)
p.add_argument("--input-tokens", type=int, default=128)
p.add_argument("--output-tokens", type=int, default=32)
p.add_argument("--mini-disable-pynccl", action="store_true")
p.add_argument("--tp-size", type=int, default=1)
p.add_argument("--device", type=int, default=0)
p.add_argument("--port", type=int, default=19673)
p.add_argument("--output", default="results/comparison.json")
a = p.parse_args()
assert a.requests > 0 and 0 < a.concurrency <= 32
base = f"http://127.0.0.1:{a.port}"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
env["CUDA_HOME"] = "/usr/local/cuda"
env["PATH"] = str(ROOT / ".venv-verify/bin") + ":/usr/local/cuda/bin:" + env["PATH"]
env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
# PyNCCL JIT links -lnccl; pip runtimes commonly expose only the versioned SONAME.
# A workspace-local linker symlink avoids changing any system CUDA installation.
nccl = env.get("RSGLANG_NCCL_LIBRARY")
if nccl and pathlib.Path(nccl).is_file():
    link_dir = ROOT / "target/verify-native"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / "libnccl.so"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(pathlib.Path(nccl).resolve())
    env["LIBRARY_PATH"] = str(link_dir) + ":" + env.get("LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(pathlib.Path(nccl).resolve().parent) + ":" + env["LD_LIBRARY_PATH"]
    )
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
ids = tok.encode(
    "The quick brown fox jumps over the lazy dog. " * (a.input_tokens // 8 + 10)
)[: a.input_tokens]
prompt = tok.decode(ids)
assert len(tok.encode(prompt)) == a.input_tokens
devices = list(range(a.device, a.device + a.tp_size))
assert a.tp_size > 0 and a.device >= 0
visible = env.get("CUDA_VISIBLE_DEVICES")
if visible is not None:
    visible = [v.strip() for v in visible.split(",") if v.strip()]
    assert a.device + a.tp_size <= len(visible), "TP size exceeds CUDA_VISIBLE_DEVICES"
    physical_devices = [visible[d] for d in devices]
else:
    physical_devices = list(map(str, devices))
config = json.loads((pathlib.Path(a.model) / "config.json").read_text())
local_heads = max(1, config["num_key_value_heads"] // a.tp_size)
num_pages = (
    2048
    * 1024
    * 1024
    // (config["num_hidden_layers"] * 2 * 16 * local_heads * config["head_dim"] * 2)
)
output_path = ROOT / a.output
output_path.parent.mkdir(parents=True, exist_ok=True)
results = []
process = None


def client():
    s = requests.Session()
    s.trust_env = False
    return s


def memory():
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + ",".join(physical_devices),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [int(v) * 1024 * 1024 for v in value.splitlines()]


def stop():
    global process
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    # Kill leftover children only in the process group created by this script.
    if process:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    process = None


def generate(endpoint, model, max_tokens):
    s = client()
    start = time.perf_counter()
    times = []
    done = False
    with s.post(
        base + endpoint,
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        },
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        pending = b""
        for packet in r.raw.stream(amt=None, decode_content=True):
            pending += packet
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                assert "error" not in event, event
                for c in event.get("choices", []):
                    if c.get("finish_reason") is None:
                        times.append(time.perf_counter() - start)
            if done:
                break
    assert done and len(times) == max_tokens, (done, len(times), max_tokens)
    return {
        "ttft_s": times[0],
        "itl_s": np.diff(times).tolist(),
        "latency_s": time.perf_counter() - start,
        "tokens": len(times),
    }


try:
    for framework in ["mini-rsglang", "mini-sglang"]:
        for warm in [False, True]:
            cache = "radix" if warm else "naive"
            log_path = (
                output_path.parent / f"compare-tp{a.tp_size}-{framework}-{cache}.log"
            )
            log = open(log_path, "w")
            if framework == "mini-rsglang":
                cmd = [
                    str(ROOT / "target/release/mini-rsglang"),
                    "--model",
                    a.model,
                    "--device",
                    str(a.device),
                    "--tensor-parallel-size",
                    str(a.tp_size),
                    "--prefill-budget",
                    "512",
                    "serve",
                    "--port",
                    str(a.port),
                ]
                if not warm:
                    cmd.insert(1, "--no-prefix-cache")
                proc_env = env.copy()
                endpoint = "/v1/completions"
            else:
                cmd = [
                    str(ROOT / ".venv-verify/bin/python"),
                    "-m",
                    "minisgl",
                    "--model",
                    a.model,
                    "--tensor-parallel-size",
                    str(a.tp_size),
                    "--dtype",
                    "bfloat16",
                    "--max-running-requests",
                    "32",
                    "--max-seq-len-override",
                    "4096",
                    "--num-pages",
                    str(num_pages),
                    "--page-size",
                    "16",
                    "--max-prefill-length",
                    "512",
                    "--attn",
                    "fi",
                    "--graph",
                    "0",
                    "--cache",
                    cache,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(a.port),
                ]
                proc_env = env.copy()
                if a.mini_disable_pynccl:
                    cmd.append("--disable-pynccl")
                proc_env["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices)
                endpoint = "/v1/chat/completions"
            baseline = memory()
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=proc_env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            deadline = time.monotonic() + 120
            s = client()
            model_id = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{framework} failed; see {log_path}")
                try:
                    r = s.get(base + "/v1/models", timeout=1)
                    if r.status_code == 200:
                        model_id = r.json()["data"][0]["id"]
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            assert model_id, "startup timeout"
            # Warm libraries for both runs; with naive cache this cannot reuse prefixes.
            generate(endpoint, model_id, 1)
            # Warm the actual prefill/decode batch shapes, excluding JIT from timing.
            for _ in range(2):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=a.concurrency
                ) as pool:
                    list(
                        pool.map(
                            lambda _: generate(endpoint, model_id, a.output_tokens),
                            range(a.concurrency),
                        )
                    )
            metrics_before = None
            if framework == "mini-rsglang":
                metrics_before = s.get(base + "/metrics", timeout=2).text
            peak = [memory()]
            finished = threading.Event()

            def monitor():
                while not finished.wait(0.1):
                    peak.append(memory())

            thread = threading.Thread(target=monitor)
            thread.start()
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=a.concurrency
            ) as pool:
                raw = list(
                    pool.map(
                        lambda _: generate(endpoint, model_id, a.output_tokens),
                        range(a.requests),
                    )
                )
            elapsed = time.perf_counter() - start
            finished.set()
            thread.join()
            peak.append(memory())
            metrics = None
            if framework == "mini-rsglang":
                metrics = s.get(base + "/metrics", timeout=2).text
            cached_tokens = None
            if metrics_before is not None:

                def cache_count(value):
                    return next(
                        int(line.split()[1])
                        for line in value.splitlines()
                        if line.startswith("rsglang_cached_tokens ")
                    )

                cached_tokens = cache_count(metrics) - cache_count(metrics_before)
            ttft = [r["ttft_s"] for r in raw]
            itl = [v for r in raw for v in r["itl_s"]]
            result = {
                "framework": framework,
                "cache": cache,
                "model": a.model,
                "dtype": "bfloat16",
                "devices": devices,
                "physical_devices": physical_devices,
                "tp_size": a.tp_size,
                "mini_disable_pynccl": a.mini_disable_pynccl,
                "logical_kv_pages": num_pages,
                "nccl_p2p_level": env.get("NCCL_P2P_LEVEL"),
                "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens,
                "requests": a.requests,
                "concurrency": a.concurrency,
                "elapsed_s": elapsed,
                "output_tokens_per_s": a.requests * a.output_tokens / elapsed,
                "ttft_ms": {
                    "p50": float(np.percentile(ttft, 50)) * 1000,
                    "p95": float(np.percentile(ttft, 95)) * 1000,
                },
                "itl_ms": {
                    "p50": float(np.percentile(itl, 50)) * 1000,
                    "p95": float(np.percentile(itl, 95)) * 1000,
                },
                "peak_memory_bytes": max(sum(v) - sum(baseline) for v in peak),
                "peak_memory_per_gpu_bytes": [
                    max(v[i] - baseline[i] for v in peak) for i in range(len(devices))
                ],
                "memory_samples_bytes": peak,
                "baseline_memory_bytes": baseline,
                "cached_tokens": cached_tokens,
                "cuda_graph": False,
                "overlap": False,
                "warmup_waves": 2,
                "command": cmd,
                "raw": raw,
                "metrics": metrics,
                "metrics_before": metrics_before,
            }
            results.append(result)
            output_path.write_text(json.dumps(results, indent=2))
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("raw", "metrics", "command")
                    },
                    indent=2,
                ),
                flush=True,
            )
            stop()
            log.close()
finally:
    stop()
