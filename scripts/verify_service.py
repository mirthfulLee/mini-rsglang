"""Real CUDA service acceptance checks; starts/stops only its own child process."""

import argparse
import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import time

import requests
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model", default="/models/store/Qwen/Qwen3-0.6B")
p.add_argument("--device", type=int, default=1)
p.add_argument("--devices")
p.add_argument("--output", default="results/service-report.json")
p.add_argument("--request-timeout", type=float, default=180)
p.add_argument("--port", type=int, default=18789)
a = p.parse_args()
base = f"http://127.0.0.1:{a.port}"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
client = requests.Session()
client.trust_env = False
report = {}
behavior_failures = []
output = ROOT / a.output
output.parent.mkdir(parents=True, exist_ok=True)
log = open(output.with_suffix(".log"), "w")
devices = list(map(int, a.devices.split(","))) if a.devices else [a.device]
config = json.loads((pathlib.Path(a.model) / "config.json").read_text())
local_heads = max(1, config["num_key_value_heads"] // len(devices))
page_bytes = config["num_hidden_layers"] * 2 * 16 * local_heads * config["head_dim"] * 2
capacity = 64 * 1024 * 1024 // page_bytes
report["devices"] = devices
report["request_timeout_s"] = a.request_timeout
report["nccl_p2p_level"] = env.get("NCCL_P2P_LEVEL")
process = None


def start(kv=64):
    global process
    process = subprocess.Popen(
        [
            str(ROOT / "target/release/mini-rsglang"),
            "--model",
            a.model,
            "--devices",
            ",".join(map(str, devices)),
            "--kv-mib",
            str(kv),
            "--max-seq-len",
            "384",
            "--prefill-budget",
            "17",
            "--max-running",
            "4",
            "--max-waiting",
            "8",
            "serve",
            "--port",
            str(a.port),
        ],
        env=env,
        cwd=ROOT,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"service failed; see {output.with_suffix(chr(46) + 'log')}"
            )
        try:
            if client.get(base + "/health", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise RuntimeError("service startup timed out")


def stop():
    global process
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    process = None


def post(body, chat=False):
    return client.post(
        base + ("/v1/chat/completions" if chat else "/v1/completions"),
        json=body,
        timeout=a.request_timeout,
        stream=body.get("stream", False),
    )


def metrics():
    return {
        line.split()[0][8:]: int(line.split()[1])
        for line in client.get(base + "/metrics", timeout=2).text.splitlines()
        if line and not line.startswith("#")
    }


def idle():
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        m = metrics()
        if m["running"] == m["waiting"] == 0:
            return m
        time.sleep(0.02)
    raise AssertionError("scheduler did not become idle")


def stream(body):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    response = post(body, chat=True)
    response.raise_for_status()
    parts = []
    finish = None
    usage = None
    done = False
    for rawline in response.iter_lines(chunk_size=1, decode_unicode=False):
        line = rawline.decode("utf-8")
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            done = True
            break
        v = json.loads(value)
        assert "error" not in v, v
        if v.get("usage"):
            usage = v["usage"]
        for c in v.get("choices", []):
            parts.append(c.get("delta", {}).get("content", ""))
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    assert done and finish and usage
    return "".join(parts), usage, finish


try:
    start()
    model_id = client.get(base + "/v1/models", timeout=2).json()["data"][0]["id"]
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "请用一句话解释Rust的所有权。"}],
        "max_tokens": 24,
        "temperature": 0,
    }
    r = post(body, chat=True)
    r.raise_for_status()
    reference = r.json()
    text = reference["choices"][0]["message"]["content"]
    streamed, usage, reason = stream(body)
    assert streamed == text, (streamed, text)
    assert "�" not in streamed
    assert (
        usage["completion_tokens"] == 24
        and usage["prompt_tokens_details"]["cached_tokens"] > 0
    )
    report["chinese_stream_and_prefix_cache"] = {
        "text": text,
        "usage": usage,
        "finish_reason": reason,
    }
    eos_body = {
        **body,
        "messages": [{"role": "user", "content": "只回答数字1，不要解释。"}],
        "max_tokens": 128,
    }
    eos_response = post(eos_body, chat=True)
    eos_response.raise_for_status()
    eos_result = eos_response.json()
    assert eos_result["choices"][0]["finish_reason"] == "stop", eos_result
    assert eos_result["usage"]["completion_tokens"] < 128, eos_result
    report["eos_termination"] = eos_result
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    prompt = "Explain what a mutex does in one sentence."
    ids = tok.encode(prompt)
    token = post({"prompt": ids, "max_tokens": 12}).json()
    raw = post({"prompt": prompt, "max_tokens": 12}).json()
    assert token["choices"][0]["text"] == raw["choices"][0]["text"]
    report["text_and_token_ids_equal"] = True
    # Batch the same prompt alongside other lengths, retaining a single-request reference.
    reference = post({"prompt": prompt, "max_tokens": 12}).json()["choices"][0]["text"]
    prompts = [prompt, "A short story about a cat.", "Write about Rust. " * 20, prompt]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(
            pool.map(lambda x: post({"prompt": x, "max_tokens": 12}), prompts)
        )
    assert all(r.status_code == 200 for r in replies), [r.text for r in replies]
    outputs = [r.json()["choices"][0]["text"] for r in replies]
    matches = outputs[0] == outputs[-1] == reference
    report["mixed_length_batch_equal"] = matches
    report["mixed_length_batch_outputs"] = {
        "reference": reference,
        "prompts": prompts,
        "outputs": outputs,
    }
    if not matches:
        behavior_failures.append("single-request vs mixed-length batch text mismatch")
    sample = {
        "prompt": prompt,
        "max_tokens": 12,
        "temperature": 0.8,
        "top_k": 8,
        "top_p": 0.9,
        "seed": 9,
    }
    x = post(sample)
    y = post(sample)
    assert x.status_code == y.status_code == 200
    assert x.json()["choices"][0]["text"] == y.json()["choices"][0]["text"]
    report["seeded_sampling_equal"] = True
    for name, bad in {
        "empty": {"prompt": []},
        "token": {"prompt": [999999]},
        "context": {"prompt": [1] * 380, "max_tokens": 20},
        "top_p": {"prompt": "hi", "top_p": 0},
        "n": {"prompt": "hi", "n": 2},
        "tools": {"prompt": "hi", "tools": []},
        "model": {"prompt": "hi", "model": "wrong"},
    }.items():
        r = post(bad)
        assert r.status_code == 400, (name, r.status_code, r.text)
        assert "error" in r.json()
    report["invalid_requests_rejected"] = True
    # A dropped HTTP response cancels through the owned GenerationStream.
    before = idle()["cancelled"]
    response = client.post(
        base + "/v1/completions",
        json={"prompt": prompt, "max_tokens": 200, "ignore_eos": True, "stream": True},
        stream=True,
        timeout=30,
    )
    response.raise_for_status()
    for line in response.iter_lines(chunk_size=1):
        if line.startswith(b"data:"):
            break
    response.close()
    m = idle()
    assert m["cancelled"] > before, m
    report["disconnect_cancelled"] = True
    # Enough simultaneous long requests to fill the bounded admission queue.
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        statuses = list(
            pool.map(
                lambda _: (
                    post(
                        {
                            "prompt": "Continue this long story: ",
                            "max_tokens": 250,
                            "ignore_eos": True,
                        }
                    ).status_code
                ),
                range(24),
            )
        )
    assert 429 in statuses, statuses
    assert set(statuses) <= {200, 429}, statuses
    report["queue_capacity_statuses"] = statuses
    m = idle()
    assert m["free_pages"] + m["cached_pages"] == capacity, m
    assert client.get(base + "/health", timeout=2).status_code == 200
    report["idle_page_conservation"] = m
    stop()
    start(max(1, 16 // min(len(devices), config["num_key_value_heads"])))
    r = post({"prompt": [1] * 140, "max_tokens": 20})
    assert r.status_code == 429, r.text
    report["impossible_request_rejected"] = True
    client.get(base + "/health", timeout=2).raise_for_status()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    report["behavior_failures"] = behavior_failures
    report["passed"] = not behavior_failures
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert not behavior_failures, behavior_failures
finally:
    stop()
    log.close()
