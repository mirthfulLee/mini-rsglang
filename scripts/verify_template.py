"""Checkpoint chat-template byte and token equality against Hugging Face."""

import argparse
import json
import pathlib
import subprocess

from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/models/store/Qwen/Qwen3-0.6B")
parser.add_argument("--output", default="results")
args = parser.parse_args()
model = args.model
output = ROOT / args.output
output.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
fixtures = [
    [{"role": "user", "content": "你好！"}],
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Explain Rust."},
    ],
    [
        {"role": "user", "content": "2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "reasoning_content": "Add the two numbers.",
        },
        {"role": "user", "content": "Why?"},
    ],
]
report = []
for i, messages in enumerate(fixtures):
    path = output / f"template-{i}.json"
    path.write_text(json.dumps(messages))
    for thinking in [False, True]:
        raw = subprocess.check_output(
            [
                str(pathlib.Path.home() / ".cargo/bin/cargo"),
                "run",
                "--release",
                "-q",
                "-p",
                "rsglang-runtime",
                "--example",
                "template",
                "--",
                model,
                str(path),
                str(thinking).lower(),
            ],
            cwd=ROOT,
        )
        actual = json.loads(raw)
        expected = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        assert actual["text"] == expected, (i, thinking, actual["text"], expected)
        assert actual["ids"] == tok.encode(expected), (i, thinking)
        report.append(
            {
                "case": i,
                "enable_thinking": thinking,
                "tokens": len(actual["ids"]),
                "equal": True,
            }
        )
(output / "template-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
