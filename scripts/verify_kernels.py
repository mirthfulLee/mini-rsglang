"""Independent PyTorch oracle for every custom CUDA operator and mixed paged GQA."""

import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="results/kernels")
args = parser.parse_args()
out = ROOT / args.output
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def random(*shape):
    return torch.randn(shape, device="cuda").bfloat16()


f = {
    k: random(*s)
    for k, s in {
        "routing": (5, 7),
        "expert_input": (5, 19),
        "norm_x": (3, 133),
        "norm_w": (133,),
        "linear_x": (3, 19),
        "linear_w": (7, 19),
        "gate": (3, 133),
        "up": (3, 133),
        "embedding": (3, 17),
        "q": (42, 6, 128),
        "k": (42, 2, 128),
        "v": (42, 2, 128),
        "kc": (9, 16, 2, 128),
        "vc": (9, 16, 2, 128),
    }.items()
}
(out / "input.json").write_text(
    json.dumps({k: v.float().cpu().reshape(-1).tolist() for k, v in f.items()})
)
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
subprocess.run(
    [
        str(pathlib.Path.home() / ".cargo/bin/cargo"),
        "run",
        "--release",
        "-p",
        "rsglang-kernels",
        "--example",
        "verify_ops",
        "--",
        str(out / "input.json"),
        str(out / "output.json"),
    ],
    cwd=ROOT,
    env=env,
    check=True,
)
got = json.loads((out / "output.json").read_text())
x = f["norm_x"]
expected = {
    "norm": (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    * f["norm_w"],
    "linear": f["linear_x"] @ f["linear_w"].T,
    "logits": f["linear_x"].float() @ f["linear_w"].float().T,
    "add": f["gate"] + f["up"],
    "swiglu": torch.nn.functional.silu(f["gate"]) * f["up"],
    "embedding": f["embedding"][torch.tensor([1] * 5 + [2] * 37, device="cuda")],
}
pos = torch.tensor(list(range(17, 22)) + list(range(37)), device="cuda")
freq = 1 / (1000000.0 ** (torch.arange(0, 128, 2, device="cuda").float() / 128))
angles = pos[:, None].float() * freq[None, :]
cos = angles.cos().bfloat16()[:, None, :]
sin = angles.sin().bfloat16()[:, None, :]
q = f["q"]
a, b = q[..., :64], q[..., 64:]
q = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)
expected["rope"] = q
kc = f["kc"].clone()
vc = f["vc"].clone()
tables = [[2, 5], [0, 8, 3]]
for i, p in enumerate(pos.tolist()):
    seq = 0 if i < 5 else 1
    page = tables[seq][p // 16]
    off = p % 16
    kc[page, off] = f["k"][i]
    vc[page, off] = f["v"][i]
expected.update(kc=kc, vc=vc, last=q[[4, 41]])
attn = []
for i, p in enumerate(pos.tolist()):
    seq = 0 if i < 5 else 1
    k = torch.stack(
        [kc[tables[seq][j // 16], j % 16] for j in range(p + 1)]
    ).repeat_interleave(3, dim=1)
    v = torch.stack(
        [vc[tables[seq][j // 16], j % 16] for j in range(p + 1)]
    ).repeat_interleave(3, dim=1)
    scores = torch.einsum("hd,thd->ht", q[i].float(), k.float()) / (128**0.5)
    attn.append(torch.einsum("ht,thd->hd", scores.softmax(-1), v.float()).bfloat16())
expected["attention"] = torch.stack(attn)
report = {}
for name, ref in expected.items():
    r = ref.float().cpu().reshape(-1).numpy()
    g = np.array(got[name], dtype=np.float32).reshape(-1)
    assert g.shape == r.shape, name
    tolerance = 0.02 if name in ("norm", "rope", "attention", "swiglu") else 0.0001
    assert np.allclose(
        g, r, atol=tolerance, rtol=0.01 if tolerance > 0.001 else 0.0001
    ), (name, float(np.max(np.abs(g - r))))
    report[name] = {
        "max_abs_error": float(np.max(np.abs(g - r))),
        "nrmse": float(np.linalg.norm(g - r) / np.linalg.norm(r)),
    }
routing = f["routing"].float().softmax(-1)
weights, ids = routing.topk(3, dim=-1)
for normalized in [False, True]:
    selected = (
        weights / weights.sum(-1, keepdim=True) if normalized else weights
    ).bfloat16()
    expected_routes = {}
    for row in range(5):
        for j in range(3):
            expert = int(ids[row, j])
            item = expected_routes.setdefault(
                expert, {"expert": expert, "rows": [], "weights": []}
            )
            item["rows"].append(row)
            item["weights"].append(float(selected[row, j]))
    actual = got["routing"][int(normalized)]
    assert actual == [expected_routes[e] for e in sorted(expected_routes)], (
        actual,
        expected_routes,
    )
    mixture = torch.zeros((5, 19), device="cuda", dtype=torch.float32)
    for expert, route in sorted(expected_routes.items()):
        rows = route["rows"]
        weighted = (
            (f["expert_input"][rows] * (2**expert))
            * torch.tensor(route["weights"], device="cuda", dtype=torch.bfloat16)[
                :, None
            ]
        ).float()
        mixture[rows] += weighted
    ref = mixture.bfloat16().float().cpu().numpy().reshape(-1)
    actual = np.asarray(got["mixtures"][int(normalized)], dtype=np.float32)
    assert np.array_equal(ref, actual), float(np.max(np.abs(ref - actual)))
    report[f"moe_routing_scatter_normalized_{normalized}"] = {
        "max_abs_error": 0.0,
        "selected_experts": len(expected_routes),
    }
repeated = expected["linear"].float().cpu().numpy().reshape(-1)
assert np.allclose(
    np.asarray(got["blocked_linear"]), np.tile(repeated, 11), atol=1e-4, rtol=1e-4
)
report["gemm_row_block_boundary"] = {"rows": 33, "passed": True}
for part in got["fused_split"]:
    assert np.allclose(
        np.asarray(part),
        expected["linear"].float().cpu().numpy().reshape(-1),
        atol=1e-4,
        rtol=1e-4,
    )
report["fused_gemm_split"] = {"parts": 2, "passed": True}
assert got["argmax"] == expected["logits"].argmax(-1).cpu().tolist()
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
