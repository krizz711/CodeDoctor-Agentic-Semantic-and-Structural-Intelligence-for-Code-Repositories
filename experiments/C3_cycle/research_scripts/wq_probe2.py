"""Read-only fidelity probe: weight-only fake quantization of F2LLM-v2-0.6B Linear weights (activations fp32).
W8 per-output-channel symmetric; W4 asymmetric group-128. Fidelity only (no fast kernel here, so no speed claim)."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np
import torch
ROOT = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(ROOT))
from scripts.bench_query_cpu import _load_inputs, MODELS  # noqa: E402
from app.retrieval.encode import CodeEncoder  # noqa: E402

torch.set_num_threads(6)
rows, texts, stored = _load_inputs("f2llm06b", 8, 7, ROOT / "artifacts" / "apps", None)
name, rev, _ = MODELS["f2llm06b"]
enc = CodeEncoder(model_name=name, revision=rev, device="cpu", torch_dtype="float32")
enc.encode(texts[:1], mode="query")
m = enc.model
lin = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
orig = [l.weight.data.clone() for l in lin]


def enc_all():
    return np.asarray([enc.encode([t], mode="query")[0] for t in texts], np.float32)


def cos(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


v32 = enc_all()
print("fp32 vs stored median", round(float(np.median(cos(v32, stored))), 5), flush=True)
def blockq(w, bs, bits):
    o, i = w.shape
    g = w.reshape(o, i // bs, bs)
    qmax = 2 ** (bits - 1) - 1
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return ((g / s).round().clamp(-qmax, qmax) * s).reshape(o, i)
for bs, bits in ((32, 8), (128, 8), (32, 4)):
    with torch.no_grad():
        for l, w in zip(lin, orig):
            l.weight.data = blockq(w, bs, bits)
    v = enc_all()
    print(f"W{bits} sym block{bs}: vs fp32 min/median", cos(v, v32).min().round(5), np.median(cos(v, v32)).round(5), flush=True)
# per-layer sensitivity at W8 per-channel: quantize one Linear type at a time
names = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
    with torch.no_grad():
        for n, l, w in zip(names, lin, orig):
            l.weight.data = blockq(w, w.shape[1], 8) if n.endswith(key) else w
    v = enc_all()
    print(f"W8 per-channel only {key}: median cos", np.median(cos(v, v32)).round(5), flush=True)
