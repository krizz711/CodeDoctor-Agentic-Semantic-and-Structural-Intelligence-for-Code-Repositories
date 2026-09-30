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
with torch.no_grad():
    for l, w in zip(lin, orig):
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        l.weight.data = (w / s).round().clamp(-127, 127) * s
v8 = enc_all()
print("W8 per-channel: vs fp32 min/median", cos(v8, v32).min().round(5), np.median(cos(v8, v32)).round(5),
      "vs stored median", np.median(cos(v8, stored)).round(5), flush=True)
with torch.no_grad():
    for l, w in zip(lin, orig):
        o, i = w.shape
        g = w.reshape(o, i // 128, 128)
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        s = (hi - lo).clamp_min(1e-12) / 15
        q = ((g - lo) / s).round().clamp(0, 15) * s + lo
        l.weight.data = q.reshape(o, i)
v4 = enc_all()
print("W4 g128: vs fp32 min/median", cos(v4, v32).min().round(5), np.median(cos(v4, v32)).round(5),
      "vs stored median", np.median(cos(v4, stored)).round(5), flush=True)
