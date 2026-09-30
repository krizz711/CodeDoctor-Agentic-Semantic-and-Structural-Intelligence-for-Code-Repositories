"""Read-only falsifying probe: torch dynamic int8 on the F2LLM-v2-0.6B query tower, local CPU.

Same seeded TUNE statements as bench_query_cpu (seed 20260925 used by R2? we use seed 7; documented).
fp32 vs int8 (torch.ao dynamic, nn.Linear only), batch 1, torch threads = 6.
Fidelity vs stored GPU vectors; latency per query. No test-split file is opened.
"""
import os, sys, time, json
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np
import torch
ROOT = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(ROOT))
from scripts.bench_query_cpu import _load_inputs, MODELS  # noqa: E402
from app.retrieval.encode import CodeEncoder  # noqa: E402

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
tag = "f2llm06b"
rows, texts, stored = _load_inputs(tag, N, 7, ROOT / "artifacts" / "apps", None)
name, rev, _ = MODELS[tag]
enc = CodeEncoder(model_name=name, revision=rev, device="cpu", torch_dtype="float32")
enc.encode(texts[:1], mode="query")


def run(label):
    for t in texts[:2]:
        enc.encode([t], mode="query")
    lat, vec = [], []
    for t in texts:
        s = time.perf_counter(); vec.append(enc.encode([t], mode="query")[0]); lat.append(time.perf_counter() - s)
    v = np.asarray(vec, np.float32)
    cos = (v * stored).sum(1) / np.linalg.norm(v, axis=1) / np.linalg.norm(stored, axis=1)
    print(label, "threads", torch.get_num_threads(), "mean s", round(float(np.mean(lat)), 3),
          "p50", round(float(np.median(lat)), 3), "cos vs stored min/median",
          round(float(cos.min()), 5), round(float(np.median(cos)), 5), flush=True)
    return v, np.array(lat)


v32, l32 = run("fp32")
m = enc.model
torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
v8, l8 = run("int8-dynamic")
c = (v8 * v32).sum(1) / np.linalg.norm(v8, axis=1) / np.linalg.norm(v32, axis=1)
print("int8 vs fp32-cpu cos min/median", round(float(c.min()), 5), round(float(np.median(c)), 5),
      "speedup (mean)", round(float(l32.mean() / l8.mean()), 2), "chars median", int(np.median([len(t) for t in texts])))
