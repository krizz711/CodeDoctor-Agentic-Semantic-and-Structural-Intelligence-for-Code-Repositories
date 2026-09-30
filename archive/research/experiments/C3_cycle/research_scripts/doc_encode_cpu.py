"""Read-only: per-document CPU encode cost at 0.6B (interactive tier) on this laptop.
40 seeded APPS corpus docs, batch 1, float32; fidelity vs stored GPU vectors."""
import json, random, sys, time, os
from pathlib import Path
import numpy as np
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
import torch
torch.set_num_threads(6)
from app.retrieval.encode import CodeEncoder
ids = [json.loads(l)["id"] for l in open(REPO / "artifacts/apps/texts_corpus.jsonl", encoding="utf-8")]
texts = [json.loads(l)["text"] for l in open(REPO / "artifacts/apps/texts_corpus.jsonl", encoding="utf-8")]
stored = np.load(REPO / "artifacts/apps/emb_f2llm06b_corpus.npy", mmap_mode="r")
rows = random.Random(20260926).sample(range(len(texts)), 40)
t0 = time.time()
enc = CodeEncoder(device="cpu", tier="interactive")
enc.encode([texts[rows[0]]], mode="document")  # load + warm-up
load = time.time() - t0
times, cos = [], []
for r in rows:
    s = time.perf_counter(); v = np.asarray(enc.encode([texts[r]], mode="document"), dtype=np.float32)[0]
    times.append(time.perf_counter() - s)
    g = np.asarray(stored[r], dtype=np.float32)
    cos.append(float(v @ g / np.linalg.norm(v) / np.linalg.norm(g)))
try:
    import psutil; rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
except ImportError:
    rss = None
lens = [len(texts[r]) for r in rows]
out = {"model": getattr(enc, "model_name", "?"), "threads": torch.get_num_threads(), "load_plus_warmup_s": round(load, 1),
       "per_doc_mean_s": round(float(np.mean(times)), 3), "per_doc_median_s": round(float(np.median(times)), 3),
       "per_doc_p95_s": round(float(np.percentile(times, 95)), 3), "chars_mean": float(np.mean(lens)),
       "cos_vs_stored_median": round(float(np.median(cos)), 6), "cos_min": round(float(np.min(cos)), 6),
       "rss_gb_end": rss, "corr_len_time": float(np.corrcoef(lens, times)[0, 1])}
print(json.dumps(out, indent=1))
