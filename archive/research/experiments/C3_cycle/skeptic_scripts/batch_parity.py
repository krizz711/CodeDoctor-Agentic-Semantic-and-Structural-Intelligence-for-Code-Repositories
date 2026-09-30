"""Is a CPU vector the same whether a doc is encoded alone (incremental) or in a batch (full rebuild)?"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\Asus\manish\CodeDoctor")
sys.path.insert(0, str(REPO))
from app.retrieval.encode import CodeEncoder  # noqa: E402

rows = []
with open(REPO / "artifacts/apps/texts_corpus.jsonl", encoding="utf-8") as fh:
    for line in fh:
        rows.append(json.loads(line)["text"])
rng = np.random.default_rng(7)
idx = sorted(rng.choice(len(rows), size=24, replace=False).tolist())
texts = [rows[i] for i in idx]
enc = CodeEncoder(device="cpu", tier="interactive")
_ = enc.model
t = time.perf_counter()
single = np.vstack([enc.encode(x, mode="document") for x in texts])
t1 = time.perf_counter() - t
t = time.perf_counter()
batched = enc.encode(texts, mode="document")  # length-sorted batches, workbench uses windows of 8
t2 = time.perf_counter() - t
t = time.perf_counter()
win8 = np.vstack([enc.encode(texts[s:s + 8], mode="document") for s in range(0, len(texts), 8)])
t3 = time.perf_counter() - t
again = np.vstack([enc.encode(x, mode="document") for x in texts[:6]])
cos_b = np.sum(single * batched, axis=1)
cos_w = np.sum(single * win8, axis=1)
print(f"24 random corpus docs: batch-1 {t1:.1f}s ({t1/24:.3f}s/doc), full batch {t2:.1f}s, windows of 8 {t3:.1f}s")
print(f"cos(single, full batch): min {cos_b.min():.6f} median {np.median(cos_b):.6f}; bit-identical rows {int(np.sum(np.all(single == batched, axis=1)))}/24")
print(f"cos(single, win8): min {cos_w.min():.6f}; bit-identical rows {int(np.sum(np.all(single == win8, axis=1)))}/24")
print(f"repeat batch-1 bit-identical: {bool(np.array_equal(single[:6], again))}")
