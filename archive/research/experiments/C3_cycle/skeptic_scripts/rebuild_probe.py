"""Does the workbench re-index after a change without a restart, and what does a lexical rebuild cost?"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\Asus\manish\CodeDoctor")
from app.indexing.cache import EmbeddingCache  # noqa: E402
from app.workbench import Workbench  # noqa: E402


def wait(bench, limit=300):
    t = time.perf_counter()
    while True:
        s = bench.state()
        if s["phase"] in {"ready", "error"}:
            return s, time.perf_counter() - t
        if time.perf_counter() - t > limit:
            return s, -1
        time.sleep(0.05)


tmp = Path(tempfile.mkdtemp(prefix="skeptic_wb_"))
(tmp / "m.py").write_text("def alpha(x):\n    return x + 1\n", encoding="utf-8")
bench = Workbench(cache=EmbeddingCache(":memory:"))
info = bench.open_folder(str(tmp))
key = info.get("key") or info.get("target", {}).get("key")
print("open_folder ->", {k: info[k] for k in list(info)[:6]})
bench.start_build(key, dense=False)
s, dt = wait(bench)
hits = bench.search(key, "alpha", limit=5)
names1 = [h.get("name") for h in hits.get("results", hits.get("hits", []))]
print(f"build 1: {dt:.3f}s phase={s['phase']} chunks={s['chunks']} names={names1}")

(tmp / "m.py").write_text("def beta(x):\n    return x + 2\n", encoding="utf-8")
ret = bench.start_build(key, dense=False)
s, dt = wait(bench)
hits = bench.search(key, "beta", limit=5)
names2 = [h.get("name") for h in hits.get("results", hits.get("hits", []))]
print(f"build 2 after edit: returned phase={ret['phase']} {dt:.3f}s names_for_beta={names2}")
print("RE-INDEXED AFTER EDIT" if "beta" in names2 else "NOT RE-INDEXED: start_build returned early on a ready target")
shutil.rmtree(tmp, ignore_errors=True)

# Lexical-only rebuild cost of this repository's app/ as a folder target (read-only).
bench2 = Workbench(cache=EmbeddingCache(":memory:"))
info = bench2.open_folder(r"C:\Users\Asus\manish\CodeDoctor\app")
key2 = info.get("key") or info.get("target", {}).get("key")
t = time.perf_counter()
bench2.start_build(key2, dense=False)
s, dt = wait(bench2)
print(f"folder app/ lexical build: {dt:.2f}s, chunks={s['chunks']}")

# Git target 'self', 4 versions, lexical only (git ls-tree + cat-file per file per version).
bench3 = Workbench(cache=EmbeddingCache(":memory:"))
bench3.start_build("self", versions=4, dense=False)
s, dt = wait(bench3)
print(f"git target self x4 versions lexical build: {dt:.2f}s, chunks={s['chunks']} phase={s['phase']} err={s['error']}")
