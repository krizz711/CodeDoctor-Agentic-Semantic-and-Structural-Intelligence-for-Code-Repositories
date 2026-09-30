"""R2/M2: what does encoding one query cost on a CPU, per encoder?

Pre-registration: ``experiments/R2_encoder_choice/prereg.md``.

    python scripts/bench_query_cpu.py --out experiments/R2_encoder_choice/cpu_latency.json

Each model runs in its own subprocess so its memory is measured alone.  A model
is not loaded when its weights would exceed 80% of the RAM available at that
moment -- that is recorded as the result instead of letting the machine swap.
Vectors are compared with the stored GPU vectors for the same statements, so a
CPU operating point that changes the embeddings is visible.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS = {
    "f2llm160m": ("codefuse-ai/F2LLM-v2-160M", "4ffe22c31406fc321b65a05b15563d38275ee51f", 159_185_024),
    "f2llm06b": ("codefuse-ai/F2LLM-v2-0.6B", "54b4e2dc74e01be7126d4cf5f016af6b21edc563", 596_049_920),
    "f2llm17b": ("codefuse-ai/F2LLM-v2-1.7B", "3766d46e7a68545ed6190c15330983f9b39ab718", 1_720_574_976),
}
BYTES = {"float32": 4, "bfloat16": 2}
#: pre-registered stop (R2 prereg, "Pointlessness stop")
STOP_AFTER, STOP_SECONDS = 10, 60.0
HEADROOM_BYTES = 2_000_000_000


def _statements(artifacts: Path, n: int, seed: int) -> tuple[list[int], list[str]]:
    from app.pipeline.splits import make_development_split
    from scripts.dev_eval import load_jsonl_texts

    _, texts = load_jsonl_texts(artifacts / "texts_queries_train.jsonl")
    split = make_development_split(texts, match_family=None, seed=20260919)
    rng = np.random.default_rng(seed)
    rows = sorted(int(i) for i in rng.choice(split.tune, size=n, replace=False))
    return rows, [texts[i] for i in rows]


def _peak_bytes() -> int:
    import psutil

    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", 0) or info.rss)


def build_subset(artifacts: Path, n: int, seed: int, out: Path) -> None:
    """The seeded statements and their stored GPU vectors, for a machine without artifacts."""
    rows, texts = _statements(artifacts, n, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "subset.json").write_text(json.dumps({"rows": rows, "texts": texts, "n": n, "seed": seed}), encoding="utf-8")
    for tag in MODELS:
        stored = artifacts / f"emb_{tag}_qtrain.npy"
        if stored.exists():
            np.save(out / f"stored_{tag}.npy", np.load(stored)[rows])


def _load_inputs(tag: str, n: int, seed: int, artifacts: Path, subset: Path | None):
    if subset is not None:
        meta = json.loads((subset / "subset.json").read_text(encoding="utf-8"))
        stored_path = subset / f"stored_{tag}.npy"
        stored = np.load(stored_path).astype(np.float32) if stored_path.exists() else None
        return meta["rows"][:n], meta["texts"][:n], stored
    rows, texts = _statements(artifacts, n, seed)
    stored_path = artifacts / f"emb_{tag}_qtrain.npy"
    stored = np.load(stored_path).astype(np.float32)[rows] if stored_path.exists() else None
    return rows, texts, stored


def run_one(tag: str, dtype: str, n: int, seed: int, artifacts: Path, subset: Path | None = None) -> dict:
    import psutil
    import torch

    from app.retrieval.encode import CodeEncoder

    name, revision, params = MODELS[tag]
    available = psutil.virtual_memory().available
    need = params * BYTES[dtype]
    result = {"tag": tag, "model": name, "revision": revision, "dtype": dtype,
              "weights_gb": round(need / 1e9, 2), "available_gb_before": round(available / 1e9, 2),
              "threads": torch.get_num_threads()}
    # Weights plus a fixed margin for activations: long statements at 1.7B
    # otherwise push this 7.7 GiB machine into swap (safety, not a metric).
    if need > 0.8 * available or need + HEADROOM_BYTES > available:
        result["status"] = "not_loaded"
        result["reason"] = (f"{dtype} weights need {need / 1e9:.1f} GB (+{HEADROOM_BYTES / 1e9:.0f} GB headroom); "
                            f"only {available / 1e9:.1f} GB available")
        return result

    rows, texts, stored = _load_inputs(tag, n, seed, artifacts, subset)
    t0 = time.perf_counter()
    encoder = CodeEncoder(model_name=name, revision=revision, device="cpu", torch_dtype=dtype)
    encoder.encode(texts[:1], mode="query")  # forces the load; also a warm-up
    result["load_seconds"] = round(time.perf_counter() - t0, 2)
    for text in texts[1:3]:
        encoder.encode([text], mode="query")

    latencies, vectors = [], []
    for text in texts:
        t = time.perf_counter()
        vectors.append(encoder.encode([text], mode="query")[0])
        latencies.append(time.perf_counter() - t)
        if len(latencies) == STOP_AFTER and float(np.mean(latencies)) > STOP_SECONDS:
            result["stopped_early"] = (f"first {STOP_AFTER} statements averaged "
                                       f"{np.mean(latencies):.1f} s (> {STOP_SECONDS} s): pre-registered stop")
            break
    measured = texts[: len(latencies)]
    rows = rows[: len(latencies)]
    t = time.perf_counter()
    encoder.encode(measured, mode="query")
    batched = time.perf_counter() - t

    lat = np.array(latencies)
    result.update({
        "status": "ok",
        "statements": len(measured),
        "chars_median": int(np.median([len(x) for x in measured])),
        "batch1_seconds": {"p50": round(float(np.percentile(lat, 50)), 3),
                           "p95": round(float(np.percentile(lat, 95)), 3),
                           "mean": round(float(lat.mean()), 3), "max": round(float(lat.max()), 3)},
        "batched_seconds_per_query": round(batched / len(measured), 3),
        "peak_memory_gb": round(_peak_bytes() / 1e9, 2),
    })
    if stored is not None:
        stored = stored[: len(vectors)]
        mine = np.asarray(vectors, dtype=np.float32)
        cos = (mine * stored).sum(1) / np.linalg.norm(mine, axis=1) / np.linalg.norm(stored, axis=1)
        result["cosine_vs_stored_gpu_vectors"] = {"min": round(float(cos.min()), 5),
                                                  "median": round(float(np.median(cos)), 5)}
    del encoder
    gc.collect()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--plan", default="f2llm160m:float32,f2llm06b:float32,f2llm17b:float32,f2llm17b:bfloat16")
    parser.add_argument("--one", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--subset", type=Path, default=None,
                        help="a folder from --build-subset (for machines without the artifacts)")
    parser.add_argument("--build-subset", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "experiments" / "R2_encoder_choice" / "cpu_latency.json")
    args = parser.parse_args(argv)

    if args.build_subset:
        build_subset(args.artifacts, args.n, args.seed, args.build_subset)
        print(f"subset -> {args.build_subset}")
        return 0
    if args.one:
        tag, dtype = args.one.split(":")
        print(json.dumps(run_one(tag, dtype, args.n, args.seed, args.artifacts, args.subset)))
        return 0

    import psutil

    results = []
    for item in args.plan.split(","):
        print(f"--- {item}", flush=True)
        command = [sys.executable, __file__, "--one", item, "--n", str(args.n),
                   "--seed", str(args.seed), "--artifacts", str(args.artifacts)]
        if args.subset is not None:
            command += ["--subset", str(args.subset)]
        proc = subprocess.run(command,
                              capture_output=True, text=True, encoding="utf-8", check=False)
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if proc.returncode != 0 or not lines:
            results.append({"item": item, "status": "failed", "returncode": proc.returncode,
                            "stderr_tail": proc.stderr[-2000:]})
        else:
            results.append(json.loads(lines[-1]))
        print(json.dumps(results[-1]), flush=True)
    import torch

    machine = {"platform": platform.platform(), "processor": platform.processor(),
               "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
               "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
               "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}
    out = {"prereg": "experiments/R2_encoder_choice/prereg.md", "machine": machine, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(args.out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
