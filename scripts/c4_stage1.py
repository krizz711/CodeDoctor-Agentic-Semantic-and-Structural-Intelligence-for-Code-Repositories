"""C4-E1 preregistered 100-query fidelity and warm-cost screen.

Run only after the F32/Q8 conversion smoke succeeds. Inputs must be the
train-only C4 dataset. Each arm runs in its own process; no decision set is
shrunk on failure. Writes raw vectors, tokens, latencies and process-tree RSS.
"""
from __future__ import annotations

import argparse
import json
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.c4_llama_smoke import MODEL, REVISION, sha256  # noqa: E402


def saved(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def query_sample(data: Path) -> tuple[list[str], list[str]]:
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "C4-E1 train-only inputs" or manifest.get("seed") != 20260929:
        raise ValueError("incorrect C4 input manifest")
    source = data / "tune100_seed20260929.json"
    if sha256(source) != manifest["files"][source.name]["sha256"]:
        raise ValueError("C4 TUNE sample hash mismatch")
    sample = json.loads(source.read_text(encoding="utf-8"))
    if len(sample["qids"]) != 100 or len(set(sample["qids"])) != 100 or len(sample["texts"]) != 100:
        raise ValueError("C4 TUNE sample must contain 100 ordered unique queries")
    return sample["qids"], sample["texts"]


def edge_texts() -> list[str]:
    return ["", " ", " \n\t ", "naïve café 👩🏽‍💻 日本語", "line1\n\nline2\t", "<|im_end|>", "word " * 9000]


def worker(backend: str, data: Path, models: Path, out: Path) -> None:
    from app.retrieval.encode import CodeEncoder

    ids, texts = query_sample(data)
    recipe = CodeEncoder(model_name=str(models / "checkpoint"), revision=REVISION,
        config_path=ROOT / "config/candidate_v2.yaml", device="cpu", torch_dtype="float32", tier="accuracy")
    if backend == "torch":
        encoder = recipe
        def tokens(value):
            return recipe.model.tokenize([recipe.get_prompt_text(value, "query")])["input_ids"][0].tolist()
    else:
        from app.retrieval.llama_query import LlamaQueryEncoder

        encoder = LlamaQueryEncoder(binary=models / "llama.cpp/build/bin/llama-server",
            model=models / f"F2LLM-v2-1.7B-{'F32' if backend == 'f32' else 'Q8_0'}.gguf",
            model_name=str(models / "checkpoint"), revision=REVISION,
            query_prefix=recipe.query_prefix, max_seq_length=recipe.max_seq_length,
            threads=os.cpu_count() or 1, startup_timeout=180, request_timeout=180,
            log_path=out / f"{backend}-server.log")
        tokens = encoder.token_ids
    startup = time.monotonic()
    try:
        if backend == "torch":
            _ = encoder.model
        else:
            encoder.start()
        cold_startup = time.monotonic() - startup
        # Exact tokenizer IDs for every decision query and synthetic edge case.
        token_ids = [tokens(text) for text in texts]
        edge_ids = [tokens(text) for text in edge_texts()]
        encoder.encode([texts[0]], mode="query", batch_size=1)  # fixed warm-up
        vectors, seconds = [], []
        for text in texts:
            started = time.monotonic()
            vectors.append(encoder.encode([text], mode="query", batch_size=1)[0])
            seconds.append(time.monotonic() - started)
    finally:
        if backend != "torch":
            encoder.close()
    np.save(out / f"{backend}-vectors.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    saved(out / f"{backend}-raw.json", {"backend": backend, "qids": ids, "seconds": seconds,
        "token_ids": token_ids, "edge_ids": edge_ids,
        "cold_startup_seconds": cold_startup, "warmup_queries": 1,
        "threads": os.cpu_count(), "machine": platform.platform(),
        "model": MODEL, "revision": REVISION})


def run_arm(backend: str, data: Path, models: Path, out: Path) -> dict:
    import psutil

    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--backend", backend,
               "--data", str(data), "--models", str(models), "--out", str(out)]
    started = time.monotonic()
    with (out / f"{backend}-worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        peak = 0
        deadline = time.monotonic() + 5400
        try:
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    tree = [psutil.Process(process.pid)]
                    tree += tree[0].children(recursive=True)
                    peak = max(peak, sum(p.memory_info().rss for p in tree if p.is_running()))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                time.sleep(0.05)
            if process.poll() is None:
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
                process.kill()
            code = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    result = {"backend": backend, "command": command, "exit_code": code,
              "seconds": time.monotonic()-started, "peak_process_tree_rss_bytes": peak}
    saved(out / f"{backend}-process.json", result)
    if code:
        raise RuntimeError(f"{backend} worker failed; inspect its log")
    return result


def compare(out: Path, baseline: dict, treatment: dict | None = None) -> dict:
    raw = {name: json.loads((out / f"{name}-raw.json").read_text()) for name in (["torch", "f32"] + (["q8"] if treatment else []))}
    names = list(raw)
    if any(raw[n]["qids"] != raw["torch"]["qids"] for n in names[1:]):
        return {"status": "void", "reason": "query-order mismatch"}
    if raw["f32"]["token_ids"] != raw["torch"]["token_ids"] or raw["f32"]["edge_ids"] != raw["torch"]["edge_ids"]:
        return {"status": "void", "reason": "token-ID mismatch"}
    if treatment and (raw["q8"]["token_ids"] != raw["torch"]["token_ids"] or raw["q8"]["edge_ids"] != raw["torch"]["edge_ids"]):
        return {"status": "negative", "reason": "Q8 token-ID mismatch", "adoption": False}
    base = np.load(out / "torch-vectors.npy", allow_pickle=False).astype(np.float64)
    if base.shape != (100, 2048) or not np.isfinite(base).all() or np.any(np.linalg.norm(base, axis=1) == 0):
        return {"status": "void", "reason": "invalid Torch control vectors"}
    metrics = {}
    for name in names[1:]:
        vec = np.load(out / f"{name}-vectors.npy", allow_pickle=False).astype(np.float64)
        if vec.shape != base.shape or vec.shape != (100, 2048) or not np.isfinite(vec).all():
            return {"status": "void" if name == "f32" else "negative", "reason": f"invalid {name} vectors"}
        cos = (base*vec).sum(axis=1)/(np.linalg.norm(base, axis=1)*np.linalg.norm(vec, axis=1))
        metrics[name] = {"median_cosine": float(np.median(cos)), "minimum_cosine": float(np.min(cos)),
                         "cosines": cos.tolist(), "mean_seconds": float(np.mean(raw[name]["seconds"]))}
    if metrics["f32"]["median_cosine"] < .9999 or metrics["f32"]["minimum_cosine"] < .999:
        return {"status": "void", "reason": "F32 conversion control failed", "metrics": metrics}
    if treatment is None:
        return {"status": "f32_control_passed", "metrics": metrics}
    q8 = metrics["q8"]
    faster = np.mean(raw["q8"]["seconds"]) < np.mean(raw["torch"]["seconds"])
    smaller = treatment["peak_process_tree_rss_bytes"] < baseline["peak_process_tree_rss_bytes"]
    q8.update({"faster_than_torch": bool(faster), "less_memory_than_torch": bool(smaller),
        "torch_mean_seconds": float(np.mean(raw["torch"]["seconds"])),
        "torch_peak_rss_bytes": baseline["peak_process_tree_rss_bytes"],
        "q8_peak_rss_bytes": treatment["peak_process_tree_rss_bytes"]})
    passed = q8["median_cosine"] >= .995 and q8["minimum_cosine"] >= .98 and faster and smaller
    return {"status": "screen_passed" if passed else "negative", "metrics": metrics, "adoption": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "worker"])
    parser.add_argument("--backend", choices=["torch", "f32", "q8"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data, models, out = args.data.resolve(), args.models.resolve(), args.out.resolve()
    if args.command == "worker":
        if args.backend is None:
            parser.error("worker needs --backend")
        worker(args.backend, data, models, out)
        return
    out.mkdir(parents=True, exist_ok=False)
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8") if Path("/proc/cpuinfo").exists() else ""
    cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), platform.processor())
    flags = next((line.split(":", 1)[1].split() for line in cpuinfo.splitlines() if line.startswith("flags")), [])
    versions = {}
    for package in ("torch", "transformers", "sentence-transformers", "psutil", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    saved(out / "run-manifest.json", {"data_manifest_sha256": sha256(data / "MANIFEST.json"),
        "model_manifest_sha256": sha256(models / "manifest.json"),
        "binary_sha256": sha256(models / "llama.cpp/build/bin/llama-server"),
        "cpu_model": cpu_model, "isa_flags": sorted(set(flags)),
        "platform": platform.platform(), "python": sys.version, "packages": versions,
        "seed": 20260929, "stage": 1, "adoption": False})
    baseline = run_arm("torch", data, models, out)
    run_arm("f32", data, models, out)
    first = compare(out, baseline)
    if first["status"] != "f32_control_passed":
        saved(out / "stage1-result.json", first)
        return
    treatment = run_arm("q8", data, models, out)
    saved(out / "stage1-result.json", compare(out, baseline, treatment))


if __name__ == "__main__":
    main()
