"""Build pinned GGUF artifacts and check conversion on synthetic inputs.

This is an infrastructure smoke check, not the registered 100-query screen.
Run on a machine with >=16 GB RAM and >=16 GB free disk. No dataset is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = "codefuse-ai/F2LLM-v2-1.7B"
REVISION = "3766d46e7a68545ed6190c15330983f9b39ab718"
LLAMA_COMMIT = "1c3c9674de4d455f1e571bed808252af54932767"
TEXTS = ["Return the sum of two integers.", "  Unicode café ∑ 日本語\n\t", ""]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def run(command: list[str], work: Path, name: str, *, timeout: int = 3600) -> None:
    import psutil

    started = time.monotonic()
    with (work / f"{name}.log").open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            tree = psutil.Process(process.pid).children(recursive=True)
            for child in tree:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait(timeout=10)
            write_json(work / f"{name}.command.json", {"command": command, "status": "timeout", "seconds": time.monotonic()-started})
            raise
    write_json(work / f"{name}.command.json", {"command": command, "exit_code": returncode, "seconds": time.monotonic()-started})
    if returncode:
        raise RuntimeError(f"{name} failed ({returncode}); inspect {work / (name+'.log')}")


def prepare(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=False)
    source = work / "llama.cpp"
    run(["git", "clone", "--depth", "1", "--branch", "b10269", "https://github.com/ggml-org/llama.cpp.git", str(source)], work, "clone", timeout=300)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    if commit != LLAMA_COMMIT:
        raise RuntimeError(f"release pin mismatch: {commit}")
    run([sys.executable, "-m", "pip", "install", "-r", str(source / "requirements/requirements-convert_hf_to_gguf.txt")], work, "converter-dependencies", timeout=900)
    build = source / "build"
    run(["cmake", "-S", str(source), "-B", str(build), "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_CURL=OFF"], work, "configure", timeout=300)
    run(["cmake", "--build", str(build), "--config", "Release", "--target", "llama-server", "llama-quantize", "-j", "2"], work, "compile", timeout=1800)
    # Download only the pinned model repository, never an evaluation dataset.
    download = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({MODEL!r},revision={REVISION!r},local_dir={str(work/'checkpoint')!r},"
        "allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])"
    )
    run([sys.executable, "-c", download], work, "checkpoint-download", timeout=1800)
    f32 = work / "F2LLM-v2-1.7B-F32.gguf"
    q8 = work / "F2LLM-v2-1.7B-Q8_0.gguf"
    run([sys.executable, str(source / "convert_hf_to_gguf.py"), str(work / "checkpoint"), "--outfile", str(f32), "--outtype", "f32"], work, "convert-f32", timeout=1800)
    run([str(build / "bin/llama-quantize"), str(f32), str(q8), "Q8_0", "2"], work, "quantize-q8", timeout=900)
    files = [f32, q8, build / "bin/llama-server", build / "bin/llama-quantize"]
    files.extend(p for p in (work / "checkpoint").rglob("*") if p.is_file() and ".cache" not in p.parts)
    write_json(work / "manifest.json", {"model": MODEL, "revision": REVISION, "llama_commit": commit,
        "platform": platform.platform(), "files": {str(p.relative_to(work)): sha256(p) for p in files}})
    with (work / "packages.txt").open("x") as log:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=log, check=True, timeout=30)


def worker(work: Path, backend: str) -> None:
    import numpy as np
    from app.retrieval.encode import CodeEncoder

    recipe = CodeEncoder(model_name=str(work / "checkpoint"), revision=REVISION, device="cpu", torch_dtype="float32",
                         tier="accuracy", config_path=ROOT / "config/candidate_v2.yaml")
    if backend == "torch":
        encoder = recipe
        tokens = [encoder.model.tokenize([recipe.get_prompt_text(text, "query")])["input_ids"][0].tolist() for text in TEXTS]
    else:
        from app.retrieval.llama_query import LlamaQueryEncoder
        encoder = LlamaQueryEncoder(binary=work / "llama.cpp/build/bin/llama-server",
            model=work / f"F2LLM-v2-1.7B-{'F32' if backend == 'f32' else 'Q8_0'}.gguf",
            model_name=str(work / "checkpoint"), revision=REVISION, query_prefix=recipe.query_prefix,
            threads=os.cpu_count() or 1, log_path=work / f"{backend}-server.log")
        tokens = [encoder.token_ids(text) for text in TEXTS]
    started = time.monotonic()
    try:
        vectors = encoder.encode(TEXTS, mode="query", batch_size=1)
    finally:
        if backend != "torch":
            encoder.close()
    np.save(work / f"{backend}-vectors.npy", vectors, allow_pickle=False)
    write_json(work / f"{backend}-smoke.json", {"tokens": tokens, "seconds": time.monotonic()-started,
        "backend": backend, "threads": os.cpu_count(), "synthetic_only": True})


def smoke(work: Path) -> None:
    import numpy as np
    results = {}
    for backend in ("torch", "f32", "q8"):
        run([sys.executable, str(Path(__file__).resolve()), "worker", "--work", str(work), "--backend", backend], work, f"{backend}-worker", timeout=900)
        results[backend] = json.loads((work / f"{backend}-smoke.json").read_text())
        if backend != "torch":
            a = np.load(work / "torch-vectors.npy", allow_pickle=False).astype(np.float64)
            b = np.load(work / f"{backend}-vectors.npy", allow_pickle=False).astype(np.float64)
            cos = (a*b).sum(axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
            results[backend]["cosines"] = cos.tolist()
            results[backend]["token_parity"] = results[backend]["tokens"] == results["torch"]["tokens"]
            median_bar, min_bar = ((0.9999, 0.999) if backend == "f32" else (0.995, 0.98))
            passed = results[backend]["token_parity"] and np.median(cos) >= median_bar and cos.min() >= min_bar
            if not passed:
                write_json(work / "smoke-result.json", {"status": "control_failed" if backend == "f32" else "q8_smoke_failed", "results": results, "adoption": False})
                return
    write_json(work / "smoke-result.json", {"status": "synthetic_smoke_passed", "results": results, "adoption": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "worker"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--backend", choices=["torch", "f32", "q8"])
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "prepare":
        prepare(work)
    elif args.command == "smoke":
        smoke(work)
    else:
        worker(work, args.backend)


if __name__ == "__main__":
    main()
