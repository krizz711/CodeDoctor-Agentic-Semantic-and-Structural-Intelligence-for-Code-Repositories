"""Build a private Kaggle conversion/synthetic-smoke notebook, without datasets.

python scripts/build_c4_kernel.py --owner lifemeansfun
kaggle kernels push -p build/kernel_c4_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.packaging import build_payload  # noqa: E402


def build(owner: str, out: Path) -> None:
    payload, included = build_payload(("app/__init__.py", "app/retrieval/__init__.py",
                                       "app/retrieval/encode.py", "app/retrieval/llama_query.py",
                                       "config/candidate_v2.yaml", "config/frozen.yaml",
                                       "scripts/__init__.py", "scripts/c4_llama_smoke.py"))
    out.mkdir(parents=True, exist_ok=True)
    code = f'''# C4-E1 conversion infrastructure smoke; no decision dataset or adoption.
import base64, io, os, subprocess, sys, tarfile
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
subprocess.run([sys.executable, "-m", "pip", "install", "cmake==3.31.6", "sentence-transformers==5.0.0", "transformers==4.57.6", "psutil==7.0.0"], check=True, timeout=900)
root = Path("/kaggle/working/codedoctor")
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})), mode="r:gz") as archive:
    archive.extractall(root, filter="data")
work = "/kaggle/working/c4_smoke"
for command in ["prepare", "smoke"]:
    subprocess.run([sys.executable, "scripts/c4_llama_smoke.py", command, "--work", work], cwd=root, check=True, timeout=14400)
print(Path(work, "smoke-result.json").read_text())
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
                "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)}]}
    (out / "c4-smoke.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({"id": f"{owner}/codedoctor-c4-llama-smoke", "title": "CodeDoctor C4 llama conversion smoke",
        "code_file": "c4-smoke.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
    (out / "payload-files.json").write_text(json.dumps(included, indent=2), encoding="utf-8")
    print(f"Notebook prepared at {out}; {len(included)} packaged paths; no datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "build/kernel_c4_smoke")
    args = parser.parse_args()
    build(args.owner, args.out)
