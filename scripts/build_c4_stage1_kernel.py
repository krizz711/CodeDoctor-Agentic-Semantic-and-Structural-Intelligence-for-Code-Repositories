"""Build the private Kaggle C4 100-query screen from pinned source artifacts.

Only execute after conversion smoke passes. It mounts the C4 train-only dataset
and the smoke kernel's checkpoint/GGUF output. No test files are mounted.
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
        "scripts/__init__.py", "scripts/c4_llama_smoke.py", "scripts/c4_stage1.py"))
    out.mkdir(parents=True, exist_ok=True)
    code = f'''# C4-E1 preregistered stage 1. No APPS test or EXT lockbox inputs.
import base64, io, os, shutil, socket, subprocess, sys, tarfile
from pathlib import Path
from urllib.request import urlopen
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for host in ["pypi.org", "huggingface.co"]:
    print("DNS preflight:", host, socket.gethostbyname(host), flush=True)
with urlopen("https://pypi.org/simple/psutil/", timeout=15) as response:
    print("PyPI HTTP preflight:", response.status, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--index-url", "https://pypi.org/simple", "--retries", "0", "--timeout", "30", "sentence-transformers==5.0.0", "transformers==4.57.6", "psutil==7.0.0"], check=True, timeout=900)
root=Path("/kaggle/working/codedoctor"); root.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})), mode="r:gz") as archive:
    archive.extractall(root, filter="data")
data_candidates=list(Path("/kaggle/input").rglob("tune100_seed20260929.json"))
model_candidates=list(Path("/kaggle/input").rglob("F2LLM-v2-1.7B-Q8_0.gguf"))
assert len(data_candidates)==1 and len(model_candidates)==1, (data_candidates,model_candidates)
data=data_candidates[0].parent
source=model_candidates[0].parent
manifest=__import__("json").loads((source/"smoke-result.json").read_text())
assert manifest["status"]=="synthetic_smoke_passed", manifest["status"]
models=Path("/kaggle/working/c4_models"); models.mkdir()
for name in ["checkpoint", "F2LLM-v2-1.7B-F32.gguf", "F2LLM-v2-1.7B-Q8_0.gguf", "manifest.json"]:
    (models/name).symlink_to(source/name, target_is_directory=name=="checkpoint")
bindir=models/"llama.cpp/build/bin"; bindir.mkdir(parents=True)
for item in (source/"llama.cpp/build/bin").iterdir():
    if item.is_file():
        shutil.copy2(item, bindir/item.name)
        (bindir/item.name).chmod(0o755)
work="/kaggle/working/c4_stage1"
subprocess.run([sys.executable,"scripts/c4_stage1.py","run","--data",str(data),"--models",str(models),"--out",work], cwd=root, check=True, timeout=18000)
print(Path(work,"stage1-result.json").read_text())
'''
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"}},
        "cells":[{"id":"c4-stage1","cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}]}
    (out/"c4-stage1.ipynb").write_text(json.dumps(notebook,indent=1),encoding="utf-8")
    (out/"kernel-metadata.json").write_text(json.dumps({"id":f"{owner}/codedoctor-c4-stage1", "title":"codedoctor-c4-stage1",
        "code_file":"c4-stage1.ipynb","language":"python","kernel_type":"notebook","is_private":True,
        "enable_gpu":False,"enable_tpu":False,"enable_internet":True,
        "dataset_sources":[f"{owner}/codedoctor-c4-train-inputs"],
        "kernel_sources":[f"{owner}/codedoctor-c4-llama-conversion-smoke"],
        "competition_sources":[],"model_sources":[]},indent=2),encoding="utf-8")
    print(f"Stage-1 notebook prepared: {out}; {len(included)} source paths")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner",required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"build/kernel_c4_stage1")
    args=parser.parse_args()
    build(args.owner,args.out)
