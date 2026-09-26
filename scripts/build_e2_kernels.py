"""Build E2's private Kaggle dataset and kernels (experiments/C3_cycle/PREREG.md, E2).

    python scripts/build_e2_kernels.py data            # build/e2_data (+ dataset-metadata.json)
    kaggle datasets create -p build/e2_data --dir-mode zip   # private by default
    python scripts/build_e2_kernels.py kernel --stage smoke|a|b [--arm T1]
    kaggle kernels push -p build/kernel_e2<stage>

The dataset holds query/corpus TEXTS and stored vectors only.  No qrels, no
APPS test-split query, no EXT-lockbox statement is written into it; the
manifest records every file with its sha256 so the upload can be audited.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.packaging import build_payload  # noqa: E402

APPS, EXT = REPO_ROOT / "artifacts" / "apps", REPO_ROOT / "artifacts" / "ext_cc"
OUT = REPO_ROOT / "data" / "outcomes"
DATASET = "munalsingh/codedoctor-c3-e2-inputs"
KERNELS = {"smoke": "munalsingh/codedoctor-e2-smoke", "a": "munalsingh/codedoctor-e2-ov-a",
           "b": "munalsingh/codedoctor-e2-ov-b"}
PINS = ("'transformers==4.57.6' 'sentence-transformers==5.0.0' 'openvino==2026.4.0' "
        "'openvino-tokenizers==2026.4.0.0' 'nncf==3.4.0' 'optimum==2.3.0' 'optimum-intel==2.2.0' psutil")
SEED = 20260926


def texts(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        return {r["id"]: r["text"] for r in map(json.loads, handle)}


def store_entries(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        handle.readline()
        return {e["qid"]: e for e in map(json.loads, handle)}


def build_data(out: Path) -> None:
    from scripts.bench_query_cpu import _statements

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    qids_train = json.loads((APPS / "ids_queries_train.json").read_text(encoding="utf-8"))
    queries = texts(APPS / "texts_queries_train.jsonl")

    # the 100 seeded TUNE statements R2 timed (same seed, same rows)
    rows, tune_texts = _statements(APPS, 100, 20260924)
    tune_qids = [qids_train[r] for r in rows]
    (out / "tune100.json").write_text(json.dumps({"rows": rows, "qids": tune_qids, "texts": tune_texts}), encoding="utf-8")
    for tag in ("f2llm17b", "f2llm06b", "f2llm160m"):
        np.save(out / f"tune100_{tag}.npy", np.load(APPS / f"emb_{tag}_qtrain.npy")[rows])
    tune_store = store_entries(OUT / "f2llm17b_all_tune_deep50.jsonl.gz")
    assert all(q in tune_store for q in tune_qids), "the R2 statements must be TUNE statements"
    (out / "tune100_pools.json").write_text(json.dumps(
        {q: {"pool": tune_store[q]["pool"], "scores": tune_store[q]["scores"]} for q in tune_qids}), encoding="utf-8")

    shutil.copy(APPS / "texts_corpus.jsonl", out / "texts_corpus.jsonl")
    for tag in ("f2llm17b", "f2llm06b"):
        shutil.copy(APPS / f"emb_{tag}_corpus.npy", out / f"emb_{tag}_corpus.npy")
    corpus = texts(APPS / "texts_corpus.jsonl")
    doc_ids = random.Random(SEED).sample(sorted(corpus, key=lambda d: int(d[1:])), 100)
    (out / "docs100.json").write_text(json.dumps([corpus[d] for d in doc_ids]), encoding="utf-8")

    dev_qids = sorted(store_entries(OUT / "f2llm17b_all_dev.jsonl.gz"), key=lambda q: int(q[1:]))
    assert len(dev_qids) == 2500
    (out / "dev_texts.json").write_text(json.dumps({"qids": dev_qids, "texts": [queries[q] for q in dev_qids]}),
                                        encoding="utf-8")
    timing = random.Random(SEED).sample(dev_qids, 100)
    (out / "dev_timing100.json").write_text(json.dumps({"qids": timing, "texts": [queries[q] for q in timing]}),
                                            encoding="utf-8")
    splits = json.loads((EXT / "splits.json").read_text(encoding="utf-8"))
    ext_queries = texts(EXT / "texts_queries.jsonl")
    ext_ids = [q for q in ext_queries if q in set(splits["ext_dev"])]
    assert len(ext_ids) == 108 and not set(ext_ids) & set(splits["ext_lockbox"])
    (out / "extdev_texts.json").write_text(json.dumps({"qids": ext_ids, "texts": [ext_queries[q] for q in ext_ids]}),
                                           encoding="utf-8")

    manifest = {"purpose": "CodeDoctor cycle-3 E2 inputs (experiments/C3_cycle/PREREG.md)",
                "contains": "APPS train query texts (DEV half + 100 TUNE), APPS corpus texts, stored F2LLM "
                            "vectors, CodeContests EXT-dev statement texts, recorded top-50 pools of 100 TUNE statements",
                "excludes": "qrels of any split, APPS test-split queries, EXT-lockbox statements",
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(out.iterdir())}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "codedoctor-c3-e2-inputs", "id": DATASET, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    print(f"dataset -> {out} ({sum(p.stat().st_size for p in out.iterdir()) / 1e6:.1f} MB)")


KERNEL = r'''"""CodeDoctor cycle-3 E2 (__STAGE__): generated by scripts/build_e2_kernels.py; do not edit."""
import base64, io, os, subprocess, sys, tarfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
__ENV__
subprocess.run(f"{sys.executable} -m pip install -q __PINS__", shell=True, check=True)
WORK = "/kaggle/working/codedoctor"
os.makedirs(WORK, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__")), mode="r:gz") as tar:
    tar.extractall(WORK)
def find(name):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return root
    return None
DATA = find("MANIFEST.json")
STAGE_A = find("e2_A.json")
print("mounted inputs:", DATA, STAGE_A, flush=True)
subprocess.run([sys.executable, "scripts/e2_ov_kernel.py", __ARGS__], cwd=WORK, check=False)
for name in ("e2_A.json", "e2_B.json"):
    path = os.path.join("/kaggle/working/e2", name)
    if os.path.exists(path):
        print(open(path).read()[:20000])
'''


def build_kernel(stage: str, arm: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    paths = ("app/__init__.py", "app/retrieval/__init__.py", "app/retrieval/encode.py", "app/pipeline",
             "config/frozen.yaml", "config/retrieval.yaml", "config/candidate_v2.yaml",
             "scripts/__init__.py", "scripts/e2_ov_kernel.py")
    payload, included = build_payload(paths)
    env = ""
    if stage == "smoke":
        env = 'os.environ["E2_SMOKE"] = "1"\nos.environ["E2_LIMIT"] = "6"'
    if stage in ("smoke", "a"):
        args = '"stage-a", "--data", DATA, "--work", "/kaggle/working/e2"'
        sources = []
    else:
        args = (f'"stage-b", "--data", DATA, "--work", "/kaggle/working/e2", "--ir", STAGE_A + "/ir_{arm}", '
                f'"--arm", "{arm}", "--stage-a-dir", STAGE_A')
        sources = [KERNELS["a"]]
    script = (KERNEL.replace("__PAYLOAD__", payload).replace("__PINS__", PINS).replace("__ENV__", env)
              .replace("__ARGS__", args).replace("__STAGE__", stage))
    name = KERNELS[stage].split("/")[1]
    (out / f"{name}.py").write_text(script, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNELS[stage], "title": name, "code_file": f"{name}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [DATASET], "kernel_sources": sources,
        "competition_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(f"kernel {stage} -> {out} ({len(script) / 1e6:.2f} MB script; {len(included)} paths)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "e2_data")
    k = sub.add_parser("kernel")
    k.add_argument("--stage", choices=sorted(KERNELS), required=True)
    k.add_argument("--arm", default="T1")
    args = parser.parse_args(argv)
    if args.cmd == "data":
        build_data(args.out)
    else:
        build_kernel(args.stage, args.arm, REPO_ROOT / "build" / f"kernel_e2{args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
