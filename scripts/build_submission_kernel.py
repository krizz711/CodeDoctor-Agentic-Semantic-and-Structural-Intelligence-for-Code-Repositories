"""Package one submission stage as an isolated Kaggle script kernel.

    python scripts/build_submission_kernel.py --owner OWNER --stage documents  --out build/kernels/documents
    python scripts/build_submission_kernel.py --owner OWNER --stage hands-on   --out build/kernels/hands-on
    python scripts/build_submission_kernel.py --owner OWNER --stage queries --shard 0/4 --out build/kernels/q0
    python scripts/build_submission_kernel.py --owner OWNER --stage p0 --shards 4 --out build/kernels/p0
    kaggle kernels push -p build/kernels/documents

The kernel builds a fresh virtual environment, installs the pinned requirements (the lock
file when one is committed, else ``requirements-submission.txt``) and runs
``scripts/submission_run.py`` for the stage: the same commands a judge would run, recorded.

The source is packaged from the working tree, so the tree must be clean: the commit id is
baked into the kernel and recorded with the run, and the measured runtime source is then a
committed one. ``--allow-dirty`` exists for smoke tests and marks the run as dirty.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TORCH = "torch==2.8.0"
VALIDATION_TESTS = (
    "tests/test_sandbox_runner.py", "tests/test_judge_like_io.py", "tests/test_execution_rerank.py",
    "tests/test_statement_and_calls.py", "tests/test_parallel_execution.py", "tests/test_replay_outcomes.py",
    "tests/test_blank_line_fallback.py", "tests/test_corpus_cache.py", "tests/test_query_store_and_cache_guard.py",
    "tests/test_workbench_apps_provenance.py", "tests/test_cold_python_validation.py", "tests/test_submission_run.py",
    "tests/test_merge_query_stores.py", "tests/test_evaluate_mteb_cli.py", "tests/test_export_submission_csv.py",
    "tests/test_submission_validation.py", "tests/test_published_recipe.py",
)
PAYLOAD = (
    "app", "config", "requirements.txt", "requirements-submission.txt", "requirements-lock.txt",
    "scripts/__init__.py", "scripts/evaluate_mteb.py", "scripts/export_submission_csv.py",
    "scripts/validate_submission.py", "scripts/validate_hands_on.py", "scripts/prepare_document_cache.py",
    "scripts/submission_run.py", "scripts/merge_query_stores.py",
    "scripts/validate_cold_python.py",
) + VALIDATION_TESTS + ("tests/conftest.py", "pytest.ini")
TIMEOUT_HOURS = {"documents": 3, "hands-on": 3, "queries": 11, "p0": 11}


def git_state() -> tuple[str, bool]:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip())
    return head, dirty


def git_payload(commit: str) -> tuple[str, list[str]]:
    """The packaged source as committed: raw blobs of ``commit``, never the working tree.

    A working tree may carry platform line endings (this project is developed on Windows with
    autocrlf), which would make every recorded hash differ from the committed blobs, and
    ``git archive`` applies the same conversion. Reading each blob with ``git cat-file`` gives the
    exact committed bytes, so the measured runtime source is byte-identical to what is merged.
    The tarball uses fixed timestamps and sorted names, so the payload itself is reproducible.
    """
    listing = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", commit],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    wanted = sorted(f for f in listing if "__pycache__" not in f
                    and any(f == p or f.startswith(p + "/") for p in PAYLOAD))
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name in wanted:
                data = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", f"{commit}:{name}"],
                                      capture_output=True, check=True).stdout
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buffer.getvalue()).decode("ascii"), wanted


def slug(stage: str, shard: str | None, tier: str) -> str:
    if stage == "queries":
        index, count = shard.split("/")
        return f"codedoctor-submission-queries-{index}-of-{count}"
    if stage == "hands-on" and tier != "accuracy":
        return "codedoctor-submission-hands-on-06b"
    return f"codedoctor-submission-{stage}"


def stage_arguments(args: argparse.Namespace) -> list[str]:
    run = ["--stage", args.stage, "--out", "/kaggle/working/results"]
    if args.stage == "hands-on":
        run += ["--tier", args.tier]
    if args.stage == "queries":
        run += ["--shard", args.shard]
    return run


def kernel_sources(args: argparse.Namespace) -> list[str]:
    if args.stage == "queries":
        return [f"{args.owner}/codedoctor-submission-documents"]
    if args.stage == "p0":
        return [f"{args.owner}/codedoctor-submission-documents",
                *[f"{args.owner}/codedoctor-submission-queries-{i}-of-{args.shards}" for i in range(args.shards)]]
    return []


def kernel_code(payload: str, args: argparse.Namespace, commit: str, dirty: bool, requirements: str) -> str:
    gpu = args.stage == "documents"
    cold_check = f'''with open("/kaggle/working/validation.log", "w") as log:
    checked = subprocess.run([python, "-m", "pytest", "-q", "-rs", "-m", "not model and not data",
                              *{list(VALIDATION_TESTS)!r}], cwd=root, stdout=log,
                             stderr=subprocess.STDOUT, timeout=900)
print(Path("/kaggle/working/validation.log").read_text(), flush=True)
checked.check_returncode()
subprocess.run([python, "scripts/validate_cold_python.py", "--out",
                "/kaggle/working/cold-python"], cwd=root, check=True, timeout=1800)
''' if getattr(args, "cold_python_first", False) else ""
    return f'''"""CodeDoctor submission stage {args.stage!r} (generated by scripts/build_submission_kernel.py; do not edit)."""
import base64, io, json, os, platform, subprocess, sys, tarfile
from pathlib import Path
from urllib.request import urlopen

os.environ["HF_HOME"] = "/tmp/codedoctor-model-cache"
os.environ["CODEDOCTOR_COMMIT"] = {commit!r}
os.environ["CODEDOCTOR_DIRTY"] = {"1" if dirty else "0"!r}
os.environ["CUDA_VISIBLE_DEVICES"] = {"0" if gpu else ""!r}

root = Path("/kaggle/working/source")
root.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})), mode="r:gz") as archive:
    archive.extractall(root, filter="data")
with urlopen("https://pypi.org/simple/", timeout=30) as response:
    print("Internet preflight", response.status, flush=True)

venv = Path("/tmp/codedoctor-env")
try:
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
    # some images ship Python without ensurepip; virtualenv does not need it
    import shutil
    shutil.rmtree(venv, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "virtualenv"], check=True, timeout=600)
    subprocess.run([sys.executable, "-m", "virtualenv", str(venv)], check=True, timeout=300)
python = str(venv / "bin" / "python")
subprocess.run([python, "-m", "pip", "install", "--index-url",
                "https://download.pytorch.org/whl/{"cu128" if gpu else "cpu"}", "{TORCH}"],
               check=True, timeout=1800)
subprocess.run([python, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                "--extra-index-url", "https://download.pytorch.org/whl/{"cu128" if gpu else "cpu"}",
               "-r", str(root / {requirements!r})], check=True, timeout=1800)
with open("/kaggle/working/requirements-setup.txt", "w") as freeze:
    subprocess.run([python, "-m", "pip", "freeze"], stdout=freeze, check=True, timeout=120)
Path("/kaggle/working/setup-environment.json").write_text(json.dumps({{
    "commit": {commit!r}, "dirty": {dirty!r}, "platform": platform.platform(),
    "bootstrap_python": sys.version, "inference_python": python,
    "logical_cpus": os.cpu_count(), "gpu_enabled": {gpu!r}
}}, indent=2))
{cold_check}\
subprocess.run([python, "scripts/submission_run.py", *{stage_arguments(args)!r}], cwd=root, check=True,
               timeout={TIMEOUT_HOURS[args.stage] * 3600 + 900})
'''


def build(args: argparse.Namespace) -> Path:
    name = getattr(args, "kernel_name", None) or slug(args.stage, args.shard, args.tier)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise SystemExit("kernel name must contain lowercase words/numbers separated by hyphens")
    if getattr(args, "cold_python_first", False) and args.stage != "p0":
        raise SystemExit("--cold-python-first is only supported before the p0 stage")
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("working tree has uncommitted changes; commit first (or --allow-dirty for a smoke test)")
    requirements = "requirements-lock.txt" if (ROOT / "requirements-lock.txt").exists() else "requirements-submission.txt"
    payload, included = git_payload(commit)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(kernel_code(payload, args, commit, dirty, requirements), encoding="utf-8")
    meta = {
        "id": f"{args.owner}/{name}",
        "title": name.replace("-", " "),
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": args.stage == "documents", "enable_internet": True,
        "dataset_sources": [], "kernel_sources": kernel_sources(args), "competition_sources": [], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "payload-files.json").write_text(json.dumps({"commit": commit, "dirty": dirty, "requirements": requirements,
                                                        "files": included}, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Kaggle username that will own the kernel")
    parser.add_argument("--stage", choices=sorted(TIMEOUT_HOURS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", help="queries stage: I/N")
    parser.add_argument("--shards", type=int, default=4, help="p0 stage: how many query-shard kernels feed it")
    parser.add_argument("--tier", choices=["accuracy", "interactive"], default="accuracy")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--kernel-name", help="Distinct notebook slug; upstream artifact names remain unchanged")
    parser.add_argument("--cold-python-first", action="store_true",
                        help="Run focused Linux tests and the real 1.7B CPU cold Python check before p0 scoring")
    args = parser.parse_args(argv)
    if args.stage == "queries" and not args.shard:
        parser.error("--stage queries needs --shard I/N")
    print(build(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
