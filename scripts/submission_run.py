"""Recorded entrypoint for the final submission stages.

    python scripts/submission_run.py --stage documents --out results/documents
    python scripts/submission_run.py --stage hands-on  --out results/hands-on [--tier accuracy|interactive]
    python scripts/submission_run.py --stage queries   --out results/queries-0 --shard 0/4 --corpus-cache DIR
    python scripts/submission_run.py --stage p0        --out results/p0 --corpus-cache DIR [--query-store-in DIR ...]

Each stage runs one documented command as a child process and records, beside its
output: the exact command, the source tree's hashes, the configuration's checksum, the
commit, the machine (CPU model and instruction sets, cores, RAM, GPU), the installed
packages (``pip freeze``), the wall time, the peak memory of the process tree and the
exit status. A stage that fails or times out leaves that record and a non-zero exit; it is
never replaced by a smaller or historical result.

The ``queries`` and ``p0`` stages read the AppsRetrieval test split (through MTEB's own
loader, inside ``scripts/evaluate_mteb.py``); ``documents`` and ``hands-on`` never do.
``queries`` encodes one shard of the query vectors on the CPU, and ``p0`` runs the official
MTEB evaluation from the merged shards; both start from a verified document cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: subprocess cap per stage, in hours (a Kaggle session is 12 h including installs)
DEFAULT_HOURS = {"documents": 3.0, "hands-on": 3.0, "queries": 11.0, "p0": 11.0}
SOURCE_SUFFIXES = {".py", ".yaml", ".json"}
CPU_FLAGS = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "avx512_bf16", "amx_int8", "amx_bf16", "fma")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    """sha256 of every source, config and requirements file the run can execute."""
    hashes: dict[str, str] = {}
    for top in ("app", "config", "scripts"):
        for path in sorted((root / top).rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    for path in sorted(root.glob("requirements*.txt")):
        hashes[path.name] = sha256_file(path)
    return hashes


def tree_hash(hashes: dict[str, str]) -> str:
    """One hash for the whole source tree (path and content, order independent)."""
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(f"{name}\0{hashes[name]}\n".encode("utf-8"))
    return digest.hexdigest()


def cpu_info() -> dict:
    info: dict = {"processor": platform.processor(), "model": None, "flags": {}}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("model name") and info["model"] is None:
                info["model"] = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not info["flags"]:
                present = set(line.split(":", 1)[1].split())
                info["flags"] = {flag: flag in present for flag in CPU_FLAGS}
    return info


def gpu_info() -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def commit_id() -> dict:
    """The commit the source came from: baked into a packaged run, else asked of git."""
    if os.environ.get("CODEDOCTOR_COMMIT"):
        return {"commit": os.environ["CODEDOCTOR_COMMIT"], "dirty": os.environ.get("CODEDOCTOR_DIRTY") == "1",
                "source": "packaged"}
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=20, check=True).stdout.strip())
        return {"commit": head, "dirty": dirty, "source": "git"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "source": "unknown"}


def json_safe(value):
    """Normalize nested CLI paths without losing the provenance argument shape."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def capture_environment(stage: str, args: argparse.Namespace, out: Path) -> dict:
    import psutil

    hashes = source_hashes()
    config = ROOT / "config" / "submission.yaml"
    environment = {
        "stage": stage,
        "args": json_safe(vars(args)),
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        **commit_id(),
        "configuration": {"path": "config/submission.yaml", "sha256": sha256_file(config)},
        "source_tree_sha256": tree_hash(hashes),
        "source_sha256": hashes,
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": cpu_info(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu": gpu_info(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    if Path("/proc/cpuinfo").exists():
        shutil.copyfile("/proc/cpuinfo", out / "cpuinfo.txt")
    with (out / "requirements-measured.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)
    return environment


def write_environment(out: Path, environment: dict) -> None:
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


# --- the document cache the CPU stages start from ------------------------------

def find_one(name: str, roots: tuple[str, ...] = ("/kaggle/input",)) -> Path | None:
    found = [p for root in roots if Path(root).exists() for p in Path(root).rglob(name)]
    if len(found) > 1:
        raise RuntimeError(f"expected exactly one {name} under {roots}, found {len(found)}: pass it explicitly")
    return found[0] if found else None


def verify_document_cache(directory: Path, *, require_official: bool = False) -> dict:
    """The offline cache must be the complete, pinned one; anything else is an error."""
    from app.indexing.cache import EmbeddingCache
    from app.submission import read_config

    database, manifest_path = directory / "corpus.sqlite", directory / "manifest.json"
    if not database.exists() or not manifest_path.exists():
        raise RuntimeError(f"{directory} must hold corpus.sqlite and manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoder = read_config(ROOT / "config" / "submission.yaml")["encoder"]
    problems = []
    if sha256_file(database) != manifest.get("sha256"):
        problems.append("corpus.sqlite does not match its manifest checksum")
    if manifest.get("revision") != encoder["revision"] or manifest.get("model") != encoder["production_model"]:
        problems.append("the cache was built with a different model or revision than config/submission.yaml")
    if manifest.get("limit") is not None:
        problems.append(f"the cache is a smoke test (limit={manifest['limit']})")
    counted = EmbeddingCache(database)
    try:
        count = counted.count()
    finally:
        counted.conn.close()
    if count != manifest.get("unique_vectors"):
        problems.append(f"cache holds {count} vectors, manifest says {manifest.get('unique_vectors')}")
    if require_official:
        from app.indexing.cache_validation import APPS_DATASET, APPS_DIMENSION, APPS_DOCUMENTS, APPS_UNIQUE_DOCUMENTS
        from app.indexing.cache_validation import validate_store

        if manifest.get("dataset") != APPS_DATASET:
            problems.append("the dataset identity or revision differs from the frozen AppsRetrieval task")
        if manifest.get("documents") != APPS_DOCUMENTS or count != APPS_UNIQUE_DOCUMENTS:
            problems.append("the corpus or unique-vector count differs from the frozen AppsRetrieval task")
        fingerprint = manifest.get("texts_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            problems.append("the ordered corpus text fingerprint is missing or invalid")
        store = EmbeddingCache(database)
        try:
            validate_store(store.conn, dimension=APPS_DIMENSION)
        except ValueError as error:
            problems.append(str(error))
        finally:
            store.conn.close()
    if problems:
        raise RuntimeError("document cache rejected: " + "; ".join(problems))
    return manifest


def verify_query_inputs(stores: list[Path], *, document_manifest: dict | None = None) -> tuple[list[dict], list[tuple[int, int]]]:
    """Verify shard provenance before reading/merging any vectors or evaluating."""
    from app.indexing.cache_validation import APPS_DATASET

    if not stores:
        raise RuntimeError("strict P0 replay requires all query shard inputs")
    records, shards, reference = [], [], None
    recipe_files = ("app/retrieval/encode.py", "app/indexing/cache.py", "config/submission.yaml")
    current = source_hashes()
    for path in stores:
        if not path.is_file():
            raise RuntimeError(f"query shard store is missing: {path}")
        environment_path, report_path = path.parent / "environment.json", path.parent / "shard_report.json"
        if not environment_path.is_file() or not report_path.is_file():
            raise RuntimeError(f"query shard {path} lacks environment.json or shard_report.json")
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if environment.get("stage") != "queries" or environment.get("status") != "completed" or environment.get("exit_code") != 0:
            raise RuntimeError(f"query shard {path} was not a completed queries run")
        command = environment.get("command", [])
        if "--device" not in command or command[command.index("--device") + 1] != "cpu":
            raise RuntimeError(f"query shard {path} was not encoded on CPU")
        if environment.get("configuration", {}).get("sha256") != sha256_file(ROOT / "config/submission.yaml"):
            raise RuntimeError(f"query shard {path} used a different submission configuration")
        for name in recipe_files:
            if environment.get("source_sha256", {}).get(name) != current[name]:
                raise RuntimeError(f"query shard {path} used a different encoder recipe source: {name}")
        cache = environment.get("document_cache", {})
        if cache.get("dataset") != APPS_DATASET:
            raise RuntimeError(f"query shard {path} used a different dataset revision")
        if document_manifest is not None and any(cache.get(k) != document_manifest.get(k)
                                                 for k in ("sha256", "texts_sha256", "model", "revision", "dtype")):
            raise RuntimeError(f"query shard {path} used a different document cache from the scoring input")
        shard = report.get("shard")
        if (not isinstance(shard, list) or len(shard) != 2 or
                any(type(v) is not int for v in shard) or not 0 <= shard[0] < shard[1]):
            raise RuntimeError(f"query shard {path} has an invalid declared shard")
        if environment.get("args", {}).get("shard") != f"{shard[0]}/{shard[1]}":
            raise RuntimeError(f"query shard {path} report and command disagree")
        if type(report.get("store_vectors")) is not int or report["store_vectors"] <= 0:
            raise RuntimeError(f"query shard {path} has an invalid reported vector count")
        if not environment.get("source_tree_sha256") or any(not environment.get("packages", {}).get(k) for k in
                ("torch", "transformers", "sentence-transformers", "mteb", "datasets", "numpy")):
            raise RuntimeError(f"query shard {path} has incomplete source or dependency provenance")
        identity = (environment.get("source_tree_sha256"), environment.get("configuration"),
                    cache.get("sha256"), cache.get("texts_sha256"),
                    {k: environment.get("packages", {}).get(k) for k in
                     ("torch", "transformers", "sentence-transformers", "mteb", "datasets", "numpy")})
        if reference is not None and identity != reference:
            raise RuntimeError("query shard source, packages, configuration or corpus provenance disagree")
        reference = identity
        shards.append(tuple(shard))
        records.append({"path": str(path), "environment_sha256": sha256_file(environment_path),
                        "report_sha256": sha256_file(report_path), "source_commit": environment.get("commit"),
                        "shard": shard, "reported_rows": report.get("store_vectors")})
    count = shards[0][1]
    if any(n != count for _, n in shards) or sorted(i for i, _ in shards) != list(range(count)):
        raise RuntimeError("query inputs must contain every declared shard exactly once")
    return records, shards


# --- stage commands ------------------------------------------------------------

def build_command(stage: str, args: argparse.Namespace, out: Path, cache: Path | None) -> list[str]:
    py = sys.executable
    if stage == "documents":
        return [py, "scripts/prepare_document_cache.py", "--device", "cuda", "--out", str(out / "documents")]
    if stage == "hands-on":
        return [py, "scripts/validate_hands_on.py", "--tier", args.tier, "--out", str(out / "hands-on")]
    common = [py, "scripts/evaluate_mteb.py", "--device", "cpu", "--config", "config/submission.yaml",
              "--corpus-cache", str(cache), "--require-complete-corpus-cache",
              "--corpus-manifest", str(cache.parent / "manifest.json"),
              "--query-store", str(out / "queries.sqlite")]
    if stage == "queries":
        return [*common, "--query-shard", args.shard, "--output", str(out / "shard_report.json")]
    return [*common, "--require-complete-query-store", "--query-verify", str(args.query_verify),
            "--output", str(out / "appsretrieval_results.json"),
            "--predictions", str(out / "predictions")]


def run_logged(command: list[str], out: Path, timeout_seconds: float) -> dict:
    """Run ``command`` teeing its output to run.log; enforce the timeout; track peak memory."""
    import psutil

    started, peak, timed_out = time.perf_counter(), 0, False
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with (out / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)

        def copy_output() -> None:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        while proc.poll() is None:
            try:
                parent = psutil.Process(proc.pid)
                peak = max(peak, sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.perf_counter() - started > timeout_seconds:
                timed_out = True
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
                proc.kill()
                break
            time.sleep(1.0)
        code = proc.wait()
        reader.join(timeout=10)
    return {"exit_code": code, "seconds": time.perf_counter() - started,
            "peak_process_tree_rss_bytes": peak, "timed_out": timed_out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=sorted(DEFAULT_HOURS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus-cache", type=Path, help="directory with corpus.sqlite and manifest.json")
    parser.add_argument("--shard", help="queries stage: I/N (0-based index / shard count)")
    parser.add_argument("--query-store-in", type=Path, action="append", default=[],
                        help="p0 stage: a directory holding queries.sqlite from a shard (repeatable)")
    parser.add_argument("--query-verify", type=int, default=30,
                        help="p0 stage: stored queries re-encoded live as an independent check")
    parser.add_argument("--tier", choices=["accuracy", "interactive"], default="accuracy",
                        help="hands-on stage: accuracy = 1.7B (default); interactive = 0.6B, reported separately")
    parser.add_argument("--timeout-hours", type=float, default=None)
    args = parser.parse_args(argv)
    if args.stage == "queries" and not args.shard:
        parser.error("--stage queries needs --shard I/N")
    if args.stage == "p0" and args.query_verify != 30:
        parser.error("the frozen P0 protocol requires exactly 30 live query verifications")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    environment = capture_environment(args.stage, args, out)
    write_environment(out, environment)

    cache = None
    if args.stage in ("queries", "p0"):
        source = args.corpus_cache or find_one("corpus.sqlite")
        source = source.parent if source is not None and source.name == "corpus.sqlite" else source
        if source is None:
            raise SystemExit("provide the verified document cache with --corpus-cache")
        # sqlite needs a writable directory, and a second corpus.sqlite in this stage's output would make
        # the next stage's input discovery ambiguous: work on a copy under the temp directory
        cache = Path(tempfile.mkdtemp(prefix="codedoctor-documents-")) / "cache"
        shutil.copytree(source, cache)
        environment["document_cache"] = verify_document_cache(cache, require_official=True)
        shutil.copyfile(source / "manifest.json", out / "document-cache-manifest.json")
        cache = cache / "corpus.sqlite"
    if args.stage == "p0":
        from scripts.merge_query_stores import merge

        stores = [d / "queries.sqlite" for d in (args.query_store_in or
                                                  [p.parent for p in Path("/kaggle/input").rglob("queries.sqlite")])]
        records, shards = verify_query_inputs(stores, document_manifest=environment["document_cache"])
        environment["query_shard_provenance"] = records
        if stores:
            # Kaggle input mounts are read-only; SQLite's WAL/schema setup writes.
            scratch = Path(tempfile.mkdtemp(prefix="codedoctor-queries-"))
            writable = []
            for index, source in enumerate(stores):
                target = scratch / f"shard-{index}.sqlite"
                shutil.copyfile(source, target)
                for suffix in ("-wal", "-shm"):
                    sidecar = Path(str(source) + suffix)
                    if sidecar.exists():
                        shutil.copyfile(sidecar, Path(str(target) + suffix))
                writable.append(target)
            from app.indexing.cache_validation import APPS_DIMENSION

            environment["query_store_merge"] = merge(writable, out / "queries.sqlite", dimension=APPS_DIMENSION,
                                                       shards=shards, require_disjoint=True)
            for actual, expected in zip(environment["query_store_merge"]["inputs"], records):
                if actual["rows"] != expected["reported_rows"]:
                    raise RuntimeError("query store row count differs from its shard report")
            for record, source in zip(environment["query_store_merge"]["inputs"], stores):
                record["original_path"] = str(source)
            if environment["query_store_merge"]["duplicate_keys_disagreeing"]:
                write_environment(out, environment)
                raise RuntimeError("query shards disagree; refusing final evaluation")
        write_environment(out, environment)

    command = build_command(args.stage, args, out, cache)
    environment.update(command=command, status="running")
    write_environment(out, environment)
    result = run_logged(command, out, (args.timeout_hours or DEFAULT_HOURS[args.stage]) * 3600)
    environment.update(result, finished_utc=datetime.now(timezone.utc).isoformat(),
                       status="timeout" if result["timed_out"] else "completed" if result["exit_code"] == 0 else "failed")
    audit = ROOT / "data" / "test_split_audit_log.json"
    if args.stage in ("queries", "p0") and audit.exists():
        shutil.copyfile(audit, out / "test_split_audit_log.json")
    write_environment(out, environment)
    if result["exit_code"] != 0:
        return result["exit_code"] or 1
    if args.stage == "p0":
        validation = subprocess.run([sys.executable, "scripts/validate_submission.py",
                                     str(out / "appsretrieval_results.json"), "--official-apps"], cwd=ROOT)
        environment["official_result_validation_exit_code"] = validation.returncode
        if validation.returncode:
            environment["status"] = "failed_validation"
        write_environment(out, environment)
        if validation.returncode:
            return validation.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
