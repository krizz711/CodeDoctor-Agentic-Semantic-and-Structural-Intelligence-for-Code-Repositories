"""Functional CPU check on new Python snapshots, without benchmark data.

    python scripts/validate_cold_python.py --out build/validation/cold-python

The fixture and cache are created in a new output directory. This exercises
the real workbench document encoder and versioned retrieval engine; it does
not estimate unseen accuracy or produce a screening score. Model weights may
already be downloaded, but no document or query vectors are reused initially.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.indexing.cache import EmbeddingCache  # noqa: E402
from app.indexing.versioned import VersionedIndex  # noqa: E402
from app.retrieval.encode import CodeEncoder  # noqa: E402
from app.workbench import Workbench  # noqa: E402

QUERY = "Where does this code calculate the combined total of two input amounts?"
SOURCE_FILES = (
    "app/workbench.py", "app/indexing/cache.py", "app/indexing/versioned.py",
    "app/parser/chunker.py", "app/parser/edges.py", "app/retrieval/encode.py",
    "app/retrieval/hybrid.py", "app/submission.py", "config/submission.yaml",
    "scripts/validate_cold_python.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    files = set(SOURCE_FILES)
    files.update(path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "app").rglob("*.py"))
    return {name: digest(REPO_ROOT / name) for name in sorted(files)}


class ObservedEncoder:
    """Record actual forward calls; delegate all inference to CodeEncoder."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def encode(self, texts, mode="document", **kwargs):
        inputs = [texts] if isinstance(texts, str) else list(texts)
        started = time.perf_counter()
        vectors = self.encoder.encode(texts, mode=mode, **kwargs)
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim not in (1, 2) or not np.isfinite(values).all():
            raise RuntimeError("encoder returned invalid vectors")
        self.calls.append({"mode": mode, "count": len(inputs),
                           "input_sha256": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in inputs],
                           "seconds": time.perf_counter() - started,
                           "dimension": int(values.shape[-1])})
        return vectors


def write_fixture(folder: Path, *, edited: bool = False) -> None:
    operator = "+" if edited else "-"
    (folder / "amounts.py").write_text(
        'def combine_amounts(left, right):\n    """Combine two integer amounts."""\n'
        f'    return left {operator} right\n\n'
        'def order_records(records):\n    """Order records in ascending order."""\n'
        '    return sorted(records)\n', encoding="utf-8")
    (folder / "formatting.py").write_text(
        'def format_title(text):\n    """Clean up the title of a report."""\n'
        '    return text.strip().title()\n', encoding="utf-8")


def functional_check(out: Path, encoder, report: dict | None = None) -> dict:
    fixture = out / "fixture"
    fixture.mkdir()
    write_fixture(fixture)
    cache = EmbeddingCache(out / "cold.sqlite")
    observed = ObservedEncoder(encoder)
    bench = Workbench(cache=cache)
    bench._encoder = observed
    key = bench.open_folder(str(fixture))["key"]
    target = bench._target(key)
    snapshots = VersionedIndex(encoder=observed)
    report = report if report is not None else {}
    report.update({"kind": "functional synthetic Python check; not an unseen-accuracy estimate",
                    "query": QUERY, "initial_cache_entries": cache.count(),
                    "snapshot_kind": "constructed code versions, not Git commits",
                    "builds": [], "queries": [], "encoder_calls": observed.calls})
    if cache.count() != 0:
        raise RuntimeError("cold-path check requires an empty document cache")
    try:
        for label in ("v1", "v2"):
            if label == "v2":
                write_fixture(fixture, edited=True)
            saved_fixture = out / "snapshots" / label
            saved_fixture.mkdir(parents=True)
            for path in fixture.glob("*.py"):
                (saved_fixture / path.name).write_bytes(path.read_bytes())
            started = time.perf_counter()
            bench._build(target, versions=1, dense=True)
            state = bench.state()
            if state["phase"] != "ready":
                raise RuntimeError(f"{label} build did not finish: {state}")
            built = bench._require(key)
            documents = built.index.documents_at("working")
            embeddings = built.index._embeddings["working"]
            snapshots.add_snapshot(label, documents, embeddings=embeddings.copy())
            report["builds"].append({"version": label, "seconds": time.perf_counter() - started,
                                      "includes_encoder_load_and_warmup": label == "v1",
                                      "chunks": len(documents), "encoded": state["encoded"],
                                      "reused": state["reused"],
                                      "file_sha256": {p.name: digest(p) for p in sorted(fixture.glob("*.py"))}})
        if not report["builds"][0]["encoded"] or not report["builds"][1]["encoded"]:
            raise RuntimeError("both cold indexing and the edit must encode real documents")
        if not report["builds"][1]["reused"]:
            raise RuntimeError("the unchanged file was not reused after editing")

        for label in ("v1", "v2"):
            started = time.perf_counter()
            hits = snapshots.retrieve(QUERY, label, limit=10)
            eligible = {d.document_id: d.content for d in snapshots.documents_at(label)}
            if not hits or not any("dense" in hit.sources for hit in hits):
                raise RuntimeError(f"{label} query did not return dense retrieval results")
            if not all(eligible.get(h.document.document_id) == h.document.content for h in hits):
                raise RuntimeError(f"{label} query leaked another version's code")
            report["queries"].append({"version": label, "seconds": time.perf_counter() - started,
                                      "timings": snapshots._retrievers[label].last_timings.copy(),
                                      "version_isolation": True,
                                      "hits": [{"file": h.document.file_path, "name": h.document.name,
                                                "start_line": h.document.start_line, "end_line": h.document.end_line,
                                                "content": h.document.content, "score": float(h.score),
                                                "sources": list(h.sources)} for h in hits]})
        try:
            snapshots.retrieve(QUERY, "missing-version")
        except KeyError:
            report["unknown_version_rejected"] = True
        else:
            raise RuntimeError("an unknown version silently fell back")
        started = time.perf_counter()
        across = snapshots.retrieve_across_versions(QUERY, limit=10)
        if not across or not any(set(hit.versions) == {"v1", "v2"} for hit in across):
            raise RuntimeError("cross-version retrieval did not retain both version identities")
        changed = [hit for hit in across if hit.document.name == "combine_amounts"]
        if not changed or not any(hit.variants == 2 for hit in changed):
            raise RuntimeError("cross-version retrieval lost the edited function variant")
        report["cross_version"] = {"seconds": time.perf_counter() - started,
                                   "hits": [{"file": h.document.file_path, "name": h.document.name,
                                             "versions": h.versions, "variants": h.variants,
                                             "content": h.document.content,
                                             "score": float(h.score), "sources": list(h.sources)} for h in across]}
        report["final_cache_entries"] = cache.count()
        report["dimension"] = observed.calls[0]["dimension"]
        return report
    finally:
        cache.conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ("torch", "sentence-transformers", "transformers", "numpy", "tree-sitter", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    hashes = source_hashes()
    report = {"status": "started", "platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "logical_cpus": os.cpu_count(),
              "physical_cpus": psutil.cpu_count(logical=False), "memory_bytes": psutil.virtual_memory().total,
              "packages": packages, "source_hashes": hashes,
              "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
              "model_weight_cache": "May be populated; document cache starts empty and query vectors are never cached."}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=30, check=False)
    report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else os.environ.get("CODEDOCTOR_COMMIT")
    state = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                           text=True, timeout=30, check=False)
    report["git_worktree_dirty"] = (bool(state.stdout.strip()) if state.returncode == 0 else
                                    {"0": False, "1": True}.get(os.environ.get("CODEDOCTOR_DIRTY")))
    started = time.perf_counter()
    try:
        encoder = CodeEncoder(tier="accuracy", device="cpu", torch_dtype="float32")
        report["encoder"] = {"model": encoder.model_name, "revision": encoder.revision,
                              "tier": encoder.tier, "device": encoder.device, "dtype": encoder.torch_dtype,
                              "max_seq_length": encoder.max_seq_length, "max_batch": encoder.max_batch}
        functional_check(args.out, encoder, report)
        if source_hashes() != hashes:
            raise RuntimeError("runtime source changed during the check")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    report["total_seconds"] = time.perf_counter() - started
    if find_spec("resource") is not None:
        import resource

        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.out / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
