"""Final P1 and Bonus measurements through the shipping workbench and version index.

    python scripts/validate_hands_on.py --tier accuracy --out results/hands-on

P1 (rebuild for any change). A fixed 40-function Python fixture is indexed with
``Workbench.start_build`` and then updated through the workbench's own ``rebuild`` path:
no change, one edited function, ten edited functions, an added file, a deleted file and a
renamed (moved) file. Three repetitions, each in a fresh cache. For every operation it
records wall time, process memory, chunks encoded versus reused, and whether the encoded
count equals the number of chunks whose encoder input really changed. At the end the
incrementally updated index is compared with a fresh rebuild of the same files (vectors and
rankings), and the edited function must be retrievable.

Bonus (retrieve across versions). Twelve lineages of one function across three versions,
where the correct variant sits in the oldest, middle or newest version in turn, some wrong
variants repeat across versions and some lineages never change. Each query carries worked
examples. The ranking is produced by ``Workbench.across``, the method behind the
``/api/workbench/across`` endpoint. Reported: correct-version-first, lineage-first, wrong
versions ranked above the correct one, distinct lineages in the top ten, and query latency.
The wrong variants are chosen to fail the worked examples, so this is a functional
diagnostic and not an estimate of accuracy on unseen repositories.

Nothing here reads the APPS test split.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.indexing.cache import EmbeddingCache, recipe_cache_key  # noqa: E402
from app.indexing.versioned import VersionedIndex  # noqa: E402
from app.retrieval.encode import CodeEncoder  # noqa: E402
from app.retrieval.hybrid import Document  # noqa: E402
from app.submission import execution_config  # noqa: E402
from app.workbench import (  # noqa: E402
    DENSE_MAX_INPUT_CHARS,
    Target,
    TargetIndex,
    VersionInfo,
    Workbench,
    _documents_from_source,
    _encoder_input,
)

OPERATION_CAP_SECONDS = 1800          # protocol: no single indexing operation may exceed 30 minutes
FUNCTIONS, PER_FILE = 40, 4
OPERATIONS = ("unchanged", "one", "ten", "add", "delete", "rename")


def rss_bytes() -> int:
    """Peak resident memory of this process so far (bytes)."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", 0)
    if not peak:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except ImportError:
            peak = 0
    return int(peak or info.rss)


# --- P1 ---------------------------------------------------------------------

def file_source(file_index: int, values: list[int]) -> str:
    functions = []
    for offset in range(PER_FILE):
        n, v = file_index * PER_FILE + offset, values[file_index * PER_FILE + offset]
        functions.append(f'def add_{n}(value):\n    """Add {v} to an integer."""\n    return value + {v}\n')
    return "\n\n".join(functions)


def write_fixture(folder: Path, values: list[int]) -> None:
    for file_index in range(FUNCTIONS // PER_FILE):
        (folder / f"m{file_index}.py").write_text(file_source(file_index, values), encoding="utf-8")


def encoder_inputs(folder: Path) -> set[str]:
    found = set()
    for path in sorted(folder.rglob("*.py")):
        for doc in _documents_from_source(path.read_text(encoding="utf-8"), path.relative_to(folder).as_posix()):
            text = _encoder_input(doc)
            if len(text) <= DENSE_MAX_INPUT_CHARS:
                found.add(text)
    return found


def build(bench: Workbench, key: str, rebuild: bool) -> dict:
    started = time.perf_counter()
    bench.start_build(key, dense=True, rebuild=rebuild)
    while time.perf_counter() - started < OPERATION_CAP_SECONDS:
        state = bench.state()
        if state["phase"] == "error":
            raise RuntimeError(state)
        if state["phase"] == "ready":
            return {"seconds": time.perf_counter() - started, "rss_bytes": rss_bytes(),
                    **{k: state.get(k) for k in ("encoded", "reused", "chunks")}}
        time.sleep(0.02)
    raise TimeoutError(f"index operation exceeded {OPERATION_CAP_SECONDS} s")


def identities(result: dict) -> list[tuple]:
    return [(r["file_path"], r["name"], r["score"]) for r in result["results"]]


def p1(encoder, work: Path, repetitions: int = 3) -> dict:
    runs = []
    for repetition in range(repetitions):
        folder = work / f"repo{repetition}"
        folder.mkdir()
        values = list(range(FUNCTIONS))
        write_fixture(folder, values)
        bench = Workbench(cache=EmbeddingCache(work / f"cache{repetition}.db"))
        bench._encoder = encoder
        key = bench.open_folder(str(folder))["key"]
        seen = encoder_inputs(folder)
        record = {"repetition": repetition, "full": build(bench, key, False), "updates": []}
        for operation in OPERATIONS:
            if operation == "one":
                values[0] = 1001
                write_fixture(folder, values)
            elif operation == "ten":
                values[4:14] = [2000 + i for i in range(10)]
                write_fixture(folder, values)
            elif operation == "add":
                (folder / "added.py").write_text('def added(x):\n    """Multiply by 71."""\n    return x * 71\n',
                                                 encoding="utf-8")
            elif operation == "delete":
                (folder / "added.py").unlink()
            elif operation == "rename":
                (folder / "m9.py").rename(folder / "moved.py")
            current = encoder_inputs(folder)
            expected = len(current - seen)
            row = {"operation": operation, "expected_new_inputs": expected, **build(bench, key, True)}
            row["encoded_equals_changed"] = row["encoded"] == expected
            record["updates"].append(row)
            seen |= current
            if not row["encoded_equals_changed"]:
                raise AssertionError(f"cache accounting mismatch: {row}")

        # the edited function (uncached until the update) must be found by its new behaviour
        edited = bench.search(key, "Add 1001 to an integer.", verify=False)
        record["edited_function_retrievable"] = bool(edited["results"]) and edited["results"][0]["name"] == "add_0"

        # equivalence with a fresh rebuild of the same files
        fresh = Workbench(cache=EmbeddingCache(work / f"fresh{repetition}.db"))
        fresh._encoder = encoder
        fresh_key = fresh.open_folder(str(folder))["key"]
        record["fresh"] = build(fresh, fresh_key, False)
        queries = ["Add 1001 to an integer.", "Add 2003 to an integer.", "Add 17 to an integer.",
                   "Multiply by 71.", "add_5", "Add 30 to an integer."]
        same_rankings = sum(identities(bench.search(key, q, verify=False))
                            == identities(fresh.search(fresh_key, q, verify=False)) for q in queries)
        texts = sorted(encoder_inputs(folder))
        cosines = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                   for a, b in ((bench._cache.get(recipe_cache_key(encoder, t)),
                                 fresh._cache.get(recipe_cache_key(encoder, t))) for t in texts)]
        record["equivalence"] = {"queries": len(queries), "identical_rankings": same_rankings,
                                 "chunks_compared": len(cosines), "min_vector_cosine": min(cosines)}
        record["fresh_ranking_equal"] = same_rankings == len(queries) and min(cosines) >= 0.99999
        if not (record["fresh_ranking_equal"] and record["edited_function_retrievable"]):
            raise AssertionError(f"incremental index differs from a fresh rebuild: {record}")
        runs.append(record)
    return {"fixture": f"{FUNCTIONS} Python functions in {FUNCTIONS // PER_FILE} files; {repetitions} repetitions",
            "runs": runs, "summary": summarise_p1(runs)}


def summarise_p1(runs: list[dict]) -> dict:
    def stat(values):
        return {"median": statistics.median(values), "min": min(values), "max": max(values)}

    out = {"full_index": {"seconds": stat([r["full"]["seconds"] for r in runs]),
                          "chunks": runs[0]["full"]["chunks"], "encoded": runs[0]["full"]["encoded"]}}
    for operation in OPERATIONS:
        rows = [u for r in runs for u in r["updates"] if u["operation"] == operation]
        out[operation] = {"seconds": stat([u["seconds"] for u in rows]),
                          "encoded": stat([u["encoded"] for u in rows]),
                          "reused": stat([u["reused"] for u in rows]),
                          "rss_bytes_max": max(u["rss_bytes"] for u in rows)}
    out["fresh_rebuild_seconds"] = stat([r["fresh"]["seconds"] for r in runs])
    out["all_encoded_counts_exact"] = all(u["encoded_equals_changed"] for r in runs for u in r["updates"])
    out["all_equivalent_to_fresh_rebuild"] = all(r["fresh_ranking_equal"] for r in runs)
    return out


# --- Bonus ------------------------------------------------------------------

LINEAGES = 12


def lineage_code(lineage: int, version: int) -> tuple[str, bool]:
    """(source, is_correct) of one lineage in one version.

    Each document is a whole stdin/stdout program (like an APPS corpus document), so it can be
    run against the worked examples; its identity across versions is its file path and name.

    Lineages 0-8: the correct variant sits in version (lineage mod 3); the wrong variants are two
    different bugs (odd lineages) or the same bug repeated (even lineages). Lineages 9-11 never
    change: the correct code in all three versions.
    """
    target = lineage + 1
    if lineage >= 9 or version == lineage % 3:
        return f"n = int(input())\nprint(n + {target})\n", True
    bug = target + 100 if (lineage % 2 == 0 or version % 2 == 0) else target + 200
    return f"n = int(input())\nprint(n + {bug})\n", False


def bonus(encoder, work: Path) -> dict:
    index = VersionedIndex(encoder=encoder)
    vectors: dict[str, np.ndarray] = {}
    refs = ("v1", "v2", "v3")
    for version, ref in enumerate(refs):
        docs = []
        for lineage in range(LINEAGES):
            code, _ = lineage_code(lineage, version)
            docs.append(Document(document_id=f"{lineage}-{version}", file_path=f"lineage{lineage}.py",
                                 content=code, name=f"add_{lineage + 1}", type="FUNCTION"))
        for d in docs:
            text = _encoder_input(d)
            if text not in vectors:
                vectors[text] = np.asarray(encoder.encode([text], mode="document", batch_size=1)[0])
        index.add_snapshot(ref, docs, embeddings=np.vstack([vectors[_encoder_input(d)] for d in docs]))
    target = Target(key="validation:versions", label="validation fixture", language="Python", kind="git")
    built = TargetIndex(target=target, index=index, versions=[
        VersionInfo(ref=ref, sha=ref, subject=f"version {i + 1}", author="", timestamp=i, chunks=len(index._snapshots[ref]),
                    dense=True) for i, ref in enumerate(refs)])
    bench = Workbench(cache=EmbeddingCache(work / "bonus.db"))
    bench._encoder = encoder
    bench._built[target.key] = built

    rows = []
    for lineage in range(LINEAGES):
        n = lineage + 1
        query = (f"Add {n} to the input integer.\n\n-----Example-----\nInput\n3\n\nOutput\n{3 + n}\n\n"
                 f"-----Example-----\nInput\n8\n\nOutput\n{8 + n}\n")
        correct = lineage_code(lineage, lineage % 3 if lineage < 9 else 0)[0]
        started = time.perf_counter()
        result = bench.across(target.key, query, limit=10)
        seconds = time.perf_counter() - started
        family, collapsed, raw = f"add_{n}", result["collapsed"], result["ungrouped"]
        control = index.retrieve_across_versions(query, limit=10)
        rank = next((j for j, r in enumerate(raw) if r["name"] == family and r["code"] == correct), len(raw))
        rows.append({
            "lineage": lineage, "kind": "changes" if lineage < 9 else "unchanged", "seconds": seconds,
            "correct_version_first": bool(collapsed) and collapsed[0]["name"] == family and collapsed[0]["code"] == correct,
            "lineage_first": bool(collapsed) and collapsed[0]["name"] == family,
            "wrong_above_correct": any(r["name"] == family and r["code"] != correct for r in raw[:rank]),
            "distinct_at_10": result["distinct_collapsed"],
            "verified_by_execution": bool(collapsed) and "execution" in collapsed[0]["sources"],
            "top_versions": collapsed[0]["versions"] if collapsed else [],
            "control_no_verification_correct_first": bool(control) and control[0].document.name == family
            and control[0].document.content == correct,
        })
    keys = ("correct_version_first", "lineage_first", "wrong_above_correct", "verified_by_execution",
            "control_no_verification_correct_first")
    metrics = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    metrics.update(distinct_at_10=float(np.mean([r["distinct_at_10"] for r in rows])),
                   query_seconds_median=statistics.median(r["seconds"] for r in rows),
                   query_seconds_max=max(r["seconds"] for r in rows))
    return {"kind": "functional synthetic diagnostic (wrong variants fail the worked examples); "
                    "not an estimate of accuracy on unseen repositories",
            "path": "Workbench.across (backs /api/workbench/across)", "versions": len(refs), "lineages": LINEAGES,
            "metrics": metrics, "queries": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tier", choices=["accuracy", "interactive"], default="accuracy")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    encoder = CodeEncoder(tier=args.tier, device="cpu", torch_dtype="float32")
    smoke = encoder.encode(["Return the sum of two integers."], mode="query", batch_size=1)
    assert smoke.shape[0] == 1 and np.isfinite(smoke).all()
    metadata = {"model": encoder.model_name, "revision": encoder.revision, "tier": args.tier,
                "dimension": int(smoke.shape[1]), "device": encoder.device, "dtype": encoder.torch_dtype,
                "max_batch": encoder.max_batch, "platform": platform.platform(),
                "cold_load_and_first_query_seconds": time.perf_counter() - started,
                "execution": {k: v for k, v in vars(execution_config()).items() if not k.startswith("_")
                              and k in ("call_style", "blank_line_fallback", "judge_like_io", "deepen_to",
                                        "max_candidates", "timeout_mode", "skip_interactive")}}
    (args.out / "validation-environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="codedoctor-validation-") as folder:
        work = Path(folder)
        result = p1(encoder, work)
        (args.out / "p1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("P1 done:", json.dumps(result["summary"]["all_encoded_counts_exact"]), flush=True)
        result = bonus(encoder, work)
        (args.out / "bonus.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("Bonus done:", json.dumps(result["metrics"]), flush=True)
    print("Hands-on validation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
