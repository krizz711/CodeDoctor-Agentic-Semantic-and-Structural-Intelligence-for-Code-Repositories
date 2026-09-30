"""E3 Part 3 (experiments/C3_cycle/PREREG.md): P1 re-index, measured end to end.

    python scripts/exp_e3_p1_timing.py --out experiments/C3_cycle/E3_result.json

(R) the demo workbench on a COPY of this repo's app/ (edits never touch the real
code); (A) the scored path's content-addressed corpus cache on the APPS corpus.
Operating point: F2LLM-v2-0.6B, CPU, float32, threads as the shipped encoder sets
them (measured).  Descriptive, with the pre-registered claim rule and gates.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEED = 20260926
KILL_RUN, KILL_FULL = 600.0, 1800.0
#: --dry checks the mechanics on app/indexing with k in {0, 1}; never a result
DRY = {"on": False}


def available_gb() -> float:
    import psutil

    return round(psutil.virtual_memory().available / 1e9, 2)


def wait(bench, limit: float) -> tuple[dict, dict]:
    """Poll the build; returns (final state, first time each phase was seen)."""
    seen: dict[str, float] = {}
    start = time.perf_counter()
    while True:
        state = bench.state()
        seen.setdefault(state["phase"], round(time.perf_counter() - start, 3))
        if state["phase"] in ("ready", "error"):
            return state, seen
        if time.perf_counter() - start > limit:
            return {**state, "phase": "killed"}, seen
        time.sleep(0.02)


def build_timed(bench, key: str, rebuild: bool, limit: float, query: str) -> dict:
    t0 = time.perf_counter()
    bench.start_build(key, dense=True, rebuild=rebuild)
    state, phases = wait(bench, limit)
    ready = time.perf_counter() - t0
    out = {"phase": state["phase"], "phases_s": phases, "ready_s": round(ready, 3),
           "encoded": state.get("encoded"), "reused": state.get("reused"), "chunks": state.get("chunks"),
           "dense": state.get("dense"), "free_gb": available_gb()}
    if state["phase"] == "ready":
        t1 = time.perf_counter()
        bench.search(key, query, verify=False)
        out["first_dense_answer_s"] = round(time.perf_counter() - t1, 3)
        out["end_to_end_s"] = round(time.perf_counter() - t0, 3)
    return out


def encoder_inputs(root: Path) -> dict[str, list[str]]:
    from app.workbench import DENSE_MAX_INPUT_CHARS, _documents_from_source, _encoder_input, _relative_source_files

    out = {}
    for rel in _relative_source_files(root, ""):
        docs = _documents_from_source((root / rel).read_text(encoding="utf-8", errors="ignore"), rel)
        out[rel] = [_encoder_input(d) for d in docs if len(_encoder_input(d)) <= DENSE_MAX_INPUT_CHARS]
    return out


def changed_inputs(before: dict, after: dict) -> int:
    old = {t for texts in before.values() for t in texts}
    return len({t for texts in after.values() for t in texts} - old)


def edit_functions(root: Path, k: int, tag: str, rng: random.Random) -> int:
    """Insert a comment as the first body line of k distinct top-level/method functions."""
    import re

    sites = []
    for path in sorted(root.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(lines):
            m = re.match(r"^(\s*)def \w+\(.*\).*:\s*$", line)
            if m and i + 1 < len(lines) and lines[i + 1].strip():
                sites.append((path, i, lines[i + 1][: len(lines[i + 1]) - len(lines[i + 1].lstrip())]))
    chosen = rng.sample(sites, k) if k else []
    by_file: dict[Path, list] = {}
    for path, i, indent in chosen:
        by_file.setdefault(path, []).append((i, indent))
    for path, edits in by_file.items():
        lines = path.read_text(encoding="utf-8").split("\n")
        for i, indent in sorted(edits, reverse=True):
            lines.insert(i + 1, f"{indent}# p1 edit {tag}")
        path.write_text("\n".join(lines), encoding="utf-8")
    return len(chosen)


def ols(x: list[float], y: list[float]) -> tuple[float, float]:
    a = np.vstack([np.asarray(x, float), np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(a, np.asarray(y, float), rcond=None)[0]
    return float(slope), float(intercept)


def part_r(work: Path, result: dict) -> None:
    from app.indexing.cache import EmbeddingCache, recipe_cache_key
    from app.retrieval.encode import get_shared_encoder
    from app.workbench import Workbench

    root = work / "app_copy"
    source = REPO_ROOT / "app" / "indexing" if DRY["on"] else REPO_ROOT / "app"
    shutil.copytree(source, root, ignore=shutil.ignore_patterns("__pycache__"))
    encoder = get_shared_encoder(tier="interactive")
    t = time.perf_counter()
    encoder.encode("warm up the query path", mode="query")
    result["model_load_s"] = round(time.perf_counter() - t, 2)
    result["encoder"] = {"model": encoder.model_name, "revision": encoder.revision, "device": encoder.device,
                         "dtype": encoder.torch_dtype, "threads_set_by_encode": __import__("os").cpu_count()}

    # batch-1 sanity and batch-1 vs windows of 8 (24 seeded chunks)
    sample = random.Random(SEED).sample([t for v in encoder_inputs(root).values() for t in v], 24)
    t = time.perf_counter()
    for text in sample:
        encoder.encode([text], mode="document")
    b1 = (time.perf_counter() - t) / len(sample)
    t = time.perf_counter()
    for i in range(0, len(sample), 8):
        encoder.encode(sample[i:i + 8], mode="document")
    w8 = (time.perf_counter() - t) / len(sample)
    result["per_doc_batch1_s"], result["per_doc_windows8_s"] = round(b1, 3), round(w8, 3)
    result["sanity_batch1_within_25pct_of_1.275"] = abs(b1 / 1.275 - 1) <= 0.25

    # the full-rebuild baseline: empty cache, same operating point, model warm
    bench = Workbench(cache=EmbeddingCache(work / "cache_r.db"))
    bench._encoder = encoder
    key = bench.open_folder(str(root))["key"]
    query = "rank candidates by executing them on the worked examples"
    full = build_timed(bench, key, rebuild=False, limit=KILL_FULL, query=query)
    phases = full["phases_s"]
    encode_s = full["ready_s"] - phases.get("encoding", phases.get("searchable", 0.0))
    full["per_doc_in_path_s"] = round(encode_s / max(1, full["encoded"] or 1), 3)
    result["R_full_rebuild"] = full
    if full["phase"] != "ready":
        result["void"] = f"full rebuild did not finish: {full['phase']}"
        return

    rng = random.Random(SEED)
    runs = []
    for k in ((0, 1) if DRY["on"] else (0, 1, 10, 100)):
        for rep in range(1 if DRY["on"] else 3):
            before = encoder_inputs(root)
            edited = edit_functions(root, k, f"k{k}r{rep}", rng)
            expected = changed_inputs(before, encoder_inputs(root))
            run = build_timed(bench, key, rebuild=True, limit=KILL_RUN, query=query)
            run.update({"k": k, "rep": rep, "functions_edited": edited, "expected_encodes": expected})
            runs.append(run)
            if run["phase"] != "ready":
                result["void"] = f"k={k} rep={rep}: {run['phase']}"
                result["R_runs"] = runs
                return
    result["R_runs"] = runs

    events = {}
    before = encoder_inputs(root)
    (root / "p1_added.py").write_text("def p1_added_one(x):\n    return x + 11\n\n\ndef p1_added_two(y):\n    return y * 13\n",
                                      encoding="utf-8")
    events["add_file"] = build_timed(bench, key, True, KILL_RUN, query) | {"expected_encodes": changed_inputs(before, encoder_inputs(root))}
    (root / "p1_added.py").unlink()
    events["delete_file"] = build_timed(bench, key, True, KILL_RUN, query) | {"expected_encodes": 0}
    victim = sorted(root.rglob("*.py"))[3]
    target = root / "moved_here"
    target.mkdir(exist_ok=True)
    shutil.move(str(victim), str(target / victim.name))
    events["pure_move"] = build_timed(bench, key, True, KILL_RUN, query) | {"expected_encodes": 0, "moved": victim.name}
    result["R_events"] = events

    # parity: the final tree rebuilt from an empty cache at the same operating point
    fresh = Workbench(cache=EmbeddingCache(work / "cache_r_fresh.db"))
    fresh._encoder = encoder
    fkey = fresh.open_folder(str(root))["key"]
    result["R_parity_full_rebuild"] = build_timed(fresh, fkey, rebuild=False, limit=KILL_FULL, query=query)
    texts = sorted({t for v in encoder_inputs(root).values() for t in v})
    inc = [bench._cache.get(recipe_cache_key(encoder, t)) for t in texts]
    new = [fresh._cache.get(recipe_cache_key(encoder, t)) for t in texts]
    cos = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(inc, new)
           if a is not None and b is not None]
    queries = [q["query"] for q in bench._built[key].preset_queries][:10]
    names = sorted({d.name for d in bench._built[key].index.documents_at("working") if d.name})
    queries += random.Random(SEED).sample(names, max(0, 20 - len(queries)))
    same = sum([r["document_id"] for r in bench.search(key, q, verify=False)["results"]]
               == [r["document_id"] for r in fresh.search(fkey, q, verify=False)["results"]] for q in queries)
    result["R_parity"] = {"chunks_compared": len(cos), "cos_min": round(min(cos), 7) if cos else None,
                          "queries": len(queries), "top10_identical": same,
                          "pass": bool(cos and min(cos) >= 0.99999 and same == len(queries))}

    # claim rule (R)
    xs = [r["encoded"] for r in runs]
    ys = [r["end_to_end_s"] for r in runs]
    slope, intercept = ols(xs, ys)
    med = {k: float(np.median([r["end_to_end_s"] for r in runs if r["k"] == k])) for k in sorted({r["k"] for r in runs})}
    exact = all(r["encoded"] == r["expected_encodes"] for r in runs) and all(
        e["encoded"] == e["expected_encodes"] for e in events.values())
    k0_warm = float(np.median([r["ready_s"] for r in runs if r["k"] == 0]))
    rate = full["per_doc_in_path_s"]
    result["R_claim"] = {
        "slope_s_per_encoded_chunk": round(slope, 3), "intercept_s": round(intercept, 3),
        "per_doc_rate_in_path_s": rate, "median_end_to_end_by_k": med, "full_rebuild_ready_s": full["ready_s"],
        "rule1_encodes_equal_changed_chunks": exact,
        "rule2_slope_within_30pct_of_rate": abs(slope / rate - 1) <= 0.30 if rate else False,
        "rule3_k0_warm_le_10s": k0_warm <= 10.0,
        "rule4_k10_le_0.15x_full": med.get(10, float("inf")) <= 0.15 * full["ready_s"],
    }
    result["R_claim"]["claim_allowed"] = all(v for k, v in result["R_claim"].items() if k.startswith("rule"))


def part_a(work: Path, result: dict) -> None:
    """The scored path: content-addressed corpus cache seeded like a judge's (stored GPU vectors)."""
    from app.indexing.cache import EmbeddingCache, recipe_cache_key
    from app.pipeline.encoder import PrePostPipelineEncoder
    from app.retrieval.encode import get_shared_encoder

    apps = REPO_ROOT / "artifacts" / "apps"
    encoder = get_shared_encoder(tier="interactive")
    ids = json.loads((apps / "ids_corpus.json").read_text(encoding="utf-8"))
    with open(apps / "texts_corpus.jsonl", encoding="utf-8") as handle:
        text_of = {r["id"]: r["text"] for r in map(json.loads, handle)}
    corpus = [text_of[d] for d in ids]
    stored = np.load(apps / "emb_f2llm06b_corpus.npy").astype(np.float32)
    cache = EmbeddingCache(work / "cache_a.db")
    cache.put_many([(recipe_cache_key(encoder, t), stored[i]) for i, t in enumerate(corpus)])
    pipe = PrePostPipelineEncoder(encoder, corpus_cache=cache)

    # negative gate: a changed recipe must miss everything (key lookups only, no encoding)
    class Changed:
        def __init__(self, **kw):
            self.__dict__.update({k: getattr(encoder, k) for k in ("model_name", "revision", "document_prefix",
                                                                   "max_seq_length", "add_eos_token",
                                                                   "normalize_embeddings")})
            self.__dict__.update(kw)

    variants = {"revision": Changed(revision="other"), "prefix": Changed(document_prefix="X: "),
                "max_seq": Changed(max_seq_length=512), "eos": Changed(add_eos_token=False),
                "norm": Changed(normalize_embeddings=False), "model": Changed(model_name="other/model")}
    hits = {name: len(cache.get_many([recipe_cache_key(v, t) for t in corpus[:500]])) for name, v in variants.items()}
    result["A_negative_gate_hits"] = hits
    result["A_negative_gate_pass"] = all(h == 0 for h in hits.values())

    qrels = json.loads((apps / "qrels_train.json").read_text(encoding="utf-8"))
    import gzip
    with gzip.open(REPO_ROOT / "data" / "outcomes" / "f2llm17b_all_tune.jsonl.gz", "rt", encoding="utf-8") as handle:
        handle.readline()
        tune_qids = [json.loads(line)["qid"] for line in handle]
    golds = sorted({next(iter(qrels[q])) for q in tune_qids}, key=lambda d: int(d[1:]))
    row_of = {d: i for i, d in enumerate(ids)}
    rng = random.Random(SEED)
    runs = []
    texts = list(corpus)
    for k in ((1,) if DRY["on"] else (1, 10, 100)):
        for rep in range(1 if DRY["on"] else 3):
            chosen = rng.sample(golds, k)
            for d in chosen:
                texts[row_of[d]] = texts[row_of[d]] + f"\n# p1 edit k{k}r{rep}\n"
            pipe.cache_stats = {"hits": 0, "misses": 0}
            t = time.perf_counter()
            matrix = pipe._encode_documents_cached(texts)
            runs.append({"k": k, "rep": rep, "seconds": round(time.perf_counter() - t, 3), **pipe.cache_stats})
    result["A_runs"] = runs
    rate = float(np.mean([r["seconds"] / r["misses"] for r in runs if r["k"] == max(x["k"] for x in runs)]))
    result["A_per_doc_s"] = round(rate, 3)
    result["A_full_rebuild_estimate_s"] = round(rate * len(corpus), 1)
    result["A_full_rebuild_is"] = "estimate: measured per-document rate x 8,765"
    # parity (A): unchanged rows bit-identical to the seed; changed rows equal a fresh encode
    changed_rows = sorted({i for i, (a, b) in enumerate(zip(texts, corpus)) if a != b})
    unchanged = np.setdiff1d(np.arange(len(corpus)), changed_rows)
    identical = bool(np.array_equal(matrix[unchanged], stored[unchanged]))
    again = np.asarray(encoder.encode([texts[i] for i in changed_rows], mode="document"), dtype=np.float32)
    cos = (matrix[changed_rows] * again).sum(1) / np.linalg.norm(matrix[changed_rows], axis=1) / np.linalg.norm(again, axis=1)
    q = np.load(apps / "emb_f2llm06b_qtrain.npy").astype(np.float32)
    qrow = {x: i for i, x in enumerate(json.loads((apps / "ids_queries_train.json").read_text(encoding="utf-8")))}
    qv = q[[qrow[x] for x in tune_qids]]
    scratch = stored.copy()
    scratch[changed_rows] = again
    top_inc = np.argsort(-(qv @ matrix.T), axis=1)[:, :10]
    top_new = np.argsort(-(qv @ scratch.T), axis=1)[:, :10]
    same = float((top_inc == top_new).all(axis=1).mean())
    result["A_parity"] = {"unchanged_bit_identical": identical, "changed_rows": len(changed_rows),
                          "changed_cos_min": round(float(cos.min()), 7), "top10_identity_tune": round(same, 4),
                          "pass": bool(identical and cos.min() >= 0.99999 and same >= 0.995)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry", action="store_true", help="mechanics check on app/indexing; not a result")
    args = parser.parse_args(argv)
    DRY["on"] = args.dry
    result: dict = {"preflight_available_gb": available_gb(), "machine": "this laptop", "dry": args.dry}
    if result["preflight_available_gb"] < 3.0:
        result["void"] = "pre-flight: under 3 GB free; close applications and re-run (not a void attempt)"
        print(json.dumps(result))
        return 2
    with tempfile.TemporaryDirectory(prefix="codedoctor-e3-") as tmp:
        work = Path(tmp)
        try:
            part_r(work, result)
            if "void" not in result:
                part_a(work, result)
        finally:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k in ("R_claim", "R_parity", "A_parity", "void")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
