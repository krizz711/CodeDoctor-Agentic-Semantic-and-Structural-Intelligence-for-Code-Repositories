"""E2 (experiments/C3_cycle/PREREG.md): score the int8 query vectors by replay.

    python scripts/exp_e2_replay.py --kernel-out build/kernel_e2b/out/e2 --arm T1 \
        --out experiments/C3_cycle/E2_result_dev.json

Dense scores of the returned DEV (and EXT-dev) query vectors against the stored
1.7B corpus vectors, then the candidate_v2 policy (call-style + deepening to 50,
as registered: blank fallback OFF) through union_replay.  (statement, document)
pairs no store recorded are executed live by ``fill_missing`` with the
recorder's configuration; its fidelity is checked first on 100 recorded pairs.
The pre-registered quality rule is applied mechanically.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.search import build_candidate_pool  # noqa: E402
from scripts.c3_stats import compare  # noqa: E402
from scripts.replay_outcomes import Policy, candidate_tier, metrics, rerank, stdin_verdict  # noqa: E402
from scripts.union_replay import (  # noqa: E402
    OUT,
    STORES,
    entry_for,
    fill_missing,
    load_store_entries,
    outcome_index,
)

APPS, EXT = REPO_ROOT / "artifacts" / "apps", REPO_ROOT / "artifacts" / "ext_cc"
V2 = Policy("call-style+deep50", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
STDIN = "competitive_stdin"
SEED = 20260926
EXT_STORES = [OUT / "f2llm17b_extcc_dev_deep50.jsonl.gz", OUT / "f2llm17b_extcc_dev.jsonl.gz",
              OUT / "f2llm06b_extcc_dev_deep50.jsonl.gz", OUT / "f2llm06b_extcc_dev.jsonl.gz"]


def texts(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        return {r["id"]: r["text"] for r in map(json.loads, handle)}


def unit_rows(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def fidelity_check(base: dict, corpus: dict, workers: int) -> dict:
    """fill_missing on 100 already-recorded pairs (seed 20260926, every recorded timeout included)."""
    pairs, timeouts = [], []
    for entry in base.values():
        if not entry["stdin_examples"]:
            continue
        for record in entry["candidates"]:
            runs = record.get("stdin_runs") or []
            (timeouts if any(r["status"] == "timeout" for r in runs) else pairs).append((entry, record))
    rng = random.Random(SEED)
    # every recorded timeout in these statements (up to half the sample), the rest at random
    rng.shuffle(timeouts)
    chosen = timeouts[:50]
    chosen += rng.sample(pairs, min(len(pairs), 100 - len(chosen)))
    redone = fill_missing([(e, r["doc"]) for e, r in chosen], corpus, workers=workers)
    same = 0
    for (entry, record), (_, _, new) in zip(chosen, redone):
        ok = True
        for i, example in enumerate(entry["stdin_examples"][:2]):
            old_runs, new_runs = record.get("stdin_runs") or [], new.get("stdin_runs") or []
            if i < len(old_runs) and i < len(new_runs):
                ok &= (stdin_verdict(old_runs[i], example["expected"], entry["rules"], V2)
                       == stdin_verdict(new_runs[i], example["expected"], entry["rules"], V2))
        same += ok
    return {"n": len(chosen), "timeouts_included": min(50, len(timeouts)), "timeouts_available": len(timeouts),
            "identical": same,
            "pass": same >= len(chosen) - 1}


def score(split: str, qvec: np.ndarray, qids: list[str], workers: int) -> dict:
    if split == "dev":
        corpus_vec = np.load(APPS / "emb_f2llm17b_corpus.npy").astype(np.float32)
        cids = json.loads((APPS / "ids_corpus.json").read_text(encoding="utf-8"))
        corpus = texts(APPS / "texts_corpus.jsonl")
        gold = {q: next(iter(v)) for q, v in json.loads((APPS / "qrels_train.json").read_text(encoding="utf-8")).items()}
        stores = STORES["dev"]
    else:
        corpus = texts(EXT / "texts_corpus.jsonl")
        cids = list(corpus)
        corpus_vec = np.load(EXT / "emb_f2llm17b_corpus.npy").astype(np.float32)
        gold = {q: next(iter(v)) for q, v in json.loads((EXT / "qrels.json").read_text(encoding="utf-8")).items()}
        stores = EXT_STORES
    base = load_store_entries(stores[0])
    index = outcome_index(base, [p for p in stores if p.exists()])
    pool = build_candidate_pool(unit_rows(qvec), corpus_vec, depth=50)

    docs_of = {qid: [cids[j] for j in pool.indices[row]] for row, qid in enumerate(qids)}
    scores_of = {qid: pool.scores[row].tolist() for row, qid in enumerate(qids)}

    def build(qid):
        return entry_for(base[qid], docs_of[qid], scores_of[qid], index[qid], depth=50)

    # Only pairs the policy uses are executed: the top 20 always, ranks 21-50
    # only for statements where nothing in the top 20 verifies (deepening).
    filled_docs: set = set()
    needing: set = set()
    for window in (range(0, V2.exec_depth), range(V2.exec_depth, V2.deepen_to)):
        wanted = []
        for qid in qids:
            entry, lack = build(qid)
            if window.start > 0 and any(candidate_tier(r, entry, V2) == "strong"
                                        for r in entry["candidates"] if r["j"] < V2.exec_depth):
                continue
            lack = set(lack)
            wanted += [(base[qid], d) for j, d in enumerate(docs_of[qid]) if j in window and d in lack]
        if wanted:
            for qid, doc, record in fill_missing(wanted, corpus, workers=workers):
                index[qid][doc] = {k: v for k, v in record.items() if k != "j"}
                filled_docs.add((qid, doc))
                needing.add(qid)
    entries = {qid: build(qid)[0] for qid in qids}
    result = {"split": split, "n": len(qids), "filled_pairs": len(filled_docs),
              "statements_needing_fill": len(needing)}
    result["fill_missing_fidelity"] = fidelity_check({q: base[q] for q in (needing or set(qids))}, corpus, workers)

    control = np.array([metrics(rerank(base[q], V2), gold[q])[:2] for q in qids])
    treated = np.array([metrics(rerank(entries[q], V2), gold[q])[:2] for q in qids])
    dense_c = np.array([metrics(base[q]["pool"], gold[q])[0] for q in qids])
    dense_t = np.array([metrics(entries[q]["pool"], gold[q])[0] for q in qids])
    stdin = np.array([base[q]["family"] == STDIN for q in qids])
    result.update({
        "pipeline_ndcg10": compare(control[:, 0], treated[:, 0]),
        "pipeline_mrr10": compare(control[:, 1], treated[:, 1]),
        "dense_ndcg10": compare(dense_c, dense_t),
        "stdin_pipeline_ndcg10": compare(control[stdin, 0], treated[stdin, 0]) if stdin.any() else None,
    })
    # statements whose treated ranking uses a live-filled pair, and the primary without them
    used = [q for q in qids if any((q, d) in filled_docs for d in entries[q]["pool"][:50])]
    keep = np.array([q not in set(used) for q in qids])
    result["statements_using_filled_pairs"] = len(used)
    result["pipeline_ndcg10_without_them"] = compare(control[keep, 0], treated[keep, 0]) if keep.any() else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-out", type=Path, required=True, help="kernel B's /kaggle/working/e2 folder")
    parser.add_argument("--arm", default="T1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    base = load_store_entries(STORES["dev"][0])
    result: dict = {}
    # S5: the stored-vector replay reproduces the registered baseline
    gold = {q: next(iter(v)) for q, v in json.loads((APPS / "qrels_train.json").read_text(encoding="utf-8")).items()}
    a0 = float(np.mean([metrics(rerank(e, V2), gold[q])[0] for q, e in base.items()]))
    result["S5_A0_reproduced"] = round(a0, 5)
    result["S5_pass"] = round(a0, 5) == 0.95327

    for split in ("dev", "extdev"):
        vec = np.load(args.kernel_out / f"{split}_{args.arm}.npy")
        qids = json.loads((args.kernel_out / f"{split}_{args.arm}_qids.json").read_text(encoding="utf-8"))
        result[split] = score(split if split == "dev" else "ext", vec, qids, args.workers)

    dev = result["dev"]
    p, s = dev["pipeline_ndcg10"], dev["stdin_pipeline_ndcg10"]
    rules = {
        "1_ci_lower_gt_-0.002": p["ci95"][0] > -0.002,
        "2_no_significant_decline": not (p["delta"] < 0 and p["sign_flip_p"] < 0.05),
        "3_stdin_delta_ge_-0.002": s["delta"] >= -0.002,
        "4_extdev_ci_upper_ge_0": result["extdev"]["pipeline_ndcg10"]["ci95"][1] >= 0,
    }
    result["quality_rules"] = rules
    result["sanity"] = {"S5": result["S5_pass"],
                        "fill_missing_fidelity_dev": result["dev"]["fill_missing_fidelity"]["pass"],
                        "fill_missing_fidelity_ext": result["extdev"]["fill_missing_fidelity"]["pass"]}
    result["non_inferior"] = bool(all(rules.values()) and all(result["sanity"].values()))
    text = json.dumps(result, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
