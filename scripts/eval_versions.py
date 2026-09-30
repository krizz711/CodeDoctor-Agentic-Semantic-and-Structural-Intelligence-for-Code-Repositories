"""Score cross-version retrieval on the benchmark from build_version_benchmark.py.

    python scripts/record_outcomes.py --external artifacts/versions_apps --model f2llm06b
    python scripts/eval_versions.py --store data/outcomes/f2llm06b_external_ext_dev.jsonl.gz

Three rankings over the same dense pool and the same recorded executions:

* ``dense``      -- the encoder alone;
* ``execution``  -- the shipped rule (verified candidates first, dense order);
* ``lineage``    -- versions grouped into lineages, each represented by the
  version the evidence supports (``app.indexing.lineage.LineageStore``).

Metrics (the ExecRetrieval framing, plus what the brief asks):

* ``exec@1``     -- the *correct version* is ranked first;
* ``lineage@1``  -- *some* version of the right snippet is ranked first;
* ``buggy_above``-- a buggy version of the right snippet outranks the correct
  one (lower is better);
* ``distinct@10``-- distinct lineages in the top 10 (crowding by versions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.indexing.lineage import LineageStore  # noqa: E402
from scripts.replay_outcomes import Policy, candidate_tier, load_store, rerank  # noqa: E402


def _metrics(ranking: list[str], correct: str, lineage_of: dict[str, str]) -> dict[str, float]:
    key = lineage_of.get(correct, correct)
    first = ranking[0] if ranking else None
    try:
        correct_rank = ranking.index(correct)
    except ValueError:
        correct_rank = len(ranking) + 1
    siblings = [i for i, d in enumerate(ranking) if lineage_of.get(d) == key and d != correct]
    top10 = ranking[:10]
    return {
        "exec@1": float(first == correct),
        "lineage@1": float(first is not None and lineage_of.get(first, first) == key),
        "buggy_above": float(any(i < correct_rank for i in siblings)),
        "distinct@10": float(len({lineage_of.get(d, d) for d in top10})),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=REPO_ROOT / "artifacts" / "versions_apps")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    header, entries = load_store(args.store)
    qrels = json.loads((args.bench / "qrels.json").read_text(encoding="utf-8"))
    lineage_meta = json.loads((args.bench / "lineage.json").read_text(encoding="utf-8"))
    lineage_of = {d: m["lineage"] for d, m in lineage_meta.items()}
    policy = Policy("legacy+numeric", numeric_tolerance=1e-6)

    rows = {"dense": [], "execution": [], "lineage": []}
    for entry in entries:
        correct = next(iter(qrels[entry["qid"]]))
        pool = entry["pool"]
        rows["dense"].append(_metrics(pool, correct, lineage_of))
        rows["execution"].append(_metrics(rerank(entry, policy), correct, lineage_of))

        # Lineage ranking: versions in the benchmark carry their lineage key;
        # every other document is its own lineage.
        store = LineageStore()
        store.add_version("corpus", [(lineage_of.get(d, d), d) for d in pool])
        # One store per query is enough here: contents are keyed by *doc id*
        # text so the dense scores below map one-to-one onto pool entries.
        dense = {}
        tiers = {}
        by_doc = {r["doc"]: r for r in entry["candidates"]}
        for doc, score in zip(pool, entry["scores"]):
            digest = next(h for h, c in store.contents.items() if c.text == doc)
            dense[digest] = score
            record = by_doc.get(doc)
            if record is not None:
                tiers[digest] = candidate_tier(record, entry, policy)
        hits = store.rank(dense, tiers=tiers, limit=len(pool), recency=False)
        ranking = [h.best_text for h in hits]
        rows["lineage"].append(_metrics(ranking, correct, lineage_of))

    print(f"store {args.store.name}: {len(entries)} queries, model {header['model']}, "
          f"exec depth {header['exec_depth']}")
    print(f"{'ranking':12s} {'exec@1':>8s} {'lineage@1':>10s} {'buggy_above':>12s} {'distinct@10':>12s}")
    report = {}
    for name, values in rows.items():
        mean = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
        report[name] = mean
        print(f"{name:12s} {mean['exec@1']:8.3f} {mean['lineage@1']:10.3f} "
              f"{mean['buggy_above']:12.3f} {mean['distinct@10']:12.2f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"header": header, "n": len(entries), "results": report},
                                          indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
