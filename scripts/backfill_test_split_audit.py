"""Reconstruct the test-split accesses the audit log never recorded.

``data/test_split_audit_log.json`` was written only by
``DevelopmentHarness.get_test_split``.  Every evaluation that produced a
reported number went through ``scripts/evaluate_mteb.py`` instead, which did
not log, so the file claimed one access when at least six had happened.  The
control was real and did not cover the path that mattered.

``evaluate_mteb.py`` now records automatically.  This writes the history that
was missed, from the run artifacts themselves rather than from memory, and
marks every reconstructed row ``BACKFILLED`` with the file it came from --
an audit log that quietly acquires plausible entries is worth nothing.

    python scripts/backfill_test_split_audit.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.evaluation_harness import AUDIT_LOG_PATH  # noqa: E402

#: (results file, what the run was). Dense baselines are listed without a
#: results file because only their metrics survive, in artifacts/apps/manifest.json.
SOURCES = [
    ("artifacts/kaggle/top10/appsretrieval_results.json",
     "Kaggle kernel codedoctor-apps-eval: execution rerank, depth 10"),
    ("artifacts/kaggle/top20/appsretrieval_results.json",
     "Kaggle kernel codedoctor-apps-eval-top20: execution rerank, depth 20 (SUBMITTED)"),
    ("artifacts/kaggle/notebook/appsretrieval_results.json",
     "Kaggle notebook codedoctor-appsretrieval-reproduction: independent reproduction of the submitted run"),
]

UNDATED = [
    ("F2LLM-v2-0.6B dense-only baseline, recorded in artifacts/apps/manifest.json", 0.90427, 0.88369),
    ("F2LLM-v2-1.7B dense-only baseline, recorded in artifacts/apps/manifest.json", 0.93756, 0.92370),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the file (default: dry run)")
    parser.add_argument("--path", type=Path, default=AUDIT_LOG_PATH)
    args = parser.parse_args(argv)

    existing = json.loads(args.path.read_text(encoding="utf-8")) if args.path.exists() else []
    known = {row.get("caller", "") + str(row.get("metric_ndcg_at_10")) for row in existing}
    rows = list(existing)
    now = datetime.now(timezone.utc).isoformat()

    for label, ndcg, mrr in UNDATED:
        key = f"scripts/evaluate_mteb.py{ndcg}"
        if key in known:
            continue
        rows.append({
            "access_index": len(rows) + 1,
            "timestamp": None,
            "reason": "calibration",
            "caller": "scripts/evaluate_mteb.py",
            "split": "test",
            "rows": 3765,
            "metric_ndcg_at_10": ndcg,
            "metric_mrr_at_10": mrr,
            "status": "BACKFILLED",
            "note": label,
            "reconstructed_on": now,
            "timestamp_unavailable": True,
        })

    for relative, label in SOURCES:
        path = REPO_ROOT / relative
        if not path.exists():
            print(f"  skip (missing): {relative}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = payload.get("scores", {}).get("test", [{}])[0]
        key = f"scripts/evaluate_mteb.py{scores.get('ndcg_at_10')}"
        if key in known and relative.endswith("top20/appsretrieval_results.json"):
            pass  # the submitted run is distinct even if a score repeats
        rows.append({
            "access_index": len(rows) + 1,
            "timestamp": payload.get("date"),
            "reason": "final_run",
            "caller": "scripts/evaluate_mteb.py",
            "split": "test",
            "rows": 3765,
            "metric_ndcg_at_10": scores.get("ndcg_at_10"),
            "metric_mrr_at_10": scores.get("mrr_at_10"),
            "status": "BACKFILLED",
            "note": label,
            "source_artifact": relative,
            "reconstructed_on": now,
        })

    print(f"existing rows : {len(existing)}")
    print(f"after backfill: {len(rows)}")
    for row in rows:
        stamp = (row.get("timestamp") or "no timestamp")[:19]
        print(f"  {row['access_index']:>2}  {stamp:<20} {str(row.get('metric_ndcg_at_10')):<9} "
              f"{row.get('status','')}")

    if not args.apply:
        print("\nDry run. Pass --apply to write.")
        return 0

    args.path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
