"""Size and evaluate a promotion gate from one instrumented execution run.

A promotion gate changes which verified candidates receive the bonus.  It
never changes which candidates are executed, so every threshold can be
replayed exactly from a single instrumented run rather than re-executing the
sandbox once per arm.

That shortcut is only worth taking if it is verified, so the first thing this
prints is a replay of the *ungated* configuration against the control run's
recorded score.  If those disagree the replay model is wrong and nothing below
it means anything.

    python scripts/analyze_promotion_gate.py \\
        --dump data/arms/instrumented.npz --control data/arms/control.json
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

from app.pipeline.metrics import (  # noqa: E402
    metrics_from_ranks,
    paired_bootstrap,
    per_query_ndcg,
)
from scripts.dev_eval import rank_from_pool_scores  # noqa: E402

VERIFIED = 1


def replay(
    scores: np.ndarray,
    verified: np.ndarray,
    gold_columns: np.ndarray,
    gap: np.ndarray,
    threshold: float | None,
    bonus: float,
    depth: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Scores and counters for one promotion threshold."""
    top_n = verified.shape[1]
    is_verified = verified == VERIFIED
    promoted = is_verified.copy()
    if threshold is not None:
        promoted &= gap <= threshold

    out = scores.copy()
    rows, cols = np.nonzero(promoted)
    out[rows, cols] += bonus

    # Is candidate j of row i the gold?
    gold_here = np.zeros_like(is_verified)
    inside = (gold_columns >= 0) & (gold_columns < top_n)
    gold_here[np.flatnonzero(inside), gold_columns[inside]] = True

    counters = {
        "verifications": int(is_verified.sum()),
        "promotions": int(promoted.sum()),
        "blocked": int((is_verified & ~promoted).sum()),
        "false_verifications": int((is_verified & ~gold_here).sum()),
        "false_promotions": int((promoted & ~gold_here).sum()),
        "gold_verifications": int((is_verified & gold_here).sum()),
        "gold_promotions": int((promoted & gold_here).sum()),
        "golds_blocked": int((is_verified & gold_here & ~promoted).sum()),
        # What an EXECUTION gate at the same threshold would have run, for
        # comparison. A promotion gate itself saves nothing.
        "would_execute": int((gap <= threshold).sum()) if threshold is not None else int(gap.size),
    }
    return out, counters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=REPO_ROOT / "data" / "arms" / "instrumented.npz")
    parser.add_argument("--control", type=Path, default=REPO_ROOT / "data" / "arms" / "control.json")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    d = np.load(args.dump)
    scores = d["scores"].astype(np.float64)
    verified = d["verified"]
    gold_columns = d["gold_columns"]
    fired = d["fired"]
    top_n = int(d["top_n"])
    depth = int(d["depth"])

    n = scores.shape[0]
    gap = scores[:, :1] - scores[:, :top_n]          # (n, top_n), >= 0
    bonus = float(np.abs(scores).max()) + 1.0        # exactly dev_eval's rule

    is_verified = verified == VERIFIED
    gold_here = np.zeros_like(is_verified)
    inside = (gold_columns >= 0) & (gold_columns < top_n)
    gold_here[np.flatnonzero(inside), gold_columns[inside]] = True

    # ---- point 3: the three distributions ------------------------------
    print("=" * 78)
    print("GAP TO THE BEST CANDIDATE  (top_dense - candidate_dense)")
    print("=" * 78)
    print(f"queries {n}, fired {int(fired.sum())}, execution depth {top_n}, pool depth {depth}")
    print()
    groups = {
        "golds in the top-20": gap[gold_here & (verified != -1)],
        "verified & correct": gap[is_verified & gold_here],
        "verified & WRONG": gap[is_verified & ~gold_here],
        "executed, not verified": gap[(verified == 0)],
    }
    head = f"{'group':<24} {'n':>6} " + " ".join(f"{f'p{p}':>8}" for p in (50, 75, 90, 95, 99)) + f" {'max':>8}"
    print(head)
    print("-" * len(head))
    for name, vals in groups.items():
        if vals.size == 0:
            print(f"{name:<24} {0:>6}")
            continue
        pcts = " ".join(f"{np.percentile(vals, p):>8.4f}" for p in (50, 75, 90, 95, 99))
        print(f"{name:<24} {vals.size:>6} {pcts} {vals.max():>8.4f}")

    fv = gap[is_verified & ~gold_here]
    cv = gap[is_verified & gold_here]
    print()
    print("Separation is what a gate needs. Fraction of each group inside a margin:")
    cand = args.thresholds or [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    print(f"  {'delta':>7}  {'correct kept':>13}  {'wrong kept':>11}  {'wrong removed':>14}")
    for t in cand:
        ck = float((cv <= t).mean()) if cv.size else 0.0
        wk = float((fv <= t).mean()) if fv.size else 0.0
        print(f"  {t:>7.2f}  {ck:>13.4f}  {wk:>11.4f}  {1 - wk:>14.4f}")

    # ---- replay validation --------------------------------------------
    print()
    print("=" * 78)
    print("REPLAY VALIDATION")
    print("=" * 78)
    base_scores, base_counts = replay(scores, verified, gold_columns, gap, None, bonus, depth)
    base_ranks = rank_from_pool_scores(base_scores, gold_columns, depth + 1)
    base_metrics = metrics_from_ranks(base_ranks)
    base_ndcg = per_query_ndcg(base_ranks)

    control = json.loads(args.control.read_text(encoding="utf-8"))
    recorded = float(control["execution"]["ndcg_at_10"])
    delta = abs(base_metrics.ndcg_at_10 - recorded)
    print(f"  replayed ungated NDCG@10 : {base_metrics.ndcg_at_10:.5f}")
    print(f"  control run NDCG@10      : {recorded:.5f}")
    print(f"  difference               : {delta:.2e}  -> {'MATCH' if delta < 5e-5 else 'MISMATCH'}")
    if delta >= 5e-5:
        print("\nReplay does not reproduce the control. Refusing to report thresholds.")
        return 1

    # ---- point 5-7: the arms -------------------------------------------
    thresholds = args.thresholds or [0.01, 0.02, 0.05, 0.10]
    print()
    print("=" * 78)
    print(f"PROMOTION GATE ARMS   (paired bootstrap, {args.resamples} resamples, vs ungated)")
    print("=" * 78)
    head = (
        f"{'arm':<16} {'NDCG@10':>9} {'MRR@10':>9} {'R@1':>7} {'R@10':>7} "
        f"{'delta':>9} {'95% CI':>21} {'p':>7}"
    )
    print(head)
    print("-" * len(head))
    print(
        f"{'ungated':<16} {base_metrics.ndcg_at_10:>9.5f} {base_metrics.mrr_at_10:>9.5f} "
        f"{base_metrics.recall_at_1:>7.4f} {base_metrics.recall_at_10:>7.4f} "
        f"{'':>9} {'':>21} {'':>7}"
    )

    results = {"ungated": {"metrics": base_metrics.as_dict(), "counters": base_counts}}
    for t in thresholds:
        s, c = replay(scores, verified, gold_columns, gap, t, bonus, depth)
        r = rank_from_pool_scores(s, gold_columns, depth + 1)
        m = metrics_from_ranks(r)
        boot = paired_bootstrap(base_ndcg, per_query_ndcg(r), n_resamples=args.resamples)
        ci = f"[{boot.ci_low:+.5f}, {boot.ci_high:+.5f}]"
        flag = "" if boot.ci_low <= 0 <= boot.ci_high else "  *"
        print(
            f"{'gate ' + format(t, 'g'):<16} {m.ndcg_at_10:>9.5f} {m.mrr_at_10:>9.5f} "
            f"{m.recall_at_1:>7.4f} {m.recall_at_10:>7.4f} "
            f"{boot.mean_delta:>+9.5f} {ci:>21} {boot.p_value:>7.4f}{flag}"
        )
        results[f"gate_{t:g}"] = {
            "metrics": m.as_dict(),
            "counters": c,
            "bootstrap": dict(vars(boot)),
            "per_query_ndcg": [round(float(v), 6) for v in per_query_ndcg(r)],
        }

    print()
    print("=" * 78)
    print("MECHANISM  (point 7)")
    print("=" * 78)
    head = (
        f"{'arm':<16} {'verifs':>7} {'promoted':>9} {'blocked':>8} "
        f"{'false promo':>12} {'gold promo':>11} {'golds blocked':>14} {'exec if gated':>14}"
    )
    print(head)
    print("-" * len(head))
    total_exec = base_counts["would_execute"]
    for name, payload in results.items():
        c = payload["counters"]
        print(
            f"{name:<16} {c['verifications']:>7} {c['promotions']:>9} {c['blocked']:>8} "
            f"{c['false_promotions']:>12} {c['gold_promotions']:>11} {c['golds_blocked']:>14} "
            f"{c['would_execute'] / total_exec:>13.1%}"
        )
    print()
    print("A promotion gate executes everything, so its own execution reduction is 0%.")
    print("The last column is what an EXECUTION gate at the same threshold would run.")
    print("* marks an interval excluding zero.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
