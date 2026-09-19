"""Paired-bootstrap comparison between two or more dev_eval runs.

``dev_eval.py`` bootstraps each stage against the stage below it inside one
run.  It cannot compare two runs, because pairing needs the same query in both
vectors.  This does that, against a named control:

    python scripts/compare_arms.py --control data/arm_control.json \\
        data/arm_early.json data/arm_gate020.json

Every arm must have been produced with ``--report`` and must cover the same
queries in the same order; that is checked rather than assumed.
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

from app.pipeline.metrics import paired_bootstrap  # noqa: E402


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "per_query_ndcg" not in payload:
        raise SystemExit(
            f"{path} has no per_query_ndcg -- it predates the --report change, "
            "so it cannot be paired. Re-run that arm."
        )
    payload["_name"] = path.stem
    return payload


def describe(arm: dict) -> str:
    bits = [f"depth {arm.get('execution_candidates_budget', 0) // max(1, arm.get('execution_fired', 1))}"]
    if arm.get("execution_early_exit"):
        bits.append("early-exit")
    gap = arm.get("execution_max_dense_gap")
    if gap is not None:
        bits.append(f"gate {gap:g}")
    return ", ".join(bits) if len(bits) > 1 else bits[0] + " (control)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("arms", type=Path, nargs="+")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    control = load(args.control)
    base = np.array(control["per_query_ndcg"]["final"])
    dense = np.array(control["per_query_ndcg"]["dense"])

    print(f"control : {args.control.stem}  ({describe(control)})")
    print(f"queries : {len(base)}")
    print()
    header = (
        f"{'arm':<22} {'NDCG@10':>9} {'vs control':>11} {'95% CI':>22} "
        f"{'p':>7}  {'exec':>9} {'cost':>7}"
    )
    print(header)
    print("-" * len(header))

    ctrl_run = control.get("execution_candidates_run", 0)
    print(
        f"{'control':<22} {np.mean(base):>9.5f} {'':>11} {'':>22} {'':>7}  "
        f"{ctrl_run:>9,} {'100.0%':>7}"
    )

    for path in args.arms:
        arm = load(path)
        if arm.get("query_ids") != control.get("query_ids"):
            raise SystemExit(f"{path.stem}: query set differs from the control; cannot pair")
        vec = np.array(arm["per_query_ndcg"]["final"])
        boot = paired_bootstrap(base, vec, n_resamples=args.resamples)
        run = arm.get("execution_candidates_run", 0)
        ci = f"[{boot.ci_low:+.5f}, {boot.ci_high:+.5f}]"
        print(
            f"{describe(arm):<22} {np.mean(vec):>9.5f} {boot.mean_delta:>+11.5f} {ci:>22} "
            f"{boot.p_value:>7.4f}  {run:>9,} "
            f"{run / max(1, ctrl_run):>6.1%}"
        )

    print()
    print(f"dense-only reference: {np.mean(dense):.5f}")
    print()
    print("A CI excluding zero means the arm genuinely differs from the control on")
    print("these queries. An arm that matches the control while running far fewer")
    print("candidates is a cost win, not a quality change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
