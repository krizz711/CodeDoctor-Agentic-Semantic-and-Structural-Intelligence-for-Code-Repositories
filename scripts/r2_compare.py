"""R2 quality side: a smaller encoder's pipeline against the 1.7B pipeline.

Pre-registration: ``experiments/R2_encoder_choice/prereg.md`` (incl. the
stress-test addendum).  Pipeline = call-style + deepening to 50, no gate.

    python scripts/r2_compare.py --small f2llm06b --small-suffix _r2 \
        --out experiments/R2_encoder_choice/result_quality.json

Decision rule for "small is non-inferior to 1.7B":
  bootstrap lower bound of (small - large) > -0.002 on DEV **and** TUNE, and
  >= 95% of 200 random re-splits of the pooled statements non-inferior.
Reported alongside: sign-flip permutation p, statements changed, dense and
call-style-only arms, recall@20/@50, and the CodeContests EXT-dev check.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_outcomes import POLICIES, executions, metrics, paired_bootstrap, rerank  # noqa: E402

POLICY = {p.name: p for p in POLICIES}
PIPELINE = "call-style+deep50"
MARGIN = -0.002


def load(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        handle.readline()
        return {e["qid"]: e for e in map(json.loads, handle)}


def arm(entries: list[dict], gold: dict[str, str], name: str) -> np.ndarray:
    if name == "dense":
        return np.array([metrics(e["pool"], gold[e["qid"]])[0] for e in entries])
    return np.array([metrics(rerank(e, POLICY[name]), gold[e["qid"]])[0] for e in entries])


def recall(entries: list[dict], gold: dict[str, str], k: int) -> float:
    return float(np.mean([gold[e["qid"]] in e["pool"][:k] for e in entries]))


def sign_flip_p(d: np.ndarray, resamples: int = 20000, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(resamples, len(d)))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean())).mean())


def compare(small: list[dict], large: list[dict], gold: dict[str, str], name: str) -> dict:
    a, b = arm(small, gold, name), arm(large, gold, name)
    delta, low, high, p = paired_bootstrap(b, a)
    return {"small": round(float(a.mean()), 5), "large": round(float(b.mean()), 5),
            "delta": round(delta, 5), "ci": [round(low, 5), round(high, 5)], "bootstrap_p": round(p, 4),
            "sign_flip_p": round(sign_flip_p(a - b), 4), "statements_changed": int((a != b).sum())}, a - b


def examples_differ(a: dict[str, dict], b: dict[str, dict]) -> list[str]:
    key = lambda e: (e["stdin_examples"], e["call_examples"], e["rules"])  # noqa: E731
    return sorted(q for q in set(a) & set(b) if key(a[q]) != key(b[q]))


def timeout_rate(entries: dict[str, dict]) -> float:
    n = t = 0
    for e in entries.values():
        for c in e["candidates"]:
            for r in c.get("stdin_runs") or []:
                n += 1
                t += r.get("status") == "timeout"
            if c.get("call_run"):
                n += 1
                t += c["call_run"].get("status") == "timeout"
    return t / max(1, n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", default="f2llm06b")
    parser.add_argument("--small-suffix", default="_r2")
    parser.add_argument("--large", default="f2llm17b")
    parser.add_argument("--outcomes", type=Path, default=REPO_ROOT / "data" / "outcomes")
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "experiments" / "R2_encoder_choice" / "result_quality.json")
    args = parser.parse_args(argv)

    gold = {q: next(iter(v)) for q, v in
            json.loads((args.artifacts / "apps" / "qrels_train.json").read_text()).items()}
    result: dict = {"pipeline": PIPELINE, "margin": MARGIN, "small": args.small, "large": args.large, "halves": {}}
    deltas = []
    for split in ("dev", "tune"):
        s = load(args.outcomes / f"{args.small}_all_{split}{args.small_suffix}_deep50.jsonl.gz")
        l = load(args.outcomes / f"{args.large}_all_{split}_deep50.jsonl.gz")  # noqa: E741
        differ = examples_differ(s, l)
        qids = sorted(set(s) & set(l))
        small, large = [s[q] for q in qids], [l[q] for q in qids]
        half = {"statements": len(qids), "examples_differ": differ,
                "timeout_rate": {"small": round(timeout_rate(s), 5), "large": round(timeout_rate(l), 5)},
                "recall@20": {"small": recall(small, gold, 20), "large": recall(large, gold, 20)},
                "recall@50": {"small": recall(small, gold, 50), "large": recall(large, gold, 50)},
                "executions": {"small": sum(executions(e, POLICY[PIPELINE]) for e in small),
                               "large": sum(executions(e, POLICY[PIPELINE]) for e in large)}}
        for name in ("dense", "call-style", PIPELINE):
            half[name], d = compare(small, large, gold, name)
            if name == PIPELINE:
                deltas.append(d)
        result["halves"][split] = half

    pooled = np.concatenate(deltas)
    rng = np.random.default_rng(20260924)
    ok = 0
    for _ in range(200):
        h = pooled[rng.permutation(len(pooled))[: len(pooled) // 2]]
        ok += (h.mean() - 1.96 * h.std(ddof=1) / np.sqrt(len(h))) > MARGIN
    result["resplit_non_inferior_share"] = ok / 200
    lows = [result["halves"][h][PIPELINE]["ci"][0] for h in ("dev", "tune")]
    result["non_inferior"] = bool(all(x > MARGIN for x in lows) and ok / 200 >= 0.95)

    ext_gold = {q: next(iter(v)) for q, v in
                json.loads((args.artifacts / "ext_cc" / "qrels.json").read_text()).items()}
    s = load(args.outcomes / f"{args.small}_extcc_dev_deep50.jsonl.gz")
    l = load(args.outcomes / f"{args.large}_extcc_dev_deep50.jsonl.gz")  # noqa: E741
    qids = sorted(set(s) & set(l))
    ext = {"statements": len(qids), "examples_differ": examples_differ(s, l),
           "recall@50": {"small": recall([s[q] for q in qids], ext_gold, 50),
                         "large": recall([l[q] for q in qids], ext_gold, 50)}}
    for name in ("dense", PIPELINE):
        ext[name], _ = compare([s[q] for q in qids], [l[q] for q in qids], ext_gold, name)
    ext["veto"] = ext[PIPELINE]["ci"][1] < 0  # small significantly worse on EXT
    result["ext_dev"] = ext

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
