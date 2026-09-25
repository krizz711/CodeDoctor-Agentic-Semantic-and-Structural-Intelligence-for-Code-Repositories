"""Methodological stress test of the round-1/R1 conclusions (read-only).

    python scripts/stress_analysis.py --out experiments/stress_test_result.json

Everything here reads recorded outcome stores and result files; nothing is
executed or re-encoded.  Findings and their interpretation are in
``experiments/STRESS_TEST.md``.

* S1  bootstrap stability: interval bounds over 5 seeds x 20,000 resamples, a
      paired sign-flip permutation p, and how many statements actually change.
* S2  gate mechanism: how often the dense gate (gap > 0.20) keeps the gold out
      of execution, APPS vs CodeContests, Fisher exact test.
* S3  CodeContests EXT-dev split by source split (the valid split can be in
      F2LLM-v2's training data via OpenCodeReasoning-2; the test split is not).
* S6  near-duplicate sensitivity: max Jaccard of EXT-dev statements vs APPS.
* S7  parser precision per source (V3-oracle, CONF half).
* S12 split sensitivity: 200 random halves of the 5,000 train statements.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_external_set import _shingles  # noqa: E402
from scripts.replay_outcomes import POLICIES, metrics, rerank  # noqa: E402

POLICY = {p.name: p for p in POLICIES}
COMPARISONS = [
    ("call-style", "legacy+numeric1e-6"),
    ("call-style+deep50+gate0.2", "call-style"),
    ("call-style+gate0.2", "call-style"),
    ("call-style+deep50", "call-style"),
]
GAP = 0.2


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        handle.readline()
        return [json.loads(line) for line in handle]


def ndcg(entries: list[dict], gold: dict[str, str], name: str) -> np.ndarray:
    if name == "dense":
        return np.array([metrics(e["pool"], gold[e["qid"]])[0] for e in entries])
    return np.array([metrics(rerank(e, POLICY[name]), gold[e["qid"]])[0] for e in entries])


def bootstrap_bounds(d: np.ndarray, seed: int, resamples: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(resamples)])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_p(d: np.ndarray, resamples: int = 20000, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(resamples, len(d)))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean())).mean())


def gated_out(entries: list[dict], gold: dict[str, str]) -> tuple[int, int]:
    """(golds in the top 20 that the gate keeps out of execution, golds in the top 20)."""
    out = total = 0
    for e in entries:
        if not e["candidates"]:
            continue
        g = gold[e["qid"]]
        if g in e["pool"][:20]:
            total += 1
            out += e["scores"][0] - e["scores"][e["pool"].index(g)] > GAP
    return out, total


def main(argv: list[str] | None = None) -> int:
    from scipy.stats import fisher_exact

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=REPO_ROOT / "data" / "outcomes")
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "stress_test_result.json")
    args = parser.parse_args(argv)
    result: dict = {}

    gold_apps = {q: next(iter(v)) for q, v in
                 json.loads((args.artifacts / "apps" / "qrels_train.json").read_text()).items()}
    gold_ext = {q: next(iter(v)) for q, v in
                json.loads((args.artifacts / "ext_cc" / "qrels.json").read_text()).items()}
    names = {n for pair in COMPARISONS for n in pair}

    vectors = {}
    result["S1_bootstrap_stability_f2llm17b"] = {}
    for split in ("dev", "tune"):
        entries = load(args.outcomes / f"f2llm17b_all_{split}_deep50.jsonl.gz")
        vectors[split] = {n: ndcg(entries, gold_apps, n) for n in names}
        for a, b in COMPARISONS:
            d = vectors[split][a] - vectors[split][b]
            bounds = np.array([bootstrap_bounds(d, s, 20000) for s in range(5)])
            result["S1_bootstrap_stability_f2llm17b"][f"{split}: {a} - {b}"] = {
                "mean": round(float(d.mean()), 5),
                "lower_range": [round(float(bounds[:, 0].min()), 5), round(float(bounds[:, 0].max()), 5)],
                "upper_range": [round(float(bounds[:, 1].min()), 5), round(float(bounds[:, 1].max()), 5)],
                "sign_flip_p": round(sign_flip_p(d), 4),
                "statements_changed": int((d != 0).sum()),
            }

    result["S12_split_sensitivity_f2llm17b"] = {}
    pooled = {n: np.concatenate([vectors["dev"][n], vectors["tune"][n]]) for n in names}
    rng = np.random.default_rng(20260924)
    for a, b in COMPARISONS:
        d = pooled[a] - pooled[b]
        half = len(d) // 2
        stats = []
        for _ in range(200):
            h = d[rng.permutation(len(d))[:half]]
            stats.append((h.mean(), h.mean() - 1.96 * h.std(ddof=1) / np.sqrt(half)))
        s = np.array(stats)
        result["S12_split_sensitivity_f2llm17b"][f"{a} - {b}"] = {
            "pooled_mean": round(float(d.mean()), 5),
            "half_mean_range": [round(float(s[:, 0].min()), 5), round(float(s[:, 0].max()), 5)],
            "share_mean_at_least_0.003": round(float((s[:, 0] >= 0.003).mean()), 3),
            "share_lower_above_-0.002": round(float((s[:, 1] > -0.002).mean()), 3),
            "share_lower_above_0": round(float((s[:, 1] > 0).mean()), 3),
            "method": "normal approximation per half",
        }

    result["S2_gate_gold_exclusion"] = {}
    for model in ("f2llm06b", "f2llm17b"):
        apps = gated_out(load(args.outcomes / f"{model}_all_dev.jsonl.gz"), gold_apps)
        ext = gated_out(load(args.outcomes / f"{model}_extcc_dev.jsonl.gz"), gold_ext)
        _, p = fisher_exact([[ext[0], ext[1] - ext[0]], [apps[0], apps[1] - apps[0]]])
        result["S2_gate_gold_exclusion"][model] = {"apps_dev": apps, "ext_dev": ext, "fisher_p": float(p)}

    result["S3_ext_by_source_split"] = {}
    for model in ("f2llm06b", "f2llm17b"):
        entries = load(args.outcomes / f"{model}_extcc_dev_deep50.jsonl.gz")
        for sub in ("valid", "test"):
            part = [e for e in entries if e["qid"].startswith(f"cc-{sub}-")]
            dense = ndcg(part, gold_ext, "dense")
            exe = ndcg(part, gold_ext, "call-style")
            gate = ndcg(part, gold_ext, "call-style+gate0.2")
            result["S3_ext_by_source_split"][f"{model} {sub}"] = {
                "n": len(part), "dense": round(float(dense.mean()), 4), "execution": round(float(exe.mean()), 4),
                "gate_minus_execution": round(float((gate - exe).mean()), 4),
                "statements_changed_by_gate": int((gate != exe).sum()),
            }

    apps_statements = []
    for name in ("texts_queries_train.jsonl", "texts_queries_test.jsonl"):
        with (args.artifacts / "apps" / name).open(encoding="utf-8") as handle:
            apps_statements += [json.loads(line)["text"] for line in handle]
    index: dict[int, list[int]] = defaultdict(list)
    sizes = []
    for k, text in enumerate(apps_statements):
        shingles = _shingles(text)
        sizes.append(len(shingles))
        for s in shingles:
            index[s].append(k)
    dev = set(json.loads((args.artifacts / "ext_cc" / "splits.json").read_text())["ext_dev"])
    buckets: Counter = Counter()
    with (args.artifacts / "ext_cc" / "texts_queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            q = json.loads(line)
            if q["id"] not in dev:
                continue
            mine = _shingles(q["text"])
            overlap: Counter = Counter()
            for s in mine:
                for k in index.get(s, ()):
                    overlap[k] += 1
            best = max((c / (len(mine) + sizes[k] - c) for k, c in overlap.items()), default=0.0)
            buckets["<0.1" if best < 0.1 else "0.1-0.3" if best < 0.3 else "0.3-0.5" if best < 0.5 else ">=0.5"] += 1
    result["S6_ext_dev_max_jaccard_vs_apps"] = dict(buckets)

    oracle = json.loads((REPO_ROOT / "experiments" / "V3_parser_gold" / "result_oracle.json").read_text(encoding="utf-8"))
    groups = oracle["configs"]["primary"]["O2_codecontests_train"]["CONF"]["by_group"]
    per_source, num, den = {}, 0, 0
    for g, v in groups.items():
        p = v["stdin"]["precision"]
        per_source[g] = {"precision": p["value"], "extracted": p["den"],
                         "statements_with_visible_tests": v["stdin"]["statements_with_visible_tests"]}
        if g != "hackerearth":
            num += p["num"]
            den += p["den"]
    result["S7_parser_precision_by_source_conf"] = {
        "per_source": per_source,
        "without_hackerearth": round(num / den, 4) if den else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
