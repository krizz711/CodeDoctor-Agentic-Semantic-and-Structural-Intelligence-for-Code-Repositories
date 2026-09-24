"""PRE-5: document-side hubness correction, fitted inductively.

    python scripts/hubness_eval.py --model f2llm06b

Some documents sit close to *every* query (short print-only scripts, generic
templates) and crowd rankings.  CSLS / QB-Norm style corrections subtract, per
document, how "hub-like" it is -- its mean similarity to its k nearest queries
in a reference bank.  The penalty is one vector computed offline, so the cost
at query time is a subtraction.

Discipline:

* the bank is the TUNE half of an all-family development split of the APPS
  *train* queries -- never DEV, never test, never the query batch being scored
  (test-batch transduction would exploit benchmark structure);
* beta is chosen on TUNE (scored against a bank built from the *other* TUNE
  half, so the fit is not in-sample), then reported once on DEV with a paired
  bootstrap against beta = 0.

Dense pass only: this is a first-pass correction; how it interacts with
execution verification is a separate experiment.
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

from app.pipeline.splits import make_development_split  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402
from scripts.replay_outcomes import paired_bootstrap  # noqa: E402


def hub_scores(documents: np.ndarray, bank: np.ndarray, k: int, block: int = 1024) -> np.ndarray:
    """Mean of each document's top-k cosine similarities to the bank queries."""
    out = np.empty(documents.shape[0], dtype=np.float32)
    for start in range(0, documents.shape[0], block):
        sims = documents[start : start + block] @ bank.T
        kk = min(k, sims.shape[1])
        top = np.partition(sims, -kk, axis=1)[:, -kk:]
        out[start : start + block] = top.mean(axis=1)
    return out


def per_query_ndcg(queries: np.ndarray, documents: np.ndarray, gold_cols: np.ndarray,
                   penalty: np.ndarray | None, beta: float) -> np.ndarray:
    scores = queries @ documents.T
    if penalty is not None and beta:
        scores = scores - beta * penalty[None, :]
    gold_scores = scores[np.arange(len(gold_cols)), gold_cols]
    rank = (scores > gold_scores[:, None]).sum(axis=1) + 1
    return np.where(rank <= 10, 1.0 / np.log2(rank + 1), 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm06b")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    art = args.artifacts
    corpus_ids, _ = load_jsonl_texts(art / "texts_corpus.jsonl")
    documents = np.load(art / f"emb_{args.model}_corpus.npy").astype(np.float32)
    query_ids, query_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
    queries = np.load(art / f"emb_{args.model}_qtrain.npy").astype(np.float32)
    qrels = json.loads((art / "qrels_train.json").read_text(encoding="utf-8"))
    column = {d: i for i, d in enumerate(corpus_ids)}
    gold_cols = np.array([column[next(iter(qrels[q]))] for q in query_ids])

    split = make_development_split(query_texts, match_family=None, seed=args.seed)
    tune, dev = split.tune, split.dev
    print(split.describe())

    # Fit beta on TUNE without scoring TUNE queries against a bank containing
    # themselves: two halves, each scored with the other's bank.
    rng = np.random.default_rng(args.seed)
    halves = np.array_split(rng.permutation(tune), 2)
    penalties = [hub_scores(documents, queries[h], args.k) for h in halves]
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    tune_curve = {}
    for beta in grid:
        values = np.concatenate([
            per_query_ndcg(queries[halves[0]], documents, gold_cols[halves[0]], penalties[1], beta),
            per_query_ndcg(queries[halves[1]], documents, gold_cols[halves[1]], penalties[0], beta),
        ])
        tune_curve[beta] = float(values.mean())
    best = max(grid, key=lambda b: (tune_curve[b], -b))
    print("TUNE NDCG@10 by beta:", {b: round(v, 5) for b, v in tune_curve.items()}, " chosen:", best)

    # Report once on DEV, with the penalty from the whole TUNE bank.
    penalty = hub_scores(documents, queries[tune], args.k)
    base = per_query_ndcg(queries[dev], documents, gold_cols[dev], None, 0.0)
    corrected = per_query_ndcg(queries[dev], documents, gold_cols[dev], penalty, best)
    delta, low, high, p = paired_bootstrap(base, corrected)
    print(f"DEV NDCG@10  dense {base.mean():.5f}  hubness(beta={best}) {corrected.mean():.5f}  "
          f"delta {delta:+.5f} [{low:+.5f}, {high:+.5f}]  p={p:.3f}  (n={len(dev)})")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "model": args.model, "k": args.k, "tune_curve": tune_curve, "beta": best,
            "dev_dense": float(base.mean()), "dev_hubness": float(corrected.mean()),
            "delta": delta, "ci": [low, high], "p": p, "n_dev": int(len(dev)),
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
