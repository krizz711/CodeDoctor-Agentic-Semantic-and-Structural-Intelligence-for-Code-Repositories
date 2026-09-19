"""Two candidate improvements to the *first* pass, measured before adoption.

Both are cheap, both are standard, and neither is assumed to work:

* **Dual-encoder fusion** — score-average or RRF over the accuracy and
  interactive tiers. Two encoders disagree on different queries, so fusing
  them usually beats either. It also doubles inference cost, which has to be
  worth paying.
* **Pseudo-relevance feedback (Rocchio)** — move the query vector toward the
  centroid of its own top-k documents and search again. One extra matrix
  product; no extra model call.

    python scripts/explore_first_pass.py --models f2llm17b f2llm06b

Everything is measured on the held-out DEV half of the distribution-matched
train split, with a paired bootstrap against the single-encoder baseline.
Nothing here is wired into the pipeline unless it clears that gate.
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

from app.pipeline.metrics import metrics_from_ranks, paired_bootstrap, per_query_ndcg  # noqa: E402
from app.pipeline.splits import TEST_DOMINANT_FAMILY, make_development_split  # noqa: E402


def load_jsonl_texts(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
    return ids, texts


def ranks_from_scores(scores: np.ndarray, gold: np.ndarray, *, depth: int = 100) -> np.ndarray:
    """1-based gold rank per query, capped at ``depth + 1``."""
    n_queries = scores.shape[0]
    ranks = np.full(n_queries, depth + 1, dtype=np.int32)
    k = min(depth, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    for row in range(n_queries):
        candidates = part[row]
        order = candidates[np.argsort(-scores[row, candidates], kind="stable")]
        hit = np.flatnonzero(order == gold[row])
        if hit.size:
            ranks[row] = int(hit[0]) + 1
    return ranks


def rrf_scores(score_matrices: list[np.ndarray], *, k: int = 60, depth: int = 200) -> np.ndarray:
    """Reciprocal rank fusion over several full score matrices."""
    n_queries, n_documents = score_matrices[0].shape
    fused = np.zeros((n_queries, n_documents), dtype=np.float32)
    for scores in score_matrices:
        part = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]
        for row in range(n_queries):
            candidates = part[row]
            order = candidates[np.argsort(-scores[row, candidates], kind="stable")]
            fused[row, order] += 1.0 / (k + np.arange(1, len(order) + 1))
    return fused


def rocchio(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    scores: np.ndarray,
    *,
    alpha: float,
    feedback_k: int,
) -> np.ndarray:
    """One round of pseudo-relevance feedback, then rescore."""
    top = np.argpartition(-scores, feedback_k - 1, axis=1)[:, :feedback_k]
    centroids = document_embeddings[top].mean(axis=1)
    expanded = query_embeddings + alpha * centroids
    norms = np.linalg.norm(expanded, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (expanded / norms) @ document_embeddings.T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--models", nargs="+", default=["f2llm17b", "f2llm06b"])
    parser.add_argument("--split", choices=["dev", "tune"], default="dev")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    art = args.artifacts
    corpus_ids, _ = load_jsonl_texts(art / "texts_corpus.jsonl")
    query_ids, query_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
    qrels = json.loads((art / "qrels_train.json").read_text(encoding="utf-8"))

    split = make_development_split(query_texts, match_family=TEST_DOMINANT_FAMILY, seed=args.seed)
    rows = split.dev if args.split == "dev" else split.tune
    print(f"{split.describe()}  ->  evaluating {args.split} ({len(rows)} queries)")

    doc_index_of = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    gold = np.array(
        [doc_index_of[next(iter(qrels[query_ids[i]]))] for i in rows], dtype=np.int32
    )

    score_matrices: dict[str, np.ndarray] = {}
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in args.models:
        corpus_path = art / f"emb_{tag}_corpus.npy"
        query_path = art / f"emb_{tag}_qtrain.npy"
        if not corpus_path.exists() or not query_path.exists():
            print(f"  skip {tag}: embeddings not found")
            continue
        corpus = np.load(corpus_path).astype(np.float32)
        queries = np.load(query_path).astype(np.float32)[rows]
        embeddings[tag] = (queries, corpus)
        score_matrices[tag] = queries @ corpus.T
        print(f"  loaded {tag}: corpus {corpus.shape}, queries {queries.shape}")

    if not score_matrices:
        print("No embeddings found. Run scripts/encode_apps.py first.")
        return 1

    results: dict[str, dict] = {}
    baseline_tag = args.models[0] if args.models[0] in score_matrices else next(iter(score_matrices))
    baseline_ranks = ranks_from_scores(score_matrices[baseline_tag], gold)
    baseline_ndcg = per_query_ndcg(baseline_ranks)

    print()
    print("=" * 78)
    print("FIRST-PASS VARIANTS")
    print("=" * 78)

    def evaluate(label: str, scores: np.ndarray) -> None:
        ranks = ranks_from_scores(scores, gold)
        metrics = metrics_from_ranks(ranks)
        boot = paired_bootstrap(baseline_ndcg, per_query_ndcg(ranks))
        marker = "  <-- " if boot.significant and boot.mean_delta > 0 else "      "
        print(f"{marker}{label:<34s} NDCG@10={metrics.ndcg_at_10:.5f} "
              f"MRR@10={metrics.mrr_at_10:.5f}  {boot.describe()}")
        results[label] = {"metrics": metrics.as_dict(), "bootstrap": boot.__dict__}

    for tag, scores in score_matrices.items():
        evaluate(f"single: {tag}", scores)

    if len(score_matrices) >= 2:
        print("-" * 78)
        tags = list(score_matrices)
        a, b = score_matrices[tags[0]], score_matrices[tags[1]]
        for weight in (0.3, 0.5, 0.7):
            evaluate(f"fuse {tags[0]}*{weight:.1f}+{tags[1]}*{1 - weight:.1f}",
                     weight * a + (1 - weight) * b)
        evaluate("fuse RRF(k=60)", rrf_scores([a, b]))

    print("-" * 78)
    queries, corpus = embeddings[baseline_tag]
    for alpha in (0.2, 0.4, 0.6):
        for feedback_k in (3, 5, 10):
            evaluate(
                f"PRF rocchio a={alpha} k={feedback_k}",
                rocchio(queries, corpus, score_matrices[baseline_tag],
                        alpha=alpha, feedback_k=feedback_k),
            )

    print("=" * 78)
    print("A variant is adopted only where the bootstrap marks it significant")
    print("AND the extra inference cost is worth paying.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
