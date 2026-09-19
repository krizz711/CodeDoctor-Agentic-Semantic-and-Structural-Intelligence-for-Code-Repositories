"""Live retrieval demo: one query in, a ranked, explained answer out.

Shows what each pass did and what it cost, which is what the hands-on
evaluation asks for — responses for a query, and how fast.

    python scripts/demo_query.py --query "reverse a linked list in place"
    python scripts/demo_query.py --interactive
    python scripts/demo_query.py --query-id q5001        # replay a dataset query

The corpus embeddings come from the cached artifacts, so only the query is
encoded at demo time; that is exactly how the deployed service behaves, and it
is why a query answers in milliseconds on CPU rather than minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    apply_execution_rerank,
    parse_worked_examples,
)
from app.pipeline.features import (  # noqa: E402
    FEATURE_NAMES,
    CorpusStatistics,
    FeatureExtractor,
)
from app.pipeline.query_analysis import analyze_query  # noqa: E402
from app.pipeline.reranker import RerankerConfig  # noqa: E402
from app.retrieval.encode import CodeEncoder  # noqa: E402

BAR = "=" * 78


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
    return ids, texts


def snippet(text: str, lines: int = 6, width: int = 72) -> str:
    out = []
    for raw in text.splitlines()[:lines]:
        out.append(textwrap.shorten(raw, width=width, placeholder=" ...") if raw.strip() else "")
    if text.count("\n") + 1 > lines:
        out.append("    ...")
    return "\n".join("    " + line for line in out)


class DemoEngine:
    def __init__(self, artifacts: Path, model_tag: str, tier: str, weights: Path | None) -> None:
        print("Loading index ...", flush=True)
        t0 = time.time()
        self.corpus_ids, self.corpus_texts = load_jsonl(artifacts / "texts_corpus.jsonl")
        self.corpus_emb = np.load(artifacts / f"emb_{model_tag}_corpus.npy").astype(np.float32)
        self.config = RerankerConfig.load(weights)
        self.stats = CorpusStatistics.build(self.corpus_texts)
        hub_path = artifacts / f"hubness_{model_tag}.npy"
        if hub_path.exists():
            self.stats.hubness = np.load(hub_path).astype(np.float32)
        self.extractor = FeatureExtractor(
            self.stats,
            self.config.family_compat,
            length_prior_slope=self.config.length_prior_slope,
            length_prior_intercept=self.config.length_prior_intercept,
        )
        self.weights = np.asarray(self.config.weights, dtype=np.float32)
        print(f"  {len(self.corpus_ids)} snippets indexed in {time.time() - t0:.1f}s")

        t0 = time.time()
        self.encoder = CodeEncoder(device="cpu", tier=tier)
        _ = self.encoder.model  # force the weight load out of the query path
        print(f"  encoder {self.encoder.model_name} loaded in {time.time() - t0:.1f}s (CPU)")
        print(f"  reranker: {self.config.describe()}")

    def answer(self, query: str, *, top_k: int = 5, execution: bool = False) -> None:
        timings: dict[str, float] = {}

        t0 = time.time()
        facts = analyze_query(query)
        timings["classify"] = time.time() - t0

        t0 = time.time()
        query_vector = self.encoder.encode(query, mode="query")
        timings["encode"] = time.time() - t0

        t0 = time.time()
        scores = self.corpus_emb @ query_vector
        depth = min(self.config.candidate_depth, len(scores))
        pool = np.argpartition(-scores, depth - 1)[:depth]
        pool = pool[np.argsort(-scores[pool], kind="stable")]
        timings["search"] = time.time() - t0
        dense_order = list(pool)

        t0 = time.time()
        if not self.config.is_identity:
            features = self.extractor.for_query(facts, pool, scores[pool])
            reranked = features @ self.weights
            order = pool[np.argsort(-reranked, kind="stable")]
        else:
            features = None
            order = pool
        timings["rerank"] = time.time() - t0

        audit = None
        if execution:
            t0 = time.time()
            candidate_texts = [self.corpus_texts[i] for i in order[:top_k]]
            base = list(range(top_k, 0, -1))
            adjusted, audit = apply_execution_rerank(
                query,
                candidate_texts,
                [float(v) for v in base],
                ExecutionConfig(enabled=True, max_candidates=top_k, verified_bonus=100.0),
            )
            promoted = [order[:top_k][i] for i in np.argsort(-np.asarray(adjusted), kind="stable")]
            order = np.array(promoted + list(order[top_k:]))
            timings["execute"] = time.time() - t0

        print()
        print(BAR)
        print("QUERY")
        print(BAR)
        print(textwrap.fill(query.strip()[:600], width=78, initial_indent="  ", subsequent_indent="  "))
        print()
        print(f"  pass 1  query family : {facts.family.value} "
              f"(confidence {facts.confidence:.2f}, evidence {facts.family_scores})")
        if facts.identifiers:
            named = sorted(facts.identifiers)[:8]
            print(f"          names in query : {', '.join(named)}")
        if audit is not None:
            print(f"  pass 3  worked examples: {audit['examples_found']}, "
                  f"executed {audit['candidates_executed']}, "
                  f"verified {audit['candidates_verified']}")

        print()
        print(f"RESULTS (top {top_k} of {len(self.corpus_ids)})")
        print("-" * 78)
        for rank, index in enumerate(order[:top_k], start=1):
            moved = ""
            if index in dense_order:
                dense_rank = dense_order.index(index) + 1
                if dense_rank != rank:
                    moved = f"  [dense #{dense_rank} -> #{rank}]"
            print(f"\n#{rank}  {self.corpus_ids[index]}   cosine={scores[index]:.4f}"
                  f"   shape={self.stats.facts[index].family.value}{moved}")
            print(snippet(self.corpus_texts[index]))
            if features is not None:
                row = np.flatnonzero(pool == index)
                if row.size:
                    contributions = features[row[0]] * self.weights
                    parts = [
                        f"{name}={value:+.4f}"
                        for name, value in zip(FEATURE_NAMES, contributions)
                        if name != "dense" and abs(value) > 1e-4
                    ]
                    if parts:
                        print(f"    why: {'  '.join(parts)}")

        total = sum(timings.values())
        print()
        print("-" * 78)
        print("LATENCY (CPU)  " + "   ".join(f"{k}={v * 1000:.1f}ms" for k, v in timings.items()))
        print(f"               total={total * 1000:.1f}ms")
        print(BAR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm06b", help="embedding tag inside the artifacts dir")
    parser.add_argument("--tier", choices=["accuracy", "interactive"], default="interactive")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--query-id", default=None, help="replay a dataset query by id, e.g. q5001")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--execution", action="store_true", help="enable execution verification")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    engine = DemoEngine(args.artifacts, args.model, args.tier, args.weights)

    if args.query_id:
        for name in ("texts_queries_test.jsonl", "texts_queries_train.jsonl"):
            ids, texts = load_jsonl(args.artifacts / name)
            if args.query_id in ids:
                engine.answer(texts[ids.index(args.query_id)], top_k=args.top_k, execution=args.execution)
                return 0
        print(f"Query id {args.query_id} not found.")
        return 1

    if args.query:
        engine.answer(args.query, top_k=args.top_k, execution=args.execution)
        return 0

    if args.interactive:
        print("\nType a query (blank line to quit).")
        while True:
            try:
                query = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not query:
                return 0
            engine.answer(query, top_k=args.top_k, execution=args.execution)

    parser.error("pass --query, --query-id, or --interactive")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
