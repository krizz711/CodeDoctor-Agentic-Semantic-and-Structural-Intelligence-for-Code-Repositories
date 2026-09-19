"""Why does the gold fail to reproduce its own worked examples?

The promotion-gate experiment established that execution's weakness on this
corpus is recall, not precision: of the golds sitting in the executed
candidate window, a large minority never verify, and that population is far
bigger than the one false promotion costs.

"Fails to verify" is not a cause, though.  A gold can miss because the
statement's examples were parsed wrong, because the reference solution is slow
and hits the timeout, because it crashes under ``-I -S``, or because its output
genuinely differs from the sample.  Those want completely different fixes, and
only one of them is a retrieval problem at all.

This re-runs exactly the golds that failed, records the reason each one gave,
and groups them.

    python scripts/diagnose_verification_misses.py --dump data/arms/instrumented.npz
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    parse_worked_examples,
    run_candidate,
)
from app.pipeline.reranker import RerankerConfig  # noqa: E402
from app.pipeline.search import build_candidate_pool  # noqa: E402
from app.pipeline.splits import TEST_DOMINANT_FAMILY, make_development_split  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402


def crash_reason(code: str, example, config: ExecutionConfig) -> str:
    """Re-run a crashing snippet to recover the exception type.

    ``run_candidate`` reports only the exit status, which is the right amount
    of detail for a scoring pass and not enough to tell an ImportError caused
    by ``-S`` apart from a genuine bug in the reference solution.
    """
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="codedoctor-diag-") as workdir:
        script = os.path.join(workdir, "candidate.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(code)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": workdir,
            "TMPDIR": workdir,
        }
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-S", script],
                input=example.stdin, text=True, capture_output=True,
                timeout=config.timeout_seconds, cwd=workdir, env=env,
            )
        except subprocess.TimeoutExpired:
            return "timeout"
        except (OSError, ValueError) as error:
            return f"spawn failed: {type(error).__name__}"

    lines = [line for line in (done.stderr or "").strip().splitlines() if line.strip()]
    if not lines:
        return "no stderr"
    last = lines[-1]
    return last.split(":")[0].strip() or last[:40]


def classify(details: list[str]) -> str:
    """One label per gold, from the reasons its examples gave."""
    if any(d == "timeout" for d in details):
        return "timeout"
    if any("max_code_chars" in d for d in details):
        return "refused: too large"
    if any("exit" in d for d in details):
        return "crashed"
    if all(d == "output mismatch" for d in details):
        return "output mismatch"
    return "other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=REPO_ROOT / "data" / "arms" / "instrumented.npz")
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm17b")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--timeout", type=float, default=None,
                        help="re-run misses at a different timeout to size that cause")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    d = np.load(args.dump)
    verified, gold_columns, fired = d["verified"], d["gold_columns"], d["fired"]
    top_n, depth = int(d["top_n"]), int(d["depth"])

    art = args.artifacts
    corpus_ids, corpus_texts = load_jsonl_texts(art / "texts_corpus.jsonl")
    corpus_emb = np.load(art / f"emb_{args.model}_corpus.npy").astype(np.float32)
    query_ids, query_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
    query_emb = np.load(art / f"emb_{args.model}_qtrain.npy").astype(np.float32)

    split = make_development_split(query_texts, match_family=TEST_DOMINANT_FAMILY, seed=args.seed)
    rows = split.dev
    query_texts = [query_texts[i] for i in rows]
    query_ids = [query_ids[i] for i in rows]
    query_emb = query_emb[rows]

    pool = build_candidate_pool(query_emb, corpus_emb, depth=max(depth, 100))
    indices = pool.indices[:, :depth]

    base = ExecutionConfig(enabled=True, max_candidates=top_n)
    config = base if args.timeout is None else __import__("dataclasses").replace(
        base, timeout_seconds=args.timeout
    )

    misses = [
        r for r in range(len(query_ids))
        if fired[r] and 0 <= gold_columns[r] < top_n and verified[r, gold_columns[r]] != 1
    ]
    hits = int(sum(
        1 for r in range(len(query_ids))
        if fired[r] and 0 <= gold_columns[r] < top_n and verified[r, gold_columns[r]] == 1
    ))
    print(f"golds in the executed window : {len(misses) + hits}")
    print(f"  verified                   : {hits}")
    print(f"  failed                     : {len(misses)}")
    print(f"timeout for this pass        : {config.timeout_seconds}s")
    print()

    buckets: collections.Counter = collections.Counter()
    crashes: collections.Counter = collections.Counter()
    now_passes = 0
    examples_hist: collections.Counter = collections.Counter()
    records = []
    for r in misses:
        examples = parse_worked_examples(query_texts[r], limit=config.max_examples)
        examples_hist[len(examples)] += 1
        gold_text = corpus_texts[indices[r][gold_columns[r]]]
        details = []
        for ex in examples:
            ok, detail = run_candidate(gold_text, ex, config)
            details.append("match" if ok else detail)
        label = "verifies now" if details and all(d == "match" for d in details) else classify(details)
        if label == "verifies now":
            now_passes += 1
        buckets[label] += 1
        reason = None
        if label == "crashed" and examples:
            reason = crash_reason(gold_text, examples[0], config)
            crashes[reason] += 1
        records.append(
            {"query": query_ids[r], "label": label, "details": details, "exception": reason}
        )

    print(f"{'reason the gold failed':<24} {'n':>5} {'share':>8}")
    print("-" * 40)
    for label, n in buckets.most_common():
        print(f"{label:<24} {n:>5} {n / max(1, len(misses)):>8.1%}")
    if crashes:
        print()
        print(f"{'exception raised by the crashing golds':<38} {'n':>5}")
        print("-" * 46)
        for name, n in crashes.most_common():
            print(f"{name:<38} {n:>5}")
    print()
    print(f"examples parsed per failing query: "
          f"{dict(sorted(examples_hist.items()))}")
    if args.timeout is not None:
        print(f"\nnewly passing at {config.timeout_seconds}s: {now_passes} "
              f"({now_passes / max(1, len(misses)):.1%} of the misses)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"timeout": config.timeout_seconds, "misses": len(misses),
                 "hits": hits, "buckets": dict(buckets),
                 "crash_exceptions": dict(crashes), "records": records},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
