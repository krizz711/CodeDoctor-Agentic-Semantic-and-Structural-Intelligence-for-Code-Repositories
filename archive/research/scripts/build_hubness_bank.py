"""Encode a hubness bank (PRE-5): query embeddings of statements *not* answered by the corpus.

    # for scoring APPS: CodeContests statements (their answers are not APPS documents)
    python scripts/build_hubness_bank.py --texts artifacts/ext_cc/texts_queries.jsonl \
        --tier accuracy --out artifacts/hubness/bank_cc_f2llm17b.npy --device cuda
    # for an unseen corpus: APPS train statements (their answers are not in it)
    python scripts/build_hubness_bank.py --texts artifacts/apps/texts_queries_train.jsonl \
        --tier accuracy --out artifacts/hubness/bank_appstrain_f2llm17b.npy --device cuda

The bank must come from the *same* encoder and prompt as the run that uses it
(a 0.6B bank is meaningless against 1.7B documents).  Offline pre-processing:
the FAQ allows a GPU here.  At query time the correction is one subtraction.

Why the answers must not be in the corpus: a document that answers a bank
query is, by construction, close to that query, so it gets a large penalty --
and in a one-answer-per-query benchmark that is a *wrong* document for every
other query.  On APPS DEV about 0.0025 of a +0.0121 gain came from exactly
that structure (experiments/E6_hubness/result.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.encode import CodeEncoder, _select_device, select_dtype  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tier", choices=["accuracy", "interactive"], default="accuracy")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    _, texts = load_jsonl_texts(args.texts)
    device = _select_device(args.device)
    encoder = CodeEncoder(device=device, tier=args.tier, torch_dtype=select_dtype(device))
    bank = encoder.encode(texts, mode="query", show_progress_bar=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, bank.astype(np.float32))
    print(f"bank: {len(texts)} statements from {args.texts} with {encoder.model_name} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
