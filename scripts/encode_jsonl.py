"""Encode any ``texts_*.jsonl`` file with a frozen tier, e.g. an external set.

    python scripts/encode_jsonl.py artifacts/ext_cc/texts_queries.jsonl --mode query \
        --out artifacts/ext_cc/emb_f2llm06b_queries.npy --device cuda

Offline pre-processing (the FAQ allows a GPU here).  Same recipe as every
other encode: the tier decides the model, never the prompt or truncation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.encode import CodeEncoder, _select_device, select_dtype  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", type=Path)
    parser.add_argument("--mode", choices=["query", "document"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tier", choices=["accuracy", "interactive"], default="interactive")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--attention-budget", type=int, default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--reuse", type=Path, nargs=2, metavar=("TEXTS", "EMB"), default=None,
                        help="reuse rows of an existing (texts.jsonl, emb.npy) pair for identical texts")
    args = parser.parse_args(argv)

    _, texts = load_jsonl_texts(args.texts)
    device = _select_device(args.device)
    encoder = CodeEncoder(device=device, tier=args.tier, torch_dtype=select_dtype(device))
    if args.attention_budget:
        encoder.attention_budget = int(args.attention_budget)
    if args.max_batch:
        encoder.max_batch = int(args.max_batch)

    out = np.zeros((len(texts), 0), dtype=np.float32)
    known: dict[str, np.ndarray] = {}
    if args.reuse:
        _, old_texts = load_jsonl_texts(args.reuse[0])
        old_emb = np.load(args.reuse[1]).astype(np.float32)
        known = {t: old_emb[i] for i, t in enumerate(old_texts)}
    todo = [i for i, t in enumerate(texts) if t not in known]
    print(f"{len(texts)} texts, {len(texts) - len(todo)} reused, encoding {len(todo)} "
          f"with {encoder.model_name} on {device}")
    started = time.time()
    fresh = encoder.encode([texts[i] for i in todo], mode=args.mode) if todo else None
    dim = fresh.shape[1] if fresh is not None else next(iter(known.values())).shape[0]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, i in enumerate(todo):
        out[i] = fresh[row]
    for i, t in enumerate(texts):
        if t in known:
            out[i] = known[t]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out.astype(np.float16))
    print(f"wrote {args.out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
