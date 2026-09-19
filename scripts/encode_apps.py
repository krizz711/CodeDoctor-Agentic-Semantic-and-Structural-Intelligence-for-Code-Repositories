"""Dump CoIR APPS texts and embeddings to ``artifacts/apps`` for offline work.

Tuning the reranker needs the dense vectors hundreds of times; encoding them
once and reusing the cache turns a multi-hour loop into a sub-second one.

    python scripts/encode_apps.py --tier interactive --device cpu
    python scripts/encode_apps.py --tier accuracy --device cuda --tag f2llm17b

Writes, for ``--tag TAG``::

    texts_corpus.jsonl  texts_queries_train.jsonl  texts_queries_test.jsonl
    ids_*.json  qrels_train.json  qrels_test.json
    emb_TAG_corpus.npy  emb_TAG_qtrain.npy  emb_TAG_qtest.npy   (float16)
    manifest.json

The text and qrel files are split-agnostic; only the ``emb_*`` files depend on
the encoder, so a second model is a cheap incremental run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.encode import CodeEncoder, _select_device, select_dtype  # noqa: E402

DEFAULT_TAGS = {"accuracy": "f2llm17b", "interactive": "f2llm06b"}


def dump_texts(out: Path) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    from datasets import load_dataset

    print("Loading CoIR-Retrieval/apps ...")
    corpus_ds = load_dataset("CoIR-Retrieval/apps", "corpus")["corpus"]
    queries_ds = load_dataset("CoIR-Retrieval/apps", "queries")["queries"]
    qrels = load_dataset("CoIR-Retrieval/apps", "default")

    corpus_ids = list(corpus_ds["_id"])
    corpus_texts = list(corpus_ds["text"])
    query_text_of = {row["_id"]: row["text"] for row in queries_ds}

    qrels_train = {r["query-id"]: {r["corpus-id"]: int(r["score"])} for r in qrels["train"]}
    qrels_test = {r["query-id"]: {r["corpus-id"]: int(r["score"])} for r in qrels["test"]}

    def sort_ids(ids):
        return sorted(ids, key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)

    train_ids = sort_ids(qrels_train)
    test_ids = sort_ids(qrels_test)
    train_texts = [query_text_of[i] for i in train_ids]
    test_texts = [query_text_of[i] for i in test_ids]

    out.mkdir(parents=True, exist_ok=True)
    (out / "ids_corpus.json").write_text(json.dumps(corpus_ids), encoding="utf-8")
    (out / "ids_queries_train.json").write_text(json.dumps(train_ids), encoding="utf-8")
    (out / "ids_queries_test.json").write_text(json.dumps(test_ids), encoding="utf-8")
    (out / "qrels_train.json").write_text(json.dumps(qrels_train), encoding="utf-8")
    (out / "qrels_test.json").write_text(json.dumps(qrels_test), encoding="utf-8")
    for name, ids, texts in (
        ("corpus", corpus_ids, corpus_texts),
        ("queries_train", train_ids, train_texts),
        ("queries_test", test_ids, test_texts),
    ):
        with (out / f"texts_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for identifier, text in zip(ids, texts):
                handle.write(json.dumps({"id": identifier, "text": text}) + "\n")

    print(f"  corpus={len(corpus_ids)}  train={len(train_ids)}  test={len(test_ids)}")
    return corpus_ids, corpus_texts, train_ids, train_texts, test_ids, test_texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--tier", choices=["accuracy", "interactive"], default="interactive")
    parser.add_argument("--tag", default=None, help="filename tag for the embedding arrays")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--texts-only", action="store_true", help="skip encoding")
    parser.add_argument("--limit", type=int, default=0, help="encode only the first N of each split")
    args = parser.parse_args()

    tag = args.tag or DEFAULT_TAGS[args.tier]
    corpus_ids, corpus_texts, train_ids, train_texts, test_ids, test_texts = dump_texts(args.out)
    if args.texts_only:
        print("--texts-only: stopping before the encoder.")
        return 0

    if args.limit:
        corpus_texts = corpus_texts[: args.limit]
        train_texts = train_texts[: args.limit]
        test_texts = test_texts[: args.limit]

    device = _select_device(args.device)
    encoder = CodeEncoder(
        device=device,
        tier=args.tier,
        torch_dtype=select_dtype(device),
    )
    print(f"Encoder: {encoder.model_name} on {device} (tier={args.tier}, tag={tag})")
    print(f"  query prefix   : {encoder.query_prefix!r}")
    print(f"  max_seq_length : {encoder.max_seq_length}   add_eos_token: {encoder.add_eos_token}")

    manifest_path = args.out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("models", {})
    timings: dict[str, float] = {}

    for split, texts, mode in (
        ("corpus", corpus_texts, "document"),
        ("qtrain", train_texts, "query"),
        ("qtest", test_texts, "query"),
    ):
        started = time.time()
        embeddings = encoder.encode(texts, mode=mode, show_progress_bar=True)
        elapsed = time.time() - started
        timings[split] = round(elapsed, 1)
        norms = np.linalg.norm(embeddings, axis=1)
        print(
            f"  {split}: {len(texts)} texts in {elapsed:.1f}s "
            f"({elapsed / max(1, len(texts)) * 1000:.0f} ms/text) "
            f"norm[{norms.min():.4f}..{norms.max():.4f}]"
        )
        np.save(args.out / f"emb_{tag}_{split}.npy", embeddings.astype(np.float16))

    manifest["instruction"] = encoder.query_prefix
    manifest["max_seq_length"] = encoder.max_seq_length
    manifest["models"][tag] = {
        "model_name": encoder.model_name,
        "revision": encoder.revision,
        "tier": args.tier,
        "device": device,
        "dim": int(encoder.model.get_sentence_embedding_dimension()),
        "seconds": timings,
    }
    manifest["counts"] = {
        "corpus": len(corpus_ids),
        "qtrain": len(train_ids),
        "qtest": len(test_ids),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
