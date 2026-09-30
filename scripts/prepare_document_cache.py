"""Offline corpus preprocessing: encode the AppsRetrieval corpus once into a document cache.

The FAQ allows a GPU for pre-processing ("You are allowed to use the GPU during pre/post
processing"), and asks for minimal GPU use *during retrieval*. This script is that
offline step. It reads only the corpus config of the pinned dataset revision (no query
or qrel split is downloaded) and writes:

    <out>/corpus.sqlite   content-addressed document vectors (see app.indexing.cache)
    <out>/manifest.json   dataset/model revisions, dtype, device, counts, checksums

The texts are the strings MTEB passes to ``encode`` (title joined, stripped), not the raw
dataset field: keying on raw text would miss the 3,687 documents whose text has leading or
trailing whitespace. ``scripts/evaluate_mteb.py --require-complete-corpus-cache`` then
refuses to run if any document is missing.

    python scripts/prepare_document_cache.py --device cuda --out build/documents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.indexing.cache import EmbeddingCache, recipe_cache_key  # noqa: E402
from app.pipeline.encoder import mteb_document_text  # noqa: E402
from app.retrieval.encode import CodeEncoder, select_dtype  # noqa: E402

WINDOW = 64


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in ("torch", "transformers", "sentence-transformers", "mteb", "datasets", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test only: encode the first N documents (the manifest records it and "
                             "the final run refuses such a cache)")
    args = parser.parse_args(argv)

    import mteb
    from datasets import load_dataset

    dataset = dict(mteb.get_task("AppsRetrieval").metadata.dataset)
    if not dataset.get("revision"):
        raise RuntimeError("MTEB dataset revision is not pinned")
    corpus = load_dataset(dataset["path"], "corpus", split="corpus", revision=dataset["revision"])
    rows = list(corpus)[: args.limit] if args.limit else list(corpus)
    texts = [mteb_document_text(row) for row in rows]

    args.out.mkdir(parents=True, exist_ok=False)
    encoder = CodeEncoder(tier="accuracy", device=args.device, torch_dtype=select_dtype(args.device))
    cache = EmbeddingCache(args.out / "corpus.sqlite")
    keys = [recipe_cache_key(encoder, text) for text in texts]
    unique: dict[str, int] = {}
    for i, key in enumerate(keys):
        unique.setdefault(key, i)

    started = time.perf_counter()
    todo = list(unique.values())
    for start in range(0, len(todo), WINDOW):
        batch = todo[start:start + WINDOW]
        vectors = encoder.encode([texts[i] for i in batch], mode="document", show_progress_bar=False)
        cache.put_many([(keys[i], v) for i, v in zip(batch, vectors)])
        print(f"Documents {min(start + WINDOW, len(todo))}/{len(todo)}", flush=True)
    seconds = time.perf_counter() - started

    stored = cache.get_many(list(unique))
    if len(stored) != len(unique):
        raise RuntimeError(f"cache holds {len(stored)} of {len(unique)} vectors after encoding")
    cache.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    cache.conn.close()

    gpu = None
    if args.device == "cuda":
        import torch

        gpu = torch.cuda.get_device_name(0)
    manifest = {
        "dataset": dataset,
        "model": encoder.model_name,
        "revision": encoder.revision,
        "device": args.device,
        "gpu": gpu,
        "dtype": encoder.torch_dtype,
        "documents": len(texts),
        "unique_vectors": len(unique),
        "limit": args.limit,
        "texts_sha256": hashlib.sha256("\n".join(hashlib.sha256(t.encode("utf-8")).hexdigest()
                                                  for t in texts).encode("ascii")).hexdigest(),
        "text_rule": "mteb._create_dataloaders._corpus_to_dict(row)['text'] (title joined, stripped)",
        "seconds": seconds,
        "packages": package_versions(),
        "sha256": hashlib.sha256((args.out / "corpus.sqlite").read_bytes()).hexdigest(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out / 'corpus.sqlite'} ({len(unique)} vectors, {seconds:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
