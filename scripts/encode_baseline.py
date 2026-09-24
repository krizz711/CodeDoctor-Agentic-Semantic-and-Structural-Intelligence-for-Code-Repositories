"""Encode APPS with an encoder that has *not* been trained on APPS.

    python scripts/encode_baseline.py --model microsoft/unixcoder-base --tag unixcoder

The jury will neutralise contamination with an unseen dataset and said model
choice should not matter much -- pre/post-processing is what is judged.  So
every gate is also run with an encoder whose training predates CoIR/APPS
(UniXcoder, 2022, CodeSearchNet-era data, MIT licence).  Its absolute scores
will be low; what matters is whether the pipeline's *gain over the encoder*
survives when the encoder's APPS familiarity is gone.

Plain sentence-transformers wrapping (mean pooling over the model's own
tokenizer, no instruction), unlike the frozen F2LLM recipe in
``app.retrieval.encode``.  Writes ``emb_<tag>_corpus.npy`` and
``emb_<tag>_qtrain.npy`` next to the F2LLM artifacts; the test split is not
encoded.
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

from scripts.dev_eval import load_jsonl_texts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device, revision=args.revision)
    model.max_seq_length = min(args.max_seq_length, model.max_seq_length or args.max_seq_length)
    print(f"{args.model} on {args.device}, max_seq_length {model.max_seq_length}")
    timings = {}
    for split, name in (("corpus", "texts_corpus.jsonl"), ("qtrain", "texts_queries_train.jsonl")):
        _, texts = load_jsonl_texts(args.artifacts / name)
        started = time.time()
        emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        timings[split] = round(time.time() - started, 1)
        np.save(args.artifacts / f"emb_{args.tag}_{split}.npy", emb.astype(np.float16))
        print(f"  {split}: {len(texts)} texts in {timings[split]}s")
    manifest_path = args.artifacts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("models", {})[args.tag] = {
        "model_name": args.model, "revision": args.revision, "recipe": "sentence-transformers default",
        "max_seq_length": model.max_seq_length, "seconds": timings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
