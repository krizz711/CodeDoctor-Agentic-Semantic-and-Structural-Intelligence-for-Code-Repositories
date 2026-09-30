"""Merge query-vector stores written by ``evaluate_mteb.py --query-shard`` into one store.

Every vector in a store was computed by the CPU encoder from a query the MTEB run
supplied; merging only copies rows. The result carries a manifest with each input's
checksum and row count so the release can show exactly what the final run started from.
When two inputs hold the same key their vectors are compared and the disagreement is
recorded: the same recipe on two machines should agree to float32 noise, and the final
run re-encodes a fixed sample live as an independent check.

    python scripts/merge_query_stores.py --out build/queries.sqlite shard0/queries.sqlite ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.indexing.cache import EmbeddingCache  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(inputs: list[Path], out: Path) -> dict:
    """Copy every row of ``inputs`` into ``out`` (created if absent); return a manifest."""
    if not inputs:
        raise ValueError("no query stores to merge")
    merged = EmbeddingCache(out)
    sources, disagreements, worst = [], 0, 1.0
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"query store not found: {path}")
        # WAL leftovers must be folded in before the file is hashed or copied
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        store = EmbeddingCache(path)
        keys = [key for (key,) in store.conn.execute("SELECT cache_key FROM embedding_cache WHERE is_real = 1")]
        fresh = []
        for key in keys:
            vector = store.get(key)
            existing = merged.get(key)
            if existing is None:
                fresh.append((key, vector))
                continue
            cosine = float(np.dot(existing, vector) / (np.linalg.norm(existing) * np.linalg.norm(vector)))
            worst = min(worst, cosine)
            disagreements += cosine < 0.9999
        merged.put_many(fresh)
        store.conn.close()
        sources.append({"path": str(path), "rows": len(keys), "added": len(fresh), "sha256": sha256(path)})
    merged.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    total = merged.count()
    merged.conn.close()
    return {"inputs": sources, "rows": total, "duplicate_keys_disagreeing": disagreements,
            "worst_duplicate_cosine": worst, "sha256": sha256(out)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=None, help="fail unless the merged store has this many rows")
    parser.add_argument("stores", type=Path, nargs="+")
    args = parser.parse_args(argv)
    manifest = merge(args.stores, args.out)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "inputs"}, indent=2))
    if args.expect is not None and manifest["rows"] != args.expect:
        raise SystemExit(f"merged store has {manifest['rows']} rows, expected {args.expect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
