"""Package only explicitly allowlisted APPS train/corpus artifacts for C4.

Never calls the development harness loader, which also opens test files.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import random
import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = ("emb_f2llm17b_corpus.npy", "emb_f2llm17b_qtrain.npy", "ids_corpus.json",
         "ids_queries_train.json", "qrels_train.json", "texts_corpus.jsonl", "texts_queries_train.jsonl")
EXPECTED = {"emb_f2llm17b_corpus.npy": "2d2b8938b80a4699887c3c839137d846c8d3e10e7bd533ad8bf71405efb2f99c",
            "emb_f2llm17b_qtrain.npy": "9e19cdc2f32be6aff0b40c4d2305263e5a3a775383620122df8a2d1551dc2fa3"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, out: Path, owner: str) -> None:
    train_ids = json.loads((source / "ids_queries_train.json").read_text())
    corpus_ids = json.loads((source / "ids_corpus.json").read_text())
    queries = [json.loads(x) for x in (source / "texts_queries_train.jsonl").read_text().splitlines()]
    corpus = [json.loads(x) for x in (source / "texts_corpus.jsonl").read_text().splitlines()]
    if [r["id"] for r in queries] != train_ids or len(set(train_ids)) != 5000:
        raise ValueError("train query identity/order mismatch")
    if [r["id"] for r in corpus] != corpus_ids or len(set(corpus_ids)) != 8765:
        raise ValueError("corpus identity/order mismatch")
    for name, expected in EXPECTED.items():
        if digest(source / name) != expected:
            raise ValueError(f"baseline vector hash mismatch: {name}")
    if np.load(source / FILES[0], mmap_mode="r", allow_pickle=False).shape != (8765, 2048):
        raise ValueError("corpus vector shape mismatch")
    if np.load(source / FILES[1], mmap_mode="r", allow_pickle=False).shape != (5000, 2048):
        raise ValueError("query vector shape mismatch")
    sets = {}
    store_hashes = {}
    for split in ("tune", "dev"):
        store = ROOT / f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz"
        with gzip.open(store, "rt", encoding="utf-8") as handle:
            header = json.loads(next(handle))
            ids = [json.loads(line)["qid"] for line in handle]
        if header.get("split") != split or len(ids) != 2500 or len(set(ids)) != 2500:
            raise ValueError(f"invalid existing {split} assignment")
        sets[split] = set(ids)
        store_hashes[store.name] = digest(store)
    if sets["tune"] & sets["dev"] or sets["tune"] | sets["dev"] != set(train_ids):
        raise ValueError("TUNE/DEV do not partition the original train queries")
    out.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copyfile(source / name, out / name)
    by_id = {r["id"]: r["text"] for r in queries}
    for split in ("tune", "dev"):
        ids = [qid for qid in train_ids if qid in sets[split]]
        value = {"qids": ids, "texts": [by_id[qid] for qid in ids]}
        (out / f"{split}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        sample = random.Random(20260929).sample(ids, 100)
        (out / f"{split}100_seed20260929.json").write_text(json.dumps({"qids": sample, "texts": [by_id[qid] for qid in sample]}, ensure_ascii=False), encoding="utf-8")
    manifest = {"purpose": "C4-E1 train-only inputs", "seed": 20260929,
        "assignment_sources": store_hashes, "sample_order": "original ids_queries_train.json order, random.Random(seed).sample",
        "excludes": ["APPS test queries, vectors and qrels", "EXT-lockbox", "original aggregate-score manifest"],
        "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({"id": f"{owner}/codedoctor-c4-train-inputs", "title": "codedoctor-c4-train-inputs", "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(f"Validated 2500 TUNE + 2500 DEV; wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/apps")
    parser.add_argument("--out", type=Path, default=ROOT / "build/c4_data")
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    build(args.source, args.out, args.owner)
