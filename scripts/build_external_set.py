"""Build an APPS-shaped held-out set from CodeContests (CC BY 4.0).

    python scripts/build_external_set.py --out artifacts/ext_cc

Why: every number on APPS DEV rides on an encoder family that trained on APPS,
and the jury will score an *unseen* APPS-like set.  CodeContests valid+test
(282 Codeforces problems, DeepMind) gives statements the pipeline was never
tuned on, in a layout the original parser does not fully read, with Python 3
solutions and -- usefully -- *incorrect* solutions, which are the hard
near-miss negatives execution has to reject.

Output (BEIR-like, the same file names ``record_outcomes.py`` reads):

* ``texts_queries.jsonl``  -- one statement per problem, id ``cc-<split>-<n>``
* ``texts_corpus.jsonl``   -- the APPS corpus (8,765 distractors) plus one
  correct Python 3 solution per problem (``ccd-...``); with
  ``--hard-negatives`` also up to K incorrect Python 3 solutions per problem
  (``ccx-...``), never marked relevant
* ``qrels.json``           -- query -> {correct solution: 1}
* ``dedup_report.json``    -- problems dropped as near-duplicates of an APPS
  statement (shingle Jaccard), so APPS familiarity cannot leak in
* ``splits.json``          -- a seeded EXT-dev / EXT-lockbox partition

Nothing here is ever used for tuning except EXT-dev; EXT-lockbox is reported
once, after a configuration is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CC = "https://huggingface.co/datasets/deepmind/code_contests/resolve/refs%2Fconvert%2Fparquet/default/"
APPS = "https://huggingface.co/datasets/CoIR-Retrieval/apps/resolve/refs%2Fconvert%2Fparquet/"
PYTHON3 = 3  # CodeContests language code


def _read(url: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(urllib.request.urlopen(url).read()))


def _shingles(text: str, k: int = 8) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        int(hashlib.blake2b(" ".join(words[i : i + k]).encode(), digest_size=8).hexdigest(), 16)
        for i in range(max(1, len(words) - k + 1))
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "ext_cc")
    parser.add_argument("--dedup-threshold", type=float, default=0.5)
    parser.add_argument("--hard-negatives", type=int, default=3,
                        help="incorrect Python 3 solutions kept per problem (0 = none)")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    print("loading CodeContests valid+test ...")
    frames = []
    for split in ("valid", "test"):
        frame = _read(CC + f"partial-{split}/0000.parquet")
        frame["cc_split"] = split
        frames.append(frame)
    cc = pd.concat(frames, ignore_index=True)
    print(f"  {len(cc)} problems")

    print("loading APPS statements and corpus (for dedup and as distractors) ...")
    apps_queries = _read(APPS + "queries/queries/0000.parquet")
    apps_corpus = _read(APPS + "corpus/corpus/0000.parquet")
    apps_shingles = [_shingles(t) for t in apps_queries["text"]]

    queries, corpus, qrels, dropped, skipped = [], [], {}, [], 0
    for position, row in cc.iterrows():
        statement = row["description"]
        solutions = row["solutions"]
        langs, codes = list(solutions["language"]), list(solutions["solution"])
        correct = [c for l, c in zip(langs, codes) if l == PYTHON3 and c.strip()]
        if not correct or not statement.strip():
            skipped += 1
            continue
        mine = _shingles(statement)
        best = 0.0
        for other in apps_shingles:
            if not other:
                continue
            overlap = len(mine & other) / max(1, len(mine | other))
            if overlap > best:
                best = overlap
        qid = f"cc-{row['cc_split']}-{position}"
        if best >= args.dedup_threshold:
            dropped.append({"qid": qid, "name": row["name"], "jaccard": round(best, 3)})
            continue
        gold_code = rng.choice(correct)
        did = f"ccd-{row['cc_split']}-{position}"
        queries.append({"id": qid, "text": statement})
        corpus.append({"id": did, "text": gold_code})
        qrels[qid] = {did: 1}
        if args.hard_negatives:
            wrong = row["incorrect_solutions"]
            bad = [c for l, c in zip(list(wrong["language"]), list(wrong["solution"]))
                   if l == PYTHON3 and c.strip()]
            for k, code in enumerate(rng.sample(bad, min(args.hard_negatives, len(bad)))):
                corpus.append({"id": f"ccx-{row['cc_split']}-{position}-{k}", "text": code})

    for doc_id, text in zip(apps_corpus["_id"], apps_corpus["text"]):
        corpus.append({"id": f"apps-{doc_id}", "text": text})

    ids = [q["id"] for q in queries]
    rng.shuffle(ids)
    half = len(ids) // 2
    splits = {"ext_dev": sorted(ids[:half]), "ext_lockbox": sorted(ids[half:])}

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "texts_queries.jsonl").open("w", encoding="utf-8") as handle:
        for q in queries:
            handle.write(json.dumps(q) + "\n")
    with (args.out / "texts_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for d in corpus:
            handle.write(json.dumps(d) + "\n")
    (args.out / "qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
    (args.out / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    (args.out / "dedup_report.json").write_text(
        json.dumps({"threshold": args.dedup_threshold, "dropped": dropped, "skipped_no_python3": skipped},
                   indent=1),
        encoding="utf-8",
    )
    hard = sum(1 for d in corpus if d["id"].startswith("ccx-"))
    print(f"queries={len(queries)}  corpus={len(corpus)} (gold {len(qrels)}, hard negatives {hard}, "
          f"APPS distractors {len(apps_corpus)})  dropped near-duplicates={len(dropped)}  "
          f"no Python 3 solution={skipped}")
    print(f"EXT-dev={len(splits['ext_dev'])}  EXT-lockbox={len(splits['ext_lockbox'])}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
