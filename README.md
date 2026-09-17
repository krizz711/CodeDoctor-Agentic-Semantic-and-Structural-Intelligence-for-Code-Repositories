# CodeDoctor

CodeDoctor is a grounded repository intelligence system for code understanding and retrieval.

CodeDoctor is a CPU-first C++ repository intelligence tool. It indexes JavaScript and TypeScript source into exact, citation-ready chunks, supports normalized symbol lookup and keyword retrieval, persists the index locally, and exposes grounded search commands without requiring a GPU, Python runtime, database server, or hosted model.

## Build

```bash
make
```

On Windows with MinGW GNU Make, the same command builds `build/codedoctor.exe`.

## Index and query a repository

```bash
build/codedoctor.exe index path/to/repository build/my-repository.index
build/codedoctor.exe symbol build/my-repository.index loginUser
build/codedoctor.exe search build/my-repository.index "password reset"
build/codedoctor.exe ask build/my-repository.index "Where is authentication handled?"
build/codedoctor.exe callers build/my-repository.index authenticate
build/codedoctor.exe stats build/my-repository.index
```

The indexer scans `.js`, `.jsx`, `.mjs`, `.cjs`, `.ts`, and `.tsx`, while skipping `.git`, `node_modules`, `dist`, `build`, `coverage`, `.cache`, `.idea`, and `.vscode`. Search results include the relative file, symbol, chunk type, score, and exact source span.

## CPU retrieval evaluation

The Python retrieval layer combines exact identifier matching, BM25-style lexical scoring, TF-IDF cosine similarity, query expansion, and reciprocal-rank fusion. It is deterministic and CPU-only.

```bash
python scripts/evaluate.py scripts/sample_eval.jsonl --output build/sample_results.json
```

The input is JSONL with `query`, `documents`, and `relevant_ids`. The output contains ranked predictions plus `mrr` and `ndcg_at_10`. Replace the sample JSONL with a converted CoIR Apps split for local experiments. The official MTEB package and Apps dataset are intentionally not downloaded as part of the default build.

For the official AppsRetrieval artifact:

```bash
pip install -r requirements-eval.txt
python scripts/verify_encoder.py --real-model          # sanity check first
python scripts/evaluate_mteb.py --output appsretrieval_results.json
```

This evaluates **CodeDoctor's own `PrePostPipelineEncoder`** (`app/retrieval/mteb_encoder.py`)
on CPU, not a stock baseline model, and writes the MTEB task JSON required for the release
upload. The encoder pre-processes queries and code differently based on `prompt_type`, encodes
with a CPU dense backbone, and concatenates a seeded hashed-lexical block so that cosine
similarity becomes a weighted sum of dense and lexical similarity — hybrid retrieval expressed
within MTEB's encode-only interface.

The script exits non-zero if the resulting scores sit at random-retrieval level, so a broken
encoder cannot silently produce a submission file. Run a dense-only ablation with
`--lexical-weight 0.0`.

See `SUBMISSION_CHECKLIST.md` for the full release procedure.

## Version-aware retrieval

`app/indexing/versioned.py` provides explicit snapshot membership and persistence:

```python
from app.indexing.versioned import VersionedIndex
from app.retrieval.hybrid import Document

index = VersionedIndex()
index.add_snapshot("commit-a", [Document("chunk-a", "src/auth.js", "function auth() {}", "auth")])
results = index.retrieve("authentication", commit="commit-a")
```

Unknown snapshots fail explicitly, and retrieval results are restricted to the requested commit. This prevents silent fallback to current HEAD content.

## Test

```bash
make test
pytest -q
python -m compileall -q app tests
```

The implementation is intentionally CPU-only. Dense model retrieval, FAISS, and an LLM response layer remain optional extensions rather than runtime requirements for the reproducible core product.
