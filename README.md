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
python scripts/evaluate_mteb.py --output appsretrieval_results.json
```

This runs the evaluation model on CPU and writes the MTEB task JSON required for a release upload. The evaluator dependencies remain separate because they are substantially heavier than the core product.

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

## BYOK model catalog

The optional model layer includes 30 configurable aliases in `config/models.json`, including DeepSeek, Kimi, GLM, Qwen, MiMo, MiniMax, Claude, GPT, Gemini, Grok, Llama, Mistral, Command, Perplexity, Ollama, and custom OpenAI-compatible endpoints.

```bash
python scripts/list_models.py
python scripts/list_models.py --configured-only
```

Set provider keys in your shell or secret manager using the names in `.env.example`. CodeDoctor never stores or prints key values. Model IDs and endpoints are configuration, because provider availability and naming change over time. The client supports OpenAI-compatible `/chat/completions` APIs and native Anthropic Messages requests. Local Ollama aliases do not require a remote key.
