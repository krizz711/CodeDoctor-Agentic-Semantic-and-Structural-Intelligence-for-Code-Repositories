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

## Test

```bash
make test
pytest -q
python -m compileall -q app tests
```

The implementation is intentionally CPU-only. Dense model retrieval, FAISS, and an LLM response layer remain optional extensions rather than runtime requirements for the reproducible core product.
