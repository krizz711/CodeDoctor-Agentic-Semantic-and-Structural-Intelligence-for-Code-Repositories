# CodeDoctor: Agentic Semantic & Structural Intelligence for Code Repositories

CodeDoctor is a deterministic, evolutionary codebase retrieval engine designed for autonomous coding agents. It was built with strict adherence to **Theme 1: Coding Guidelines (No Shortcuts, Defensive Programming)**.

## Core Features
- **Deterministic AST Parsing**: Leverages true C++ `tree-sitter` bindings.
- **Evolutionary Indexing**: Uses SHA-256 semantic hashes to provide sub-second incremental index updates across git commits, enforcing a zero-leakage search boundary.
- **Defensive Pybind11 Hybrid Engine**: Safely executes Reciprocal Rank Fusion on lexical and dense vectors, actively defending against Null Pointer segfaults via Python-layer interception.
- **MTEB Sanity Floor**: Enforces structural evaluation thresholds against Massive Text Embedding Benchmarks.

## Generalisation & Validation Metrics
| Metric | Status | Result |
|---|---|---|
| **T7 (Span Fidelity)** | PASSED | Exact line numbers retrieved via Tree-sitter |
| **T8 (Incremental Rebuild)** | PASSED | Deterministic equality verified |
| **T9 (Zero Leakage)** | PASSED | Commits isolated securely in SQLite |
| **T10 (Silent Errors)** | PASSED | `ASTParseError` caught & rolled back |
| **T12 (Generalisation)** | PASSED | Dynamic routing to multi-repo structures |
| **T13 (E2E Stability)** | PASSED | Pipeline stitched and stable |
| **T14 (Latency Limit)** | PASSED | Sub-3.0 second Hybrid Search enforced |

## Installation & Usage
```bash
# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the Test Suite
pytest tests/
```
