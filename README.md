# CodeDoctor: Agentic Semantic & Structural Intelligence for Code Repositories

CodeDoctor is a deterministic, evolutionary codebase retrieval engine designed for autonomous coding agents. It was built with strict adherence to **Theme 1: Coding Guidelines (No Shortcuts, Defensive Programming)** for the Samsung PRISM Hackathon.

## Core Features
- **Deterministic Tree-sitter AST Parsing**: Uses native C bindings for JavaScript to extract exact function and statement spans with byte-level fidelity, raising strict `ASTParseError` on syntax defects.
- **Evolutionary Indexing & Cross-Commit Search (P1 & Bonus)**: Uses SHA-256 semantic hashes to provide sub-second incremental index updates across git commits, enforcing zero-leakage search boundaries while supporting evolutionary tracking across commit histories.
- **Defensive Pybind11 Hybrid Engine**: Safely executes Reciprocal Rank Fusion on lexical and dense vectors, actively defending against Null Pointer segfaults via Python-layer interception before hitting C++.
- **MTEB Benchmark & Contrastive Fine-Tuning (P0)**: Interfaces directly with the Massive Text Embedding Benchmark (MTEB) on the CoIR `apps` dataset (`AppsRetrieval`), powered by an offline GPU-trained contrastive checkpoint (`coderank-apps-tuned`) with zero test-split leakage.
- **Real-Time CPU Latency Profiling**: Delivers end-to-end retrieval in ~1.0 ms on CPU (AST lookup: ~0.47 ms, C++ RRF fusion: ~0.01 ms).

## Submission Deliverables (Theme 1 Guidelines Pages 2 & 3)
Running `python mteb_eval.py` generates all submission artifacts:
1. `appsretrieval_results.json`: Official MTEB evaluation results file.
2. `submission_predictions.csv`: Ranked test responses (`query_id, corpus_id, rank, score`) used by judges to evaluate NDCG@10 and MRR.
3. `submission_metrics.csv`: Summary metrics table (NDCG@10, MRR, MAP, Recall, Precision).
4. `predictions/AppsRetrieval_predictions.json`: Raw MTEB prediction mappings.

## Official MTEB Benchmark Results (CoIR AppsRetrieval Test Split)
The benchmark ran strictly on CPU with multi-threading (`8 threads`) and evaluated on all 3,765 test queries and 8,768 corpus solutions:

| Evaluation Metric | Baseline / Naive Chunking | CodeDoctor (Fine-Tuned + Conditional AST) | Relative Gain |
|---|---|---|---|
| **NDCG@10** | 0.03290 | **0.47032** (47.03%) | **+14.3x leap** |
| **MRR@10** | 0.02690 | **0.42840** (42.84%) | **+15.9x leap** |
| **Hit Rate @ 10** | 0.05259 | **0.60398** (60.40%) | **+11.5x leap** |
| **Recall @ 10** | 0.05259 | **0.60398** (60.40%) | **+11.5x leap** |
| **NDCG@1** | 0.01753 | **0.35086** (35.09%) | **+20.0x leap** |
| **MAP@10** | 0.02690 | **0.42840** (42.84%) | **+15.9x leap** |

## Codebase Architecture & File Guide

Below is a detailed breakdown of all files and modules in the CodeDoctor codebase:

```text
├── app/                          # Core CodeDoctor retrieval engine modules
│   ├── chunker.py                # Tree-sitter AST parser (exact byte spans, SHA-256 semantic hashing, ASTParseError)
│   ├── db.py                     # SQLite database connection factory and schema migration runner
│   ├── eval_harness.py           # Evaluation harness interface, leakage firewall, and verified artifact loader
│   ├── evolution.py              # Incremental indexing, zero-leakage commit queries, and evolutionary cross-commit search
│   ├── execution_reranker.py     # Sandbox runner verifying candidate code against test input/output specifications
│   ├── hybrid_engine.py          # Python boundary wrapper defending against C++ null-pointer segfaults
│   ├── hybrid_eval.py            # Diagnostic script evaluating dense search fused with execution verification
│   ├── indexer.py                # SQLite chunk indexer with automatic transaction rollback on parse defects
│   ├── main.py                   # Stitched end-to-end pipeline with dynamic commit resolution and CPU latency profiling
│   └── schemas.py                # Pydantic data schemas (SearchResult, SearchResponse, FileCreate, ChunkCreate)
├── coderank-apps-tuned/          # Fine-tuned NomicBERT checkpoint (137M parameters, ~500MB RAM footprint on CPU)
│   ├── 1_Pooling/config.json     # Sentence-transformers mean pooling configuration
│   ├── config.json               # Model hyperparameters and architecture definitions
│   ├── configuration_hf_nomic_bert.py # Hugging Face configuration definition for NomicBERT
│   ├── modeling_hf_nomic_bert.py # Patched NomicBERT model code (fixed for transformers v5.17.0 compatibility)
│   ├── tokenizer.json            # Fast BPE tokenizer vocabulary and merge rules
│   └── tokenizer_config.json     # Tokenizer settings and special token mappings
├── src/                          # Native C++ performance layer
│   └── hybrid.cpp                # Native C++ Reciprocal Rank Fusion (RRF) algorithm bound via Pybind11
├── scripts/                      # Training and verification utilities
│   ├── colab_train.py            # Reproducible Google Colab GPU training script (train split only, zero test leakage)
│   └── train_coderank.py         # Local training script and leakage firewall verification utility
├── tests/                        # Comprehensive automated test suite (13/13 passing 100% green)
│   ├── test_module1.py           # T0 (Execution), T1 (Reproducibility), T2 (Leakage Firewall), T4 (Sanity Floor)
│   ├── test_module2.py           # T7 (Tree-sitter Byte Span Fidelity), T10 (No Silent Fallback / ASTParseError)
│   ├── test_module3.py           # T14 (Latency < 3s), Defensive Pybind11 Memory Boundary, Hybrid RRF Superiority
│   ├── test_module4.py           # T8 (Incremental Equals Rebuild Hash Parity), T9 (Zero Cross-Commit Leakage)
│   └── test_module5.py           # T12 (Multi-Repo Generalisation), T13 (End-to-End Pipeline Stability)
├── benchmark.py                  # Architectural edge tests (Token Boundary, Whitespace Invalidation, SWE-bench Query Noise)
├── demo.py                       # Interactive CLI live demo matching Theme 1 Page 1 query with real CPU latency profiling
├── mteb_eval.py                  # Official MTEB benchmark runner on CPU with atomic caching and submission export
├── setup.py                      # C++ Pybind11 build configuration for codedoctor_cpp
├── project_history.md            # Mandatory Architectural Decision Record (ADR) detailing Sections 1 through 15
├── appsretrieval_results.json    # Generated official MTEB benchmark result artifact (NDCG@10 = 0.47032)
├── submission_predictions.csv    # Generated ranked query responses (376,502 rows) for official judge evaluation
├── submission_metrics.csv        # Generated summary table of all benchmark metrics
└── theme1_guidelines.pdf         # Official Samsung PRISM Hackathon Theme 1 specification document
```

## Generalisation & Validation Metrics
| Metric | Status | Result |
|---|---|---|
| **T0 (Harness Execution)** | PASSED | Executes end-to-end against MTEB standard |
| **T1 (Reproducibility)** | PASSED | Deterministic outputs across multiple seeded runs |
| **T2 (Leakage Firewall)** | PASSED | Zero collision between 5,000 train queries and 3,765 test queries |
| **T7 (Span Fidelity)** | PASSED | Exact byte-level span extraction via Tree-sitter (`raw_code[start:end] == content`) |
| **T8 (Incremental Rebuild)** | PASSED | Deterministic equality verified via SHA-256 content hashing |
| **T9 (Zero Leakage)** | PASSED | Commits isolated securely in SQLite |
| **T10 (Silent Errors)** | PASSED | `ASTParseError` caught & rolled back defensively |
| **T12 (Generalisation)** | PASSED | Dynamic routing to multi-repo structures |
| **T13 (E2E Stability)** | PASSED | Pipeline stitched and stable |
| **T14 (Latency Limit)** | PASSED | Sub-3.0 second Hybrid Search enforced (~1.0 ms real latency) |

## Installation & Usage
```bash
# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the Verified Test Suite (13 tests passing)
PYTHONPATH=. pytest tests/

# Run CodeDoctor Benchmark Suite (Token Boundary, Whitespace Invalidation, Query Noise)
PYTHONPATH=. python benchmark.py

# Run the Live Interactive Demo CLI (Theme 1 Page 4 Demo Requirements)
PYTHONPATH=. python demo.py --repo AuthService --query "How is the input preprocessed before going to the main function?" --commit v2.0.0

# Run Official MTEB Benchmark on CPU using the Fine-Tuned Checkpoint
PYTHONPATH=. python mteb_eval.py
```

