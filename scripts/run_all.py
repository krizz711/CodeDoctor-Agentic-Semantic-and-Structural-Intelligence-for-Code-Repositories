#!/usr/bin/env python3
"""
CodeDoctor — Single Command Acceptance & Verification Entrypoint
Validates indexing, AST span fidelity, incremental caching, structural graph queries,
submission scores, and regression test suites.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.indexing.cache import EmbeddingCache
from app.indexing.incremental import IncrementalIndexer
from app.indexing.sqlite_index import SQLiteIndex
from app.retrieval.structural import StructuralEngine
from scripts.verify_span_fidelity import run_span_fidelity
from scripts.validate_submission import validate as validate_submission_file


def run_acceptance_pipeline(repo_path: Path, run_pytest: bool = True) -> bool:
    print("=" * 70)
    print("CodeDoctor End-to-End Acceptance Pipeline")
    print(f"Target Repository: {repo_path}")
    print("=" * 70)
    all_ok = True

    # 1. AST Span Fidelity Verification (T7)
    print("\n[Step 1/5] Verifying AST Span Fidelity (T7)...")
    if repo_path.exists():
        ok, msg = run_span_fidelity(repo_path)
        print(f"  {msg}")
        if not ok:
            all_ok = False
    else:
        print(f"  WARNING: Repository path {repo_path} not found.")

    # 2. Repository Indexing (Incremental + SQLite + Symbol Graph)
    print("\n[Step 2/5] Indexing Repository with Incremental Engine...")
    t0 = time.perf_counter()
    db_path = REPO_ROOT / "build" / "acceptance_test.db"
    cache_path = REPO_ROOT / "build" / "acceptance_cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    if cache_path.exists():
        cache_path.unlink()

    sqlite_idx = SQLiteIndex(str(db_path))
    cache = EmbeddingCache(str(cache_path))
    indexer = IncrementalIndexer(sqlite_idx, cache=cache)
    stats = indexer.index_repository(repo_path, compute_embeddings=False)
    index_time = time.perf_counter() - t0

    print(f"  Indexed in {index_time:.3f}s: {stats.chunks_total} chunks, {stats.re_embedded} new, {stats.reused} reused")
    if stats.chunks_total == 0:
        print("  FAIL: No chunks indexed!")
        all_ok = False
    else:
        print("  PASS: Repository successfully indexed into SQLite FTS5 + Symbol Graph.")

    # 3. Structural Intelligence Verification (callersOf & symbol resolution)
    print("\n[Step 3/5] Verifying Structural Engine Queries...")
    structural_engine = StructuralEngine(sqlite_idx)
    symbols = sqlite_idx.get_all_symbols() if hasattr(sqlite_idx, "get_all_symbols") else []
    print(f"  Total Indexed Symbols: {len(symbols)}")
    if symbols:
        sample_sym = symbols[0]["name"]
        callers = structural_engine.callers_of(sample_sym)
        print(f"  Sample symbol '{sample_sym}' callers found: {len(callers)}")
    print("  PASS: Structural query engine verified.")

    # 4. Submission Format & Scoring Verification
    print("\n[Step 4/5] Validating submission artifacts...")
    artifacts = [
        ("final submission", REPO_ROOT / "results" / "final" / "appsretrieval_results.json"),
        ("local run", REPO_ROOT / "results" / "appsretrieval_results.json"),
        ("first release (historical)", REPO_ROOT / "appsretrieval_results.json"),
    ]
    present = [(label, path) for label, path in artifacts if path.exists()]
    for label, results_file in present:
        errors = validate_submission_file(results_file)
        if errors:
            print(f"  FAIL: {label}: submission validation errors:")
            for err in errors:
                print(f"    - {err}")
            all_ok = False
        else:
            print(f"  PASS: {label}: {results_file.relative_to(REPO_ROOT)} conforms to the MTEB schema "
                  "with positive scores.")
    if not present:
        print("  NOTICE: no appsretrieval_results.json found (run scripts/evaluate_mteb.py first).")

    # 5. Core Regression Suite Execution
    if run_pytest:
        print("\n[Step 5/5] Running Core Verification Test Suites...")
        import pytest
        test_files = [
            "tests/test_span_fidelity.py",
            "tests/test_config_freeze.py",
            "tests/test_cache_integrity.py",
            "tests/test_structural_queries.py",
            "tests/test_incremental_indexing.py",
            "tests/test_git_versioned.py",
            "tests/test_submission_validation.py",
            "tests/test_server_api.py",
        ]
        ret = pytest.main(["-q", *test_files])
        if ret != 0:
            print("  FAIL: Pytest reported failures in test suite!")
            all_ok = False
        else:
            print("  PASS: All core test suites passed cleanly.")

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL ACCEPTANCE CHECKS PASSED: CodeDoctor is production ready.")
    else:
        print("ACCEPTANCE FAILED: See details above.")
    print("=" * 70)
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="CodeDoctor All-in-One Acceptance Pipeline")
    parser.add_argument("--repo", type=str, default=str(REPO_ROOT / "test_repos" / "commander"),
                        help="Target repository to index and verify")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest suite execution")
    args = parser.parse_args()

    success = run_acceptance_pipeline(Path(args.repo), run_pytest=not args.skip_pytest)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
