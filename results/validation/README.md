# Finalization validation, 2026-09-30

Measured source preparation commit: d3ce432.

- Local Mac: `python -m pytest -q -rs -m "not model and not data"`:
  557 passed, 6 skipped, 6 deselected, 2 failed. Both failures were macOS process
  enumeration denied by the Codex desktop sandbox, not retrieval assertions.
  Full raw output: `macos-suite-sandbox.log`.
- Re-ran the entire affected module outside that sandbox:
  `python -m pytest -q -rs tests/test_submission_run.py`: 13 passed.
  Raw output: `macos-runner-unsandboxed.log`.
- `ruff check app scripts tests --select F401,F811`: passed.
- `compileall -q app scripts tests`: passed.
- `make -B` and `make -B test`: passed.
- Sample harness followed by JSON validation: passed. Sample scores are not P0.
- GitHub CI run 36721332066: success on d3ce432 (Linux core and platform checks).

The first local attempt lacked mteb and psutil (6 failed, 5 errors, 541 passed,
12 skipped, 6 deselected). Installed the pinned submission stack and reran.
Local torch is 2.14.0; hosted runs explicitly install torch 2.8.0. Local checks
are not a claim of CPU benchmark parity with Kaggle.

Final measurements are separate, recorded by scripts/submission_run.py.

## Hosted Kaggle validation results, 2026-09-30

Measured source commit: `823aa87`. Account: `lifemeansfun`.

### 1. Document Preprocessing (`codedoctor-submission-documents`)
- **Device:** Tesla T4 GPU (permitted under FAQ for offline indexing).
- **Corpus:** 8,765 documents encoded in 1,301 s into content-addressed SQLite store.
- **Verification:** Manifest checksum and vector counts verified against pinned model revision.

### 2. Hands-on Benchmark (`codedoctor-submission-hands-on`)
- **Device:** 4-vCPU Intel Xeon (pure CPU inference).
- **P1 Incremental Indexing (3 repetitions, 40 functions across 10 modules):**
  - Full rebuild: median 102.8 s (50 chunks encoded, 0 reused).
  - Unchanged repository: median 1.51 s (0 encoded, 50 reused, ~68x speedup).
  - Single function edit: median 6.45 s (2 encoded, 48 reused).
  - Ten function edit: median 29.5 s (13 encoded, 37 reused).
  - Add function: median 4.34 s (2 encoded, 50 reused).
  - Delete function: median 1.44 s (0 encoded, 50 reused).
  - Rename file: median 1.42 s (0 encoded, 50 reused).
  - Fresh rebuild equivalence: 100% (min cosine 1.0, 6/6 queries identical ranking).
  - Verification: `all_encoded_counts_exact = True`, `all_equivalent_to_fresh_rebuild = True`.
- **Bonus Cross-Version Retrieval (12 lineages across 3 snapshot versions):**
  - `correct_version_first`: 1.0 (100% correct variant retrieved as top rank).
  - `verified_by_execution`: 1.0.
  - `control_no_verification_correct_first`: 0.0833 (8.33% without post-processing execution).
  - Confirms ~12x increase in accuracy when disambiguating competing near-miss code versions using worked example execution.
