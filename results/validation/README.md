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
