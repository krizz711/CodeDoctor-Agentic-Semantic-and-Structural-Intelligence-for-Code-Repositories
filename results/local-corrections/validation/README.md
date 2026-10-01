# Local correction validation

No remote submission was changed. These checks do not reproduce P0 or certify
the real1.7B cold model path. Both remain dependent on sufficient Linux resources.

- `initial-suite.log`: existing local Python3.12/Torch2.14 environment,
  `pytest -q -rs -m "not model and not data"`:600passed,4failed,6skipped,
  6deselected,5subtests. Failures were outdated CLI task metadata fixtures;
  fixtures were repaired while keeping runtime checks strict.
- `portable-suite.log`: new local virtual environment installed with
  `pip install --index-url https://pypi.org/simple -r requirements-submission.txt`.
  Same selection:607passed,6skipped,6deselected,5subtests. Mac platform skips
  are listed in the raw log, not hidden. This pass preceded the final MTEB
  unloading/path-serialization regression repairs.
- `requirements-*-environment.txt` and `requirements-portable-tested.txt`:
  actual freezes, retained as evidence. They are not automatically selected
  cross-platform lock files.
- `judge-lexical.log`: `scripts/judge_endtoend.py --repo test_repos/commander`
  completed. This is the existing JavaScript lexical/structural fixture, not
  a dense1.7B Python model validation or an accuracy-tier latency result.
- `presentation-validation.json`: structural/reference-font/layout/import
  checks on the exact corrected12-slide PPTX. All slides were also visually
  inspected, including the matching PDF. Native PowerPoint was not exercised.
- `final-regressions.log`:43 passed after final source fixes, covering the
  launcher, CLI lifecycle, explicit shard-path serialization, exact prediction
  export and synthetic cold-Python checks. Real weights/data were not loaded.
- `disclosure-structure.json` and `disclosure-render-diff.json`: template
  preservation and latest rendered2-page evidence. Human sign-off remains blank.

Also passed: `ruff check app scripts tests --select F401,F811`, compilation,
`make -B`, `make -B test`, sample evaluator/validator and strict historical
AppsRetrieval JSON validation. Historical result validation is format checking,
not a new CPU run. New runtime changes get focused regression checks and a
clean-clone pass before cloud scoring. Commands and outcomes are copied here as
they finish. No Docker or corrected remote CI execution is claimed.
