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
- `clean-clone-suite.log`: frozen commit `d2fb22a` in a fresh local Git clone,
  declared dependency environment, same existing test selection: **608 passed,
  7 skipped, 6 deselected, 5 subtests**. Six skips are Mac platform differences;
  one requires the optional pre-existing corpus-text artifact, absent from a
  clean clone. Exact pinned-task text coverage still gates the final P0 run.
- `downloaded-artifact-preflight.json`: existing downloaded GPU document cache
  and four CPU query shards passed checksums/provenance, dimensions, finite
  normalized-vector checks, declared membership and disjoint merging. Counts:
  8,765 documents /8,754 unique document vectors and3,765 query vectors. All52
  recorded shard source paths match submitted408a61f. No new query text, labels
  or model was loaded; exact official text coverage and live checks remain pending.
- `preflight-wrapper-failure.log`: first local evidence wrapper used a wrong
  manifest field name (`total_vectors` instead of `rows`) after gates completed;
  one corrected retry wrote the report. Raw inputs and failing attempt preserved.
- `video-cut-review.json`, `video-preview-probe.json`, `video-decode.log`: local
  cut-only4:53.410 preview; full decode passed, audio/video differ10ms, retained
  audio spot checks >0.998 correlation. Original/public video unchanged. Human
  listening and missing dense/edit demonstrations are not certified.

Also passed: `ruff check app scripts tests --select F401,F811`, compilation,
`make -B`, `make -B test`, sample evaluator/validator and strict historical
AppsRetrieval JSON validation. Historical result validation is format checking,
not a new CPU run. New runtime changes get focused regression checks and a
clean-clone pass before cloud scoring. Commands and outcomes are copied here as
they finish. No Docker or corrected remote CI execution is claimed.
