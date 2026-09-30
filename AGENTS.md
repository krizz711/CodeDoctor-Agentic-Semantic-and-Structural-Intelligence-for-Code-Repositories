# CodeDoctor: agent entry point

Read `CLAUDE.md` before project work. It is the shared research policy for all
agents, including Codex: preserve its verbatim jury FAQ, preregistration rules,
evidence standards, and final-decision review. Do not copy a shortened FAQ into
another policy file. User instructions take precedence.

## Find the current truth

- Start from the actual branch, `git status`, and the relevant implementation.
- `docs/CODEX-ANALYSIS.md` maps the two pipelines and records the audited base.
- `experiments/C3_cycle/PREREG.md` (including its addenda), result files, and
  `config/candidate_v2.yaml` are newer than parts of the README, architecture
  plan, and handoff. Check chronology before repeating a status or metric.
- Treat `docs/V2-ARCHITECTURE-PLAN.md` as historical proposals. Several were
  rejected; its strict CPU-only FAQ paraphrase is superseded by `CLAUDE.md`.
- Machine facts from `CLAUDE.local.md` or old Windows/Kaggle notes do not describe
  this checkout. Discover the current interpreter, hardware and artifact paths.

## Architecture routing

| Work | Start here |
| --- | --- |
| Scored retrieval | `scripts/evaluate_mteb.py`, `app/pipeline/encoder.py`, `app/retrieval/encode.py` |
| Verification/parsing | `app/pipeline/execution_rerank.py`, `app/pipeline/statement.py`, relevant execution tests |
| Demo search | `app/server.py`, `app/workbench.py`, `app/agent/router.py`, `app/retrieval/hybrid.py` |
| Index/cache/version work | `app/indexing/`, `tests/test_incremental_indexing.py`, `tests/test_git_versioned.py`, `tests/test_corpus_cache.py` |
| Research decisions | `experiments/C3_cycle/`, `experiments/RESULTS.md`, `experiments/STRESS_TEST.md`, `experiments/HANDOFF.md` |

The demo router, BM25, graph and version retrieval do not produce the submitted
MTEB score. Distinguish implemented, candidate-enabled, submission-enabled, and
measured gain. `config/frozen.yaml` stays immutable; candidate behavior is not a
new submitted score.

## Development commands

Use the project virtual environment (Python 3.11/3.12 matches the CI matrix).
See `docs/CODEX-WORKFLOW.md` for setup and platform differences. From repo root:

```sh
.venv/bin/python -m pytest -q -rs -m "not model and not data"
.venv/bin/python -m ruff check app scripts tests --select F401,F811
.venv/bin/python -m compileall -q app scripts tests
make -B
make -B test
.venv/bin/python scripts/evaluate.py scripts/sample_eval.jsonl --output build/sample_results.json
.venv/bin/python scripts/validate_submission.py build/sample_results.json
```

Run checks appropriate to the change. These are the existing CI selection, not
permission to weaken or skip failing tests. Report failures and skip/deselection
counts. Registry tests require the separate evaluation stack. Bare `pytest`
includes model/data tests; it is not the onboarding smoke command. A validated
result JSON establishes format validity, not reproduction of its score.

## Research execution

- Maintain `experiments/REGISTER.md` after each experiment attempt or decision.
  Link its evidence, distinguish candidate/submitted state, and record cleanup
  candidates without deleting shared code or reproduction evidence.

- Before a decision experiment, use the `codedoctor-experiment` skill and the
  committed preregistration. Routine lint/tests do not need preregistration.
- APPS test access and final submission decisions retain the explicit team
  decision boundary in `CLAUDE.md`; setup is not authorization for those runs.
- Inspect loaders before accessing data. `get_train_split()` currently reaches
  a loader that reads test files too; the method-level firewall is not complete
  I/O isolation. MTEB's audit log is best-effort and written after execution.
- Keep EXT-lockbox sealed; reused EXT-dev is descriptive/veto evidence as the
  current preregistration specifies, not fresh independent confirmation.
- Capture raw output, command, exit status, config/artifact hashes, split/seed,
  model revision, environment and hardware. `scripts/research_manifest.py`
  captures a sidecar snapshot; it does not execute or certify an experiment.
- One writer per output. Run timing and wall-timeout experiments without
  competing heavy jobs. Worktrees isolate source changes, not RAM or GPU use.
- A subagent review is useful but is not an independent human annotation or
  clean-room reproduction. Label the actual degree of independence.

## Completion

Report the change, relevant checks and raw evidence paths, unresolved limits,
and the next concrete action. Use `codedoctor-handoff` at research milestones.
Keep long results in experiment records rather than expanding these rules.
