# Working on CodeDoctor with Codex

Open the **codedoctor repository directory** as the local Codex project. A
GitHub connection provides remote access; it does not select a local checkout.
Start a new task in that project after installing these instructions/skills.
Ask it to name the branch, working directory and instruction files it loaded.

`AGENTS.md` loads the shared `CLAUDE.md` policy and routes to focused context.
Keep using Claude Code if useful; both tools should follow the same research
policy. Do not replace the verbatim FAQ with a summary.

## Environment

Python 3.11 or 3.12 matches CI. On macOS, for example:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt torch ruff
```

On Linux, install PyTorch's CPU wheel separately, as CI does:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt ruff
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\python.exe` in
place of `.venv/bin/python`. No activation is required when using these paths.
Each worktree needs its own environment. Do not copy an existing venv between
paths. Install `requirements-eval.txt` only when a task needs the evaluation
stack; it is not required for the default development check.

The 2026-09-26 local setup uses the app's bundled Python 3.12 interpreter because
the system `python3` is 3.14. The checkout's `.venv` is ready to use; a fresh
checkout still needs an available 3.11/3.12 interpreter. Machine-specific logs
and package versions are in ignored `build/codex-setup/`.

## Daily commands

```sh
# Existing CI selection: no weight-loading or real-dataset tests.
.venv/bin/python -m pytest -q -rs -m "not model and not data"
.venv/bin/python -m ruff check app scripts tests --select F401,F811
.venv/bin/python -m compileall -q app scripts tests
make -B
make -B test

# Small local harness: supplied sample rows, not AppsRetrieval evaluation.
.venv/bin/python scripts/evaluate.py scripts/sample_eval.jsonl --output build/sample_results.json
.venv/bin/python scripts/validate_submission.py build/sample_results.json

# Start the demo when needed. Dense queries can load/download model weights.
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8077
```

The five MTEB-registry tests may skip without the evaluation dependencies; report
that separately from deliberately deselected `model`/`data` tests. Passing the
development selection does not establish a reproduced benchmark or full CI on
every OS. Stop model-serving processes before memory-heavy tests or timing runs.

For desktop shortcuts, use the app's Local environments settings to add a
**Test** action and a **Run demo** action with the commands above. Let the app
generate its environment file; do not invent a configuration schema. These
instructions do not enable unattended downloads, recurring jobs, or paid runs.

## Research loop

1. Inspect the relevant implementation and `docs/CODEX-ANALYSIS.md`.
2. Use `$codedoctor-paper-to-experiment` for literature-to-code work, or
   `$codedoctor-experiment` for a concrete hypothesis.
3. Adapt `experiments/TEMPLATE.md` into an experiment-specific preregistration.
   Follow the existing applicable protocol if continuing an experiment.
4. Commit a decision protocol before its data is replayed. Record the commit;
   do not relabel a post hoc protocol as preregistered. Routine code checks do
   not need this ceremony.
5. Implement a bounded increment, validate it, and run only the agreed budget.
   Preserve raw output and actual exit status.
6. Interpret all arms, nulls, failed controls, multiplicity and deviations.
   Use `$codedoctor-handoff` to preserve the state and next action.

A useful first prompt:

> Read AGENTS.md, CLAUDE.md and docs/CODEX-ANALYSIS.md. Verify the current branch
> and latest C3 addenda. Propose a Mac-specific P1 incremental-index experiment
> with a fixed baseline, controls, resource budget and stop rule. Preserve the
> original laptop protocol. Finish with a preregistration draft; do not load
> weights or access benchmark data yet.

## Reproduction sidecars

`scripts/research_manifest.py` captures the current Git state, interpreter,
installed distributions, available hardware information, and hashes of explicitly
provided files. It uses only the standard library and never runs an experiment.

After the sample harness above, for example:

```sh
.venv/bin/python scripts/research_manifest.py \
  --artifact scripts/sample_eval.jsonl \
  --artifact build/sample_results.json \
  --command '.venv/bin/python scripts/evaluate.py scripts/sample_eval.jsonl --output build/sample_results.json' \
  --output build/sample-manifest.json
```

For a real run, also supply `--config` and relevant `--artifact` inputs/outputs,
plus `--seed` when the experiment uses one. Commands and seeds are declarations,
not execution evidence. Collect near execution and preserve the raw log, actual
exit status, elapsed time, dataset split/revision, model revision, threads and
budget in the result record. A later snapshot cannot prove an earlier run's
environment. Existing sidecars are never overwritten; use a new output name.

## Parallel work and durable context

Keep one task per outcome. Ask subagents for bounded reading, test design or
review. Use separate worktrees for concurrent code changes and separate output
directories for experiments. Do not run timing jobs concurrently on shared
hardware. An extra AI reviewer is not independent human validation.

`AGENTS.md` holds stable rules; experiment records hold evidence and decisions;
skills hold repeatable procedures. Avoid copying entire papers or all history
into instructions. Global communication defaults can live in
`~/.codex/AGENTS.md`; project-specific policies belong here.

## Setup verification

Verified locally for this setup on macOS arm64 with Python 3.12.14, based on
source commit `9982be8`, plus the new manifest helper/tests and documentation:

| Check | Observed result |
| --- | --- |
| Development pytest selection | 485 passed, 13 skipped, 6 deselected; 5 subtests passed |
| New manifest helper | Its 9 tests pass within that selection |
| Ruff F401/F811 | Passed |
| Compile app/scripts/tests | Passed |
| C++ build and core test | Passed |
| Sample evaluation + format validation | Passed; sample data only |
| Existing submitted JSON format validation | Passed; score not reproduced |
| Three skill validators | Passed |

The 13 skips comprise five MTEB registry checks, one optional Transformers
check, one optional psutil check, and six platform-specific checks. The six
model/data tests were deliberately excluded by the existing CI selector.
Three warnings were emitted: Starlette/httpx deprecation, an expected simulated
sandbox spawn failure, and a pytest fixture deprecation. No tests were modified
to suppress these outcomes.

Local pytest/build logs, installed package versions and a manifest are in
`build/codex-setup/` (ignored, not shipped with a fresh clone). These results do
not replace Linux/Windows CI, model integration tests, or a decision experiment.

Product references: [agent instructions](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[skills](https://learn.chatgpt.com/docs/build-skills),
[local environments](https://learn.chatgpt.com/docs/environments/local-environment),
[worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees).
