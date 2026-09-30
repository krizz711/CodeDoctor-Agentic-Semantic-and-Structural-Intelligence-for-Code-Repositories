---
name: codedoctor-experiment
description: Plan, execute, or review a CodeDoctor retrieval experiment using its preregistration, data boundaries, controls and recorded evidence. Use for research decisions or benchmark comparisons, not routine lint or unit tests.
---

# CodeDoctor experiment workflow

Read repository `AGENTS.md` and `CLAUDE.md`, then only the current experiment's
preregistration/addenda and relevant implementation. Resolve paths from the
repository root. `docs/CODEX-ANALYSIS.md` is a dated map, not live status.

Determine whether this is a new decision experiment, a continuation, or a
review of existing results. Preserve the user's requested scope.

For a new decision experiment, adapt `experiments/TEMPLATE.md`: specify the
baseline, mechanism, split exposure, minimum effect, paired statistics and
multiplicity, controls, resource budget and stop/retry rule. Quote relevant
FAQ answers from `CLAUDE.md`. Commit before decision data is replayed; if a
commit or required resource is unavailable, deliver a clearly labelled draft
and continue independent preparation rather than claiming preregistration.

For continuation, read all addenda before running. In C3, E2's first timing run
is void with one permitted unchanged repeat; E3 targets a specific laptop.
Recheck current records before relying on either statement. Do not silently
move a machine-specific timing protocol onto a different host.

Inspect data-loading code, not just the requested split label. The current
development harness can read test files while loading train. Keep APPS test
and sealed lockboxes out of ordinary development. Explicit final-run/team
authorization in the shared policy is required when applicable.

Implement behind the existing default-off mechanism when changing retrieval.
Leave `config/frozen.yaml` untouched. Run routine correctness checks first;
then execute only the settled protocol with one writer per artifact and no
competing timing jobs. Stop when controls fail or the budget/stop rule is met.

Save raw logs, actual command/exit/time, per-query data and all arms. Use
`scripts/research_manifest.py` for explicitly selected config/artifact hashes
and an environment snapshot; it does not prove that a command ran. Record
dataset/model revisions, split/seed, threads and cost alongside the sidecar.

Review against the original rule, including null/negative results and the
count of changed statements. Reused EXT-dev is not fresh replication. A
subagent is not an independent human annotator. Separate measured findings,
post hoc hypotheses and unverified facts. State whether the change is merely
implemented, candidate-enabled, or submitted. End with evidence paths and the
smallest justified next action.
