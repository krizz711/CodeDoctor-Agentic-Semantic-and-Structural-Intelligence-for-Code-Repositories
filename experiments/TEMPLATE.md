# Experiment record template

Copy the preregistration section into a new experiment directory. Replace every
prompt with concrete values before committing. This file is a template, not a
preregistered experiment or permission to run one. Existing protocols/addenda
remain authoritative for their experiments.

## Preregistration

- Question and falsifiable hypothesis:
- Current code/branch and baseline configuration:
- Relevant jury FAQ answers (verbatim) and agreement; silence where applicable:
- Data sources/revisions, split assignment and seed; prior exposure/reuse:
- Loader audit: exactly which files/splits can be read:
- Intervention/arms and which parameter is fitted on which split:
- Primary metric, minimum important effect and non-inferiority margin:
- Pairing unit, bootstrap resamples, sign-flip/permutation method, alpha,
  multiplicity family/correction and changed-statement counts:
- Applicable existing ship/kill rule; any justified new rule set before data:
- Known-answer baseline control, negative control, mechanism check:
- Ceiling/power check and conditions that make a run void:
- Hardware, Python/packages, encoder revision, config/cache hashes, workers:
- Exact commands, output paths, expected runtime/resources (label estimates):
- Runtime/compute budget, stop rule, allowed retry conditions/count:
- Strongest alternative, pre-mortem, leakage/contamination/timing objections:
- Cheapest falsifying check and independent validation plan/limitations:
- Commit recording this protocol before first decision-data access:

Keep APPS test and EXT-lockbox outside ordinary development. For reused
EXT-dev, explicitly state its reduced evidential role. Do not transfer a
machine-specific timing protocol without preregistering the change.

## Result (write only after execution)

- Protocol commit; implementation commit and dirty changes:
- Run time and machine; actual command, exit code, elapsed time:
- Data/model/config/cache identifiers; manifest and raw-log paths:
- Controls: pass/fail and evidence; valid/void status:
- Every arm, per-family/bucket metrics, paired deltas/CIs/p-values/correction:
- Changed statements and mechanism evidence:
- Quality/cost decision against the original rule:
- Deviations and post hoc observations, labelled separately:
- Adversarial objections and resolutions; unverified claims:
- Implemented / candidate-enabled / submitted status:
- Next step, exact command if settled, remaining authorization or hardware need:

Report negative and null results. A failed sanity control makes the run void;
its outputs do not support a ship decision. A stored score is not a new run.
