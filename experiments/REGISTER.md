# Experiment register

Last reviewed: 2026-09-29. Branch: `v3`, base `7d64a80`.
This is the current index of research outcomes and future cleanup candidates.
Evidence below is repository-reported; creating this register did not rerun experiments.
The protocols and raw results remain authoritative. Historical uses of “ships”
mean candidate adoption unless a submitted result explicitly establishes otherwise.

## How to maintain this register

Before starting an idea, check its row and linked evidence. Add a row for new
work and update it after each attempt or decision, including negative, null and
void outcomes. Use cycle-qualified IDs: round-1 E2 and C3-E2 are different work.
Record the model/data operating point, evidence, implementation state and next
action. Update the relevant cycle summary and handoff when a blocker closes.
A pending run is never a failure; a void run is never valid outcome evidence.

Cleanup labels are proposals, not deletion approvals:
- **Keep:** active behaviour, controls, regression tests or reusable infrastructure.
- **Review removal:** rejected/disabled experimental implementation; inspect callers first.
- **Archive:** one-off probes and superseded scripts after preserving reproduction evidence.
- **Hold:** unresolved work; do not prune its dependencies.

Always retain the decision, protocol, deviations, raw evidence (or a durable
verified archive), and reproduction recipe for negative as well as positive results.

## Adopted changes and useful validation

| ID / idea | Outcome and limits | Current implementation / cleanup | Evidence |
| --- | --- | --- | --- |
| R1-E1 call-style execution | Positive: 1.7B DEV +0.0041, TUNE +0.0032; gain in call-style families, not stdin | Candidate on; Keep | [R1 protocol](R1_17b/prereg.md), [results](RESULTS.md), `execution_call_style` in [config](../config/candidate_v2.yaml) |
| R1-POST7 adaptive deepening | Positive at 0.6B and UniXcoder; neutral at 1.7B (DEV +0.0007); retained for weaker retrieval | Candidate depth 50; Keep, revisit cost/benefit before merge | [results](RESULTS.md), [config](../config/candidate_v2.yaml) |
| C3-E1 blank-line fallback | Adopted: 1.7B DEV stdin +0.0078, 11 up/1 down; all-family +0.0028; no submitted gain established | Candidate on; Keep | [result](C3_cycle/E1_result.md), [protocol](C3_cycle/PREREG.md) |
| C3-E1B judge-like I/O | Accepted fidelity fix; DEV zero ranking changes; external result descriptive | Candidate on; Keep tests and runner support | [summary](C3_cycle/SUMMARY.md), [protocol](C3_cycle/PREREG.md) |
| R1-E3 parser heading fixes | Correctness: avoid parsing specification prose as examples; 40/217 CodeContests statements affected in report | Implemented; Keep regression tests; no isolated submitted gain claim | [results](RESULTS.md) |
| V3b brace arrays / interactive statements | Correctness fixes accepted; no retrieval gain claimed; fix plan was not committed before rerun | Brace parsing implemented; skip-interactive candidate on; Keep | [fix plan and outcomes](V3_parser_gold/prereg_fixes.md), [raw result](V3_parser_gold/result_oracle_after_fixes.json) |
| INFRA sandbox / parallelism / CPU time / corpus cache | Engineering validation reported; CPU limits reduce load sensitivity; sandbox is not filesystem/network isolation | Implemented; candidate CPU-time mode; Keep | [results](RESULTS.md), [policy](../CLAUDE.md), [config](../config/candidate_v2.yaml) |
| C3-E3 parts 1–2 cache recipe / rebuild | Correctness fixes implemented; recipe-aware keys and in-process rebuild; timing claim still pending | Keep implementation and regression tests | [handoff A3–A4](HANDOFF.md), [protocol](C3_cycle/PREREG.md) |
| R1-E7 UniXcoder control | Original prediction of larger absolute verification gain failed; exposed first-pass recall limit; deepening helped | Keep benchmark evidence; Archive one-off runs after dependency review | [results](RESULTS.md), [protocol](E2_E7_prereg.md) |
| C3-P0 small-pool ceiling | Report attributes low execution gain on CC-onesol to limited headroom; APPS small-pool control supports it | Completed diagnostic; Keep evidence, Archive probe when reproducible | [result](P0_ceiling_anomaly/result.md), [protocol](P0_ceiling_anomaly/prereg.md) |

## Rejected, disabled or limited findings

| ID / idea | Outcome and scope | Implementation / cleanup | Evidence |
| --- | --- | --- | --- |
| R1-E1 judge-aware comparator | Below minimum effect: +0.0021 at 0.6B, +0.0004 at 1.7B | Candidate off; Review removal of optional policy, preserve shared comparison code | [results](RESULTS.md), [config](../config/candidate_v2.yaml) |
| R1-E1 input consumption / static pruning | Weak signal; pruning would discard 16 golds for small run savings | Rejected as standalone policies; consumption is used by accepted blank fallback, so preserve that support | [results](RESULTS.md), [C3 protocol](C3_cycle/PREREG.md) |
| R1-E2 160M → 0.6B cascade | Rejected: −0.0076, 62% escalation, 11% encoding savings | No candidate cascade; Review removal of dedicated simulator after archiving | [results](RESULTS.md), [protocol](E2_E7_prereg.md) |
| R1-E4 learned fusion | Original −0.0021; later within-tier variant about +0.002, below bar and post hoc | Not adopted; Review removal of dedicated fitting path | [results](RESULTS.md), [raw result](E4_fusion/result.json) |
| R1-E5 version / lineage ranking | 0.953 correct-version-first on mutants selected to fail examples; optimistic construction, not general near-miss validation | In code; Hold pending Bonus scope; preserve lineage features needed elsewhere | [results](RESULTS.md), [raw result](E5_versions/result.json) |
| R1-E6 hubness | Dense gains; failed convincing replication with execution on TUNE | Off / not configured; Review removal of dedicated option after caller audit | [results](RESULTS.md), [raw result](E6_hubness/result.json) |
| R1-E8 metamorphic amplification | Too small: APPS about +0.0002; weak against real near-misses | Not adopted; Archive experimental work, retain negative finding | [results](RESULTS.md) |
| R1 dense gap gate 0.20 | Earlier cost win superseded by external gold-exclusion failures; 1.7B DEV non-inferiority failed | Candidate gate off; frozen path may depend on it, so Hold until submission compatibility review | [stress test F4](STRESS_TEST.md), [config](../config/candidate_v2.yaml) |
| R2 smaller production encoder | 0.6B fails non-inferiority (DEV −0.0169); 160M decision arm not run; lower CPU cost is a real tradeoff | Production stays 1.7B; 0.6B still serves demo, Keep both uses | [protocol/outcome](R2_encoder_choice/prereg.md) |
| R2 CPU bfloat16 | Fidelity failed; CPU timing also unfavorable on measured machine | Not adopted at that operating point; Archive probe | [results](RESULTS.md), [R2 record](R2_encoder_choice/prereg.md) |
| V3 parser oracle | Original precision threshold failed; comparison imperfect for multiple valid answers; later reinterpretation does not replace original verdict | Keep matcher, tests, result audits; human labels remain outstanding | [protocol](V3_parser_gold/prereg_oracle.md), [fix analysis](V3_parser_gold/prereg_fixes.md) |
| C3 small-to-large query mapping | Map DEV 0.896 below native 0.6B 0.901; fused result still below 1.7B | Research rejection; Archive dedicated probes | [research reports](C3_cycle/research_reports.json), [summary](C3_cycle/SUMMARY.md) |
| C3 torch dynamic int8 / int4 probes | Tested variants damaged vector fidelity; does not reject every quantisation method | Research rejection; Archive probes; Hold OpenVINO dependencies for C3-E2 | [research reports](C3_cycle/research_reports.json) |
| C3 0.6B → 1.7B cascade | 56–57% escalation; only 7–9% estimated encoding savings overall | Not selected; Archive probes; distinct from round-1 cascade | [research reports](C3_cycle/research_reports.json) |
| C3 dense PRF | Worse at tested TUNE settings | Rejected; Archive dedicated probe | [summary](C3_cycle/SUMMARY.md), [research reports](C3_cycle/research_reports.json) |
| C3 crash demotion / perturbation tie-break / identifier bonus | Harmful or null in reported probes | Rejected; Archive probes, not production identifier search | [summary](C3_cycle/SUMMARY.md), [research reports](C3_cycle/research_reports.json) |
| PRE4 offline spec cards | Rejected during literature/applicability review; no local full experiment establishes ineffectiveness | Not implemented; Archive proposal, reconsider only with changed assumptions | [C3 protocol](C3_cycle/PREREG.md), [research reports](C3_cycle/research_reports.json) |

## Pending, deferred and untested

| ID / item | Last evidenced state | Next action / cleanup | Evidence |
| --- | --- | --- | --- |
| C3-E2 OpenVINO int8 | Both attempts void under S2 on Intel Xeon; downloaded repeat fails its REF sanity band in both arms. Output version inferred from handoff/latest-run metadata; [raw repeat](C3_cycle/E2_runs/e2_A_run2_downloaded.json), [assessment](C3_cycle/E2_RECONCILIATION.md) | Stopped under A3; no third repeat, no adoption; preserve raw evidence | [handoff](HANDOFF.md), [protocol A1–A3](C3_cycle/PREREG.md) |
| C4-E1 llama.cpp Q8_0 query encoder | [Protocol committed](C4_q8_query/PREREG.md); train-only 2,500/2,500 inputs prepared; first synthetic conversion smoke failed on Kaggle DNS before model work. No decision run or adoption. | One documented infrastructure retry remains after network correction. Stage-1 screen next, then stage-2 design pause. Preserve infra failure. | [setup/exposure](C4_q8_query/ADDENDUM_A1.md), [raw smoke](C4_q8_query/runs/smoke_v1_infrastructure_failure.log) |
| C3-E3 part 3 P1 timing | Not run in current records | Hold. Machine-change addendum and local baseline before timed runs; handoff A4/B2 | [handoff](HANDOFF.md), [protocol](C3_cycle/PREREG.md) |
| NEXT-MBR execution agreement | Pilot 0.46 vs chance 0.33, n=33; no confirmation or adoption | Hold pilot; preregister a new cycle before decision run | [summary](C3_cycle/SUMMARY.md), [pilot](C3_cycle/research_scripts/pilot_nearmiss_result.json) |
| PRE2 prose-only encoder input | Proposed; no completed decision result identified | Deferred; check probes before new proposal | [historical plan](../docs/V2-ARCHITECTURE-PLAN.md), [research reports](C3_cycle/research_reports.json) |
| PRE3 repaired body-only documents | Not implemented in current results | Deferred; no success/failure verdict | [results](RESULTS.md) |
| INFRA2 factorial study | Not run as designed; some combinations measured separately | Preserve proposal as unexecuted; no factorial attribution claims | [results](RESULTS.md) |
| V3 human labels | 207-row sheet prepared; external references substituted; no completed independent human set established | Hold sheet | [plan](V3_parser_gold/prereg_oracle.md), [sheet](V3_parser_gold/to_label.jsonl) |
| V5 clean-room / V8 independent red team | Open; agent reviews are not independent human validation | Hold required reproduction tooling | [handoff](HANDOFF.md) |
| Markdown normaliser / disclosure / README | Deferred or awaiting update according to handoff; documentation work, not measured experiments | Track in handoff; do not mark rejected | [handoff A4/B4](HANDOFF.md) |
| Final lockbox and APPS test report | Not spent for the new candidate in current records | Keep sealed until authorized freeze/reporting decision | [handoff](HANDOFF.md), [policy](../CLAUDE.md) |

## Before pruning for main

1. Close or explicitly defer pending rows; establish the exact candidate to merge.
2. Trace imports, CLI use, configurations, tests and reproduction commands for
   every proposed removal. A failed experiment can share code with a successful one.
3. Preserve frozen submission compatibility unless an explicit migration is approved.
4. Archive one-off research/skeptic probes, old queue scripts and redundant large
   artifacts only with a durable evidence location and verified reproduction inputs.
5. Keep negative-result summaries and protocols discoverable here. Do not delete
   tests merely because their original experiment failed; assess current coverage.
6. Record actual removed paths and archive locations in the corresponding rows,
   run affected checks, and review the final diff before merging.

No files have been designated safe to delete solely by this register.
