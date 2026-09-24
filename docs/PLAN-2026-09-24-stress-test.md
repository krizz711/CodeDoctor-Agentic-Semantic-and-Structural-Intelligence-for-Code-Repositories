# Stress test of the method, then the execution round it implies

## Context
You approved the reviewer's CLAUDE.md fixes, switching the dense gate off, and commit + push. The conditions: double-check everything, run a methodological stress test first, and treat `C:\Users\Asus\manish` plus GitHub as the only workspace.

The stress test below was run read-only on the recorded stores, the papers and the result files. It found errors in our own claims and fragile results, and it changes several steps of the execution round. The previous plan is preserved in the repo as `docs/V2-ARCHITECTURE-PLAN.md`.

Still running in the background: the 0.6B re-record for R2 (DEV done, TUNE running) and the Kaggle CPU latency benchmark.

## Stress-test findings (V = verified here, J = judgment)

**F1. Contamination claims: wrong or unsupported (V).**
- **What the F2LLM-v2 paper lists.** Its training table (arXiv 2603.19223, Table 8, checked in the PDF text) lists no APPS and no CodeContests. It does list OpenCodeReasoning-2 (16,632 samples) and xCodeEval NL2Code (51,072 Codeforces pairs).
- **Where OpenCodeReasoning-2's problems come from.** Its dataset card says "TACO, APPS, CodeContests" (train and valid splits), excluding CodeContests test.
- **README.** It attributes APPS training to F2LLM via the C2LLM report. The honest wording: APPS and CodeContests (train/valid) exposure is plausible through OpenCodeReasoning-2; per-problem exposure is unverified.
- **My "uncontaminated CodeContests EXT" claim is wrong.** 47 of the 108 EXT-dev statements come from the valid split.
- **RESULTS.md's "UniXcoder predates APPS" is false** (APPS May 2021, UniXcoder March 2022). The correct basis: pre-trained on C4 and CodeSearchNet (paper §4), with no APPS.
- **No contamination signal in the data:** valid-split problems score *lower* on dense than test-split ones (1.7B 0.701 vs 0.736).

**F2. Effects rest on very few statements (V).**
- **How few:** call-style vs the submitted rule changes 23 of 2,500 DEV statements; deepening + gate vs call-style changes 6 (DEV) and 10 (TUNE).
- **Why it matters:** a percentile bootstrap is unreliable at this sparsity. R1's "TUNE deepening + gate +0.0018, interval excludes zero" gets sign-flip permutation p = 0.061. **It is fragile.**
- **What holds:** bootstrap bounds are stable across seeds (±0.00005).

**F3. Split sensitivity: 200 random re-splits of the 5,000 statements at 1.7B (V).**
| Change | Stays positive | Meets +0.003 | Non-inferior (> −0.002) |
|---|---|---|---|
| Call-style | 100% | 86% | — |
| Gate alone | — | — | 82% (not robust) |
| Deepening ± gate | — | — | 98–100% |

**F4. The gate (V + J).**
- **Verified:** its gold exclusion is dataset-dependent. It keeps the correct solution out of execution for 4 of 107 CodeContests statements vs 1–3 of ~1,450 APPS statements (Fisher p = 1e-4 and 6e-4).
- **Verified:** the NDCG harm rests on 2–4 statements, all in the CodeContests test split.
- **Stated plainly:** at 1.7B the gate passes the *letter* of R1's pre-registered EXT rule (interval spans zero).
- **Judgment:** "gate off" therefore rests on the mechanism, the 0.6B result, FAQ 4 (don't overfit APPS) and F3. It will be recorded as a post-hoc decision.

**F5. Construct validity (V + estimate).**
- APPS test is ~99% stdin statements (0.9% call-style, from the round-1 statement survey).
- At 1.7B the v2 changes move the stdin family by 0.0000 to −0.0003 on DEV.
- **Estimate:** a test-split run of `candidate_v2` gains about 0 on the screening number. v2's value is on call-style problems and weaker first passes. The one test-split run should not be spent expecting a higher score.

**F6. EXT-dev has been used adaptively (V).** It fed E3, E8, R1 and now R2, so it is no longer pristine. EXT-lockbox is untouched and is the only valid final generalisation check.

**F7. The parser failure is robust (V).**
- Leaving out HackerEarth (whose sample-test labels are malformed): precision 0.919, still below the 0.95 bar.
- Errors concentrate in Codeforces (0.885, n = 980) and CodeChef (0.847).
- The call-style oracle is uninformative: APPS's official tests often don't contain the statement's examples.

**F8. Measurement gaps (V).**
- Stores don't record run durations, so timeout sensitivity can't be measured.
- Replay has been checked against live runs only for the legacy ranking (150/150) and execution counts (30/30). The call-style and deepening rankings have never been checked against live runs.

**F9. Process (V).** Pre-registrations and addenda are uncommitted, so their timestamps can't be verified. Some addenda were written after data was recorded, though before it was replayed.

**F10. Robust (V).**
- Deduplication: 107/108 EXT statements have max Jaccard < 0.1 against APPS.
- Negative control: parser matcher 0.0000–0.0002.
- Pool sanity and headroom checks pass.
- The 1.7B vectors reproduce on Kaggle (min cosine 0.99996).

## Execution round (in order; each step has a check that must pass)

**1. Put the stress test in the workspace.**
- New `experiments/STRESS_TEST.md` with F1–F10, the commands used, and the analysis script as `scripts/stress_test.py` (the read-only code run above).
- Copy `theme1_guidelines.pdf` into `C:\Users\Asus\manish\reference\`, so the workspace is self-contained. It is not committed; it's the organisers' document.

**2. Verify every reviewer claim before acting**, marking each one confirmed or refuted in `STRESS_TEST.md`:
- `evaluate_mteb.py:49-55` swallows config errors.
- A spawn failure silently degrades ranking to dense-only.
- `resolve_workers` uses logical cores.
- Pins, missing tests, `os.replace` usage, the `eval` in the runner.
- The contradictions (lines 16/52, README:309, CPU-time vs frozen wall-clock).

**3. CLAUDE.md** (repo file = the single source; the workspace file stays imports-only, since sessions start in `manish`):
- Engineering rules split into "Enforced now (file:line)" and "Target (not yet true)".
- Resolve the test-split wording: one run per frozen config, point to README:309 for the two past test-split choices.
- State that the frozen config uses wall-clock timeouts as a known exception.
- Define "run", "decision experiment" and "final decision".
- Precedence: user's chat instruction > judge's FAQ > brief > these rules.
- Name the runner's `eval` as an exception.
- Label the 1.7B memory claim as an estimate.
- Stay at or under 120 lines; the FAQ stays byte-identical (re-checked).
- Duplicate loading: kept as a known, unverified cost (the text is identical, so there's no contradiction). It's checked in the next session by watching for on-demand "Contents of CodeDoctor/CLAUDE.md" reminders.

**4. Robustness fixes on the scored path** (behaviour changes only in error cases; a regression test first, each failing before its fix):
- **Config parse error:** fail loud.
- **Spawn failures:** counted in the audit, with a warning printed when more than 0 occur.
- **`resolve_workers` default:** physical cores (`psutil.cpu_count(logical=False)`, falling back to `os.cpu_count()`).
- **Outcome stores:** record per-run elapsed time (closes F8's timeout gap for future runs).

**5. Gate off.** `config/candidate_v2.yaml`: `execution_max_dense_gap: null`. Update `tests/test_candidate_config.py` with a comment pointing to F4. `frozen.yaml` is untouched.

**6. Docs corrections:**
- **README:** the contamination paragraph (F1), the gate claim, the v2 table.
- **RESULTS.md:**
  - The UniXcoder basis, and "partly protected" instead of "uncontaminated".
  - Round-1 execution counts corrected, with the definition "candidates that start ≥ 1 sandbox process", validated 30/30.
  - New sections: R1 at 1.7B, V3-oracle, the stress-test summary.
  - Lesson 6 updated.
- **HANDOFF:** mark the jury item closed; parser labels are now external.
- **R1/R2 pre-registrations:** correction addenda (F1, F4).

**7. R2 (encoder choice)**, before replaying the re-recorded 0.6B stores:
- **Add to the R2 pre-registration:**
  - The paired sign-flip permutation test is reported with the bootstrap.
  - Non-inferiority also requires ≥ 95% of 200 re-splits to be non-inferior.
  - The number of changed statements is reported.
  - Pipeline = call-style + deepening to 50, **no gate** (step 5).
- **Then:** the confound check on the `_r2` stores (0 differing statements expected), the replays, the Kaggle CPU latency results (with their benchmarks: known number, fidelity ≥ 0.999, stop rule), and the fixed-sequence decision.
- **If 0.6B fails the margin,** the quantified trade-off comes to you.

**8. Replay fidelity for the policies that matter.** Live vs replay top-10 on 150 DEV statements for call-style, and for call-style + deepening. Bar: ≥ 149/150. If it fails, the R1/R2 numbers for that policy are void.

**9. Parser (separate pre-registration, after the above):**
- Fix only defects seen on DIAG: spec prose taken as a sample, multi-sample mis-splits, interactive statements, plus the Codeforces and CodeChef layouts. Each fix gets a regression test.
- Judge once on CONF, reporting precision/recall per source.
- If precision improves, the retrieval effect needs re-recorded stores; that's measured in a later round.

**10. Commit and push** to `v2/prepost-architecture` in logical commits (code + tests; experiments + stores; docs + CLAUDE.md + .gitignore), with the attribution trailer. Verify with `git ls-remote`. `main` stays untouched. `CLAUDE.local.md`, `artifacts/` and `build/` are not committed.

## Verification
- Full `pytest` on Windows. Sandbox tests in WSL (Linux-only ones). `ruff check app scripts tests --select F401,F811`.
- New regression tests fail before their fixes and pass after. `tests/test_candidate_config.py` is updated deliberately, not loosened.
- FAQ byte check against the pasted text. CLAUDE.md line count.
- Replay fidelity ≥ 149/150 (step 8). Oracle unit tests plus the negative control.
- `git ls-remote` equals local HEAD; the three commits appear on GitHub.

## Critical files
- `CodeDoctor/CLAUDE.md`
- `config/candidate_v2.yaml`, `tests/test_candidate_config.py`
- `scripts/evaluate_mteb.py` (config load, `resolve_workers`), `app/pipeline/execution_rerank.py` (spawn-failure accounting), `scripts/record_outcomes.py` (elapsed time)
- `README.md`, `experiments/RESULTS.md`, `experiments/HANDOFF.md`, `experiments/R1_17b/prereg.md`, `experiments/R2_encoder_choice/prereg.md`
- New: `experiments/STRESS_TEST.md`, `scripts/stress_test.py`
- Reused: `scripts/replay_outcomes.py` (`POLICIES`, `rerank`, `metrics`, `executions`), `scripts/parser_oracle.py`, `scripts/bench_query_cpu.py`
