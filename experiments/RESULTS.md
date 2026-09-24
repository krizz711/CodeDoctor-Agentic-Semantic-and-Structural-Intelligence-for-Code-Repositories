# Results — CodeDoctor v2 pre/post-processing (round 1)

All numbers below were produced by the commands in `experiments/README.md`, on
the data tiers defined there. Pre-registrations: `E1_verification_policies/prereg.md`,
`E2_E7_prereg.md`. **The APPS test split was not used.** Pools come from
F2LLM-v2-0.6B encoded on this machine (RTX 2050), because the shipped 1.7B
encoder does not fit; the local 0.6B artifacts reproduce the published number
(train NDCG@10 0.90780 vs 0.90741).

Statistics everywhere: paired bootstrap over statements, 2,000 resamples, 95%
percentile interval. Ship rule: interval excludes zero after Holm correction,
effect ≥ +0.003 NDCG@10, no family regresses, mechanism confirmed.

> **Round-2 update (2026-09-24).** Everything below was re-measured with the shipped 1.7B encoder
> (section R1) and stress-tested ([`STRESS_TEST.md`](STRESS_TEST.md)). The stress test corrected
> several claims in this file: the execution counts, "UniXcoder predates APPS" (false), and
> "uncontaminated CodeContests" (only partly protected). It also switched the dense gate off.
> Corrections are marked **[corrected]** where they appear.

## What the round taught us

Interpretation of the measurements below; each point names the sections it rests on.

1. **Adding evidence worked; re-reading evidence did not.** Both changes that
   passed give the verifier evidence it did not have: executing call-style
   problems (+0.0071 DEV, +0.0053 TUNE) and executing deeper when the top 20
   found nothing (+0.0033 / +0.0026 with the gate). Every change that re-read
   evidence already collected stayed under the bar or went negative:
   judge-aware comparison +0.0021, learned fusion −0.0021, metamorphic
   amplification +0.0002, input consumption (likelihood ratio ≈ 1.2).
   (E1, E4, E8, final replications.)
2. **Post-processing is capped by first-pass recall.** Execution can only
   promote a gold that the dense pass retrieved. With UniXcoder the gold is in
   the top 20 for 18% of statements, and the absolute gain was *smaller* than
   with F2LLM — the opposite of the pre-registered prediction. Deepening was
   worth 2.5× more there. If the unseen dataset is harder for the encoder,
   recall matters more than rerank rules. (E7.)
3. **Execution separates problems, not solutions.** It tells the right problem
   from a wrong one well (promotion precision 0.78–0.87), but barely separates
   a correct solution from a near-miss to the same problem: near-misses pass
   the public examples 43.3% of the time, correct solutions 51.4%. That is the
   open frontier, and it bounds the Bonus result (0.953 was measured on mutants
   chosen to fail the examples). (E1, E3, E5.)
4. **Held-out data finds bugs APPS hides.** The Codeforces `Input`/`Output`
   heading defect hit 40 of 217 CodeContests statements and no APPS statement.
   The parser is the component most exposed to an unseen dataset, and its
   precision was unmeasured then. **[corrected]** Measured since against external
   labels, it is 0.910 on CodeContests and ≥ 0.938 on APPS: below the 0.95 bar
   (section V3-oracle). (E3.)
5. **Three measurement hazards could each have produced a false result:**
   load-dependent wall-clock timeouts (24/24 vs 7/24), benchmark structure
   inside a hubness bank (~0.0025 of +0.0121), and a concurrent writer
   corrupting a store (my error). Controls caught each one; the headline
   numbers would not have. (Timeouts, E6, Incident.)
6. **Cost fell where evidence was cheap to get, not where the model was
   small.** **[corrected]** The dense gate alone cut executions from 23,414 to
   9,660 at no measurable APPS loss (counts: candidates that start at least one
   sandbox process, validated 30/30 against live runs; the earlier 29,760 and
   11,447 counted recorded runs the policy never uses). But the gate did not
   transfer: on CodeContests it kept the correct solution out of execution
   ~20× more often, so it is now off (STRESS_TEST F4). The small-tier cascade (160M first) escalated 62% of
   statements, lost −0.0076 and saved 11% of encoding cost: the plan's
   hypothesis that most queries resolve on the small tier failed. (E2, final
   replications.)
7. **Everything above is at 0.6B.** The shipped encoder is 1.7B; transfer is
   unmeasured. Expect smaller effects (hypothesis: a stronger first pass leaves
   less to fix). (`HANDOFF.md`.)

## Validity of the measurement

- **Replay equals live.** Replaying the shipped rule over the recorded outcome
  store reproduced a live `apply_execution_rerank` top-10 on 149/150 DEV
  statements. The one difference: a gold solution needing 1.05 s alone timed
  out at the 2 s wall-clock limit under the recording's load. See "timeouts".
- **Sandbox rewrite is verdict-identical on Linux:** 0 differences in 1,844
  train-split runs against the original `main` sandbox.

## E1 — verification policies (DEV, 2,500 statements, all families)

| policy | NDCG@10 | Δ vs legacy [95% CI] | verdict |
|---|---|---|---|
| dense only | 0.90055 | −0.0241 | — |
| legacy (shipped rule, exact) | 0.92464 | — | control |
| + numeric 1e-6 (as shipped) | 0.92648 | +0.0018 [+0.0005, +0.0034] | significant, below the +0.003 minimum |
| judge-aware (statement ε, case rule, any-answer multiset) | 0.92672 | +0.0021 [+0.0007, +0.0037] | significant, below the minimum — **not shipped** |
| **call-style verification** | **0.93172** | **+0.0071 [+0.0047, +0.0098]** | **ships** |
| all of the above | 0.93197 | +0.0073 [+0.0049, +0.0101] | the extra over call-style is negligible |

Mechanism: the call-style gain comes from the call-style families only —
LeetCode 0.886 → 0.912, Codewars 0.941 → 0.945, stdin unchanged. Call-mode
promotion precision 81% (kata) and 87% (LeetCode) against 78% for stdin.

Weak or absent signals, reported as found:
- **Input consumption** barely separates right from wrong programs: among runs
  that complete but mismatch, golds read the input exactly 82.7% of the time,
  wrong programs 68.8% (likelihood ratio ≈ 1.2).
- **Static pruning** of function-style candidates on stdin statements would
  save 2.4% of runs and drop 16 golds — rejected.
- **Crashes** are strong *negative* evidence (95 gold crash runs vs 12,308 for
  wrong candidates) that the binary rule ignores — input to E4.

## E3 — external replication (CodeContests EXT-dev, 108 statements)

- The held-out statements exposed a **parser defect**: Codeforces-rendered
  statements title their *specification* sections with bare `Input`/`Output`,
  and 40 of 217 CodeContests statements had specification prose parsed as
  sample input. Fixed (examples only after an `Example(s)` heading when one
  exists); the fix changes no APPS extraction beyond the earlier `Note` fix.
- With the fix: dense 0.7058 → execution 0.7534 (**+0.048 [+0.020, +0.077]**).
  Call-style and judge-aware policies are neutral here (no call-style
  statements; intervals span zero).
- **The frontier is near-misses.** Without same-problem incorrect solutions
  in the corpus, dense NDCG@10 on these non-APPS statements is 0.937 (higher
  than on APPS) — the encoder finds the right *problem* without APPS
  familiarity. With them, rank-1 is right only 39% of the time: 62 of 66
  rank-1 errors are an incorrect submission to the same problem. And the
  public examples do not separate them: **near-miss incorrect solutions pass
  the samples 43.3% of the time, correct ones 51.4%**.

## E5 — versions (the Bonus): 150 lineages, 3 buggy + 1 correct version

| ranking | correct version first | some version first | buggy above correct | distinct in top-10 |
|---|---|---|---|---|
| dense | 0.167 | 0.793 | 0.780 | 6.67 |
| execution | 0.953 | 0.953 | 0.013 | 6.69 |
| execution + lineage | 0.953 | 0.953 | 0.013 | 10.00 |

Dense reproduces ExecRetrieval's finding (a buggy version outranks the correct
one in 78% of lineages). **Upper bound by construction:** mutants were kept
only if they fail the worked examples. Real near-misses (E3) pass them 43% of
the time.

## E6 — hubness correction of the dense pass (PRE-5)

| bank | DEV Δ NDCG@10 [95% CI] |
|---|---|
| APPS-TUNE queries, beta 0.5 (fitted on TUNE) | +0.0121 [+0.0088, +0.0155] |
| same, TUNE golds removed from the corpus (no "claimed" documents) | +0.0096 [+0.0067, +0.0129] |
| **CodeContests statements (golds not in the corpus), beta 0.3** | **+0.0046 [+0.0019, +0.0074]** |

About 0.0025 of the first figure came from benchmark structure (penalising
documents that answer other train queries), not hubness. Genuine effect:
+0.005 to +0.010 on the dense pass. Interaction with execution: see
"Hubness × execution" and the TUNE replication below (it did not replicate).

## E8 — metamorphic sample amplification (new; not in the plan)

For multi-test-case statements, "same cases twice → same output twice" gives a
new test with a known answer. Sound (98.5% of verified golds pass on APPS,
100% on CodeContests) but weak: it rejects 28% of other-problem false
verifications on APPS and **1 of 111** real near-misses on CodeContests.
Ranking effect +0.00015 (APPS) and +0.0034 [0, +0.014] (CodeContests) —
**not shipped**.

## E2 — evidence-driven cascade (POST-1)

160M first tier → 0.6B on escalation: **rejected** — Δ −0.0076
[−0.0118, −0.0036], escalation 62%, cost −11%, more executions. A strict stop
rule (stop only when the small tier's rank-1 verifies) is quality-neutral but
costs more. Estimated for 0.6B → 1.7B: stopping on rank-1 verification is 97.5%
precise; the query-encoding saving would be ~20% on stdin-style statements and
negative on Codewars — an estimate that needs the 1.7B artifacts.

## Timeouts are load-dependent

A 0.71 s program run 24-at-a-time on 12 logical cores timed out 24/24 under
the 2 s wall-clock rule and 7/24 under a CPU-time rule (`timeout_mode="cpu"`,
POSIX). Verdicts — and therefore scores — depend on the machine that runs
them; CPU-time limits and workers ≤ physical cores reduce that dependence.

## E1 replication on the TUNE half

Call-style verification replicates: **+0.0053 [+0.0033, +0.0073]** on TUNE
(DEV: +0.0071). Numeric and judge-aware effects are smaller still on TUNE
(+0.0005, +0.0007) — consistent with not shipping them.

## E4 — learned evidence fusion (LEVER-style, POST-6)

- As pre-registered (sign-constrained logistic over execution evidence, fit on
  TUNE): **rejected** — −0.0021 [−0.0045, +0.00003] against the best binary
  rule. Better on Codewars/LeetCode, −0.011 on stdin: a pointwise loss did not
  learn that verification should dominate the dense score.
- Post-hoc variant, verified-first kept and the model ordering candidates only
  *within* the verified and unverified tiers, cross-fitted both ways:
  +0.0020 [+0.0007, +0.0034] and +0.0022 [+0.0011, +0.0035]. Consistent but
  below the +0.003 bar and designed after seeing the result — a round-2
  candidate with its own pre-registration, not a shipped change.

## Candidate configuration

`config/candidate_v2.yaml` = `config/frozen.yaml` plus only what passed:
call-style verification, deepening to 50, CPU-time limits. **[corrected]** The
dense gate at 0.20 was in this list until the stress test switched it off.
`tests/test_candidate_config.py` pins that the encoder recipe is identical and
that nothing below the bar is switched on. Adopting it requires one logged
test-split run with the 1.7B encoder; DEV numbers do not transfer as-is.

## Incident: one corrupted store (my error, caught, redone)

A second copy of a recording queue, launched with `&` and wrongly believed to
have exited, ran alongside the tracked one: both executed the DEV deepening
concurrently (doubling CPU load) and wrote the same output file, which failed a
gzip integrity check. Both were stopped, the file deleted, store writes made
atomic (temporary file + rename), and the queue re-run once. Stores recorded
before the overlap passed the integrity check; the 160M store overlapped with
other foreground runs (~14 workers on 12 threads) and may carry a few extra
timeouts — its conclusion (cascade Δ −0.0076) is far outside that noise.

## POST-7 — adaptive deepening (DEV; TUNE replication in "Final replications" below)

When nothing in the first 20 verifies (861 of 2,500 DEV statements), execute
ranks 21–50 too. Increment over call-style verification:

| variant | Δ NDCG@10 vs call-style [95% CI] |
|---|---|
| deep verifications promoted to rank 1 | **+0.0047 [+0.0019, +0.0079]** |
| deep verifications placed at rank 2 (behind the dense top-1) | +0.0035 [+0.0017, +0.0055] |

Positive in every family. **[corrected]** Cost: 32,224 vs 23,414 candidate
executions (+38%; the earlier "+25,830 runs, +87%" counted everything the
recorder ran, not what the policy uses).
Ships only if the TUNE half replicates the direction.

## Hubness × execution (DEV)

Pools rebuilt with the clean CodeContests bank (β = 0.3, fitted on TUNE), then
the same execution policies replayed:

| policy | plain pools | hubness pools | Δ [95% CI] |
|---|---|---|---|
| dense only | 0.90055 | 0.90514 | +0.0046 [+0.0019, +0.0073] |
| legacy + numeric | 0.92648 | 0.93081 | +0.0043 [+0.0015, +0.0074] |
| call-style | 0.93172 | 0.93690 | **+0.0052 [+0.0021, +0.0085]** |

The dense-pass gain survives execution and adds to it. Wired as a default-off
switch (`hubness_bank` in the pipeline config, `scripts/build_hubness_bank.py`);
ships only after the TUNE replication.

## E7 — an encoder not trained on APPS (UniXcoder, contamination control)

**[corrected]** This section's title used to say UniXcoder "predates APPS". It does not
(APPS: May 2021; UniXcoder: March 2022). The control holds for a different reason: UniXcoder
was pre-trained on C4 and CodeSearchNet (its paper, §4), with no APPS or competitive-programming data.

| | NDCG@10 | R@1 |
|---|---|---|
| dense | 0.1182 | 0.081 |
| legacy execution | 0.1221 | 0.087 |
| call-style | **0.1371** | 0.111 |

The pre-registered prediction — a *larger* gain without APPS familiarity — was
**wrong** in absolute terms: +0.019 over dense here against +0.031 with
F2LLM-0.6B (larger only relatively, +16% vs +3.5%). The reason is structural:
execution can only promote a gold that the first pass retrieved, and UniXcoder
puts it in the top 20 for only 18% of statements. The pipeline's value is
bounded by first-pass recall.

With adaptive deepening (ranks 21–50 when nothing in the top 20 verifies) the
UniXcoder pipeline reaches **0.1491** (dense 0.1182 → call-style 0.1371 →
+deepening, +0.0120 [+0.0080, +0.0164]); R@1 0.081 → 0.123. Deepening is worth
2.5× more here than with F2LLM-0.6B (+0.0047) — it is the lever that attacks
the recall bound.

## Final replications on TUNE and the combined configuration

The dense gate had only been measured by the original repo on stdin DEV, and
none of the replays above simulated it — so the gate was added to the replay
and every combination re-measured on both halves (Δ vs call-style):

| | DEV | TUNE | executions (DEV) |
|---|---|---|---|
| + dense gate 0.20 | +0.0001 [−0.0006, +0.0007] | −0.0003 [−0.0011, +0.0004] | 9,660 (vs 23,414) |
| + deepening to 50 | +0.0047 [+0.0019, +0.0079] | +0.0025 [+0.0006, +0.0047] | 32,224 |
| **+ deepening + gate** | **+0.0033 [+0.0012, +0.0057]** | **+0.0026 [+0.0006, +0.0048]** | **11,342** |

**[corrected]** The executions column was wrong (11,447 / 55,590 / 14,948 counted
recorded runs, not the policy's). The corrected counts come from
`scripts/replay_outcomes.py` and were validated 30/30 against live process
starts. Deepening with the gate met the rule at 0.6B at 48% of the ungated
executions. **Superseded:** the gate is now off (STRESS_TEST F4); deepening's
status is in R1 below.

Hubness × execution on TUNE: dense +0.0030 [+0.0002, +0.0056], but with
call-style execution +0.0020 [−0.0005, +0.0047] — the interval spans zero on
the half where β was fitted. It meets the letter of the replication rule (same
sign, not significantly negative) but not its spirit; **it stays off**, wired
and documented, to be switched on only if it clears the rule at 1.7B.

## Shipped vs. not, in one table

| change | verdict |
|---|---|
| call-style verification | ships (0.6B: DEV +0.0071, TUNE +0.0053; 1.7B: DEV +0.0041, TUNE +0.0032) |
| adaptive deepening | kept for weaker first passes (0.6B +0.0047, UniXcoder +0.0120; neutral at 1.7B); final say with R2 |
| dense gate 0.20 | **off** — gold exclusion ~20× higher on CodeContests than APPS (post-hoc; STRESS_TEST F4) |
| parser fixes (`Note`, `Example` headings) | ships (code; 40/217 CodeContests statements fixed) |
| sandbox rewrite, parallelism, corpus cache, CPU-time limits | ships (engineering; verdict-identical on Linux) |
| hubness correction | off — did not replicate with execution on TUNE |
| judge-aware comparator | off — +0.0021, below the bar |
| learned fusion | rejected; tiered variant +0.002 (post-hoc, below the bar) |
| cascade | rejected with 160M; ~20% saving estimated for 0.6B→1.7B on stdin data (unmeasured) |
| metamorphic sample amplification | rejected (+0.0002 APPS, +0.003 n.s. CodeContests) |
| static pruning, input-consumption verdicts | rejected (weak or harmful) |
| lineage ranking (Bonus) | in code; exec@1 0.167 → 0.953 on example-failing mutants (upper bound) |

## Planned but not tested (from `docs/V2-ARCHITECTURE-PLAN.md`)

| item | status |
|---|---|
| PRE-2 encoder view (prose only; examples go to the verifier) | not run |
| PRE-3 repaired variant of body-only documents (19 DEV gold crashes) | not implemented |
| PRE-4 offline LLM spec cards (CodeBridge / EnrichIndex half) | not implemented; needs a GPU box |
| INFRA-2 fractional factorial | not run as designed: arms were tested one at a time, plus the combinations in "Final replications" |
| V3 parser gold set | replaced by external labels (APPS official tests, CodeContests public tests): fails the 0.95 bar; the 207-row human sheet stays for a non-author |
| V5 clean-room reproduction, V8 red team | not done; need someone other than the author |
| EXT-lockbox | sealed by design; reported once, after the freeze |
| CI matrix (ubuntu / macOS / Windows) | workflow pushed; its result has not been observed from this machine |
| 1.7B re-measurement | done (section R1) |
| logged test-split run | not spent; F5 of the stress test estimates ~0 change on that split |

Round-1 DEV/TUNE numbers use F2LLM-v2-0.6B pools; section R1 re-measures them
with the shipped 1.7B encoder.

## R1 — re-measured with the shipped 1.7B encoder

Pre-registration: `R1_17b/prereg.md`. Stores:
`data/outcomes/f2llm17b_all_{dev,tune}{,_deep50}.jsonl.gz` and `f2llm17b_extcc_dev*`.

**Benchmarks, all passed.**
- Replay equals a live run: 150/150 top-10s identical.
- Pool sanity: dense DEV 0.92792, above the 0.90055 bar.
- Headroom: the ceiling (gold at rank 1 whenever it is in the top 50) sits +0.040 above the submitted rule.

| | DEV | TUNE | verdict |
|---|---|---|---|
| call-style vs submitted rule | +0.0041 [+0.0022, +0.0064], sign-flip p < 0.001 | +0.0032 [+0.0016, +0.0049] | **ships** (LeetCode 0.924 → 0.942; stdin unchanged) |
| deepening vs call-style | +0.0007 [−0.0006, +0.0024] | +0.0016 [−0.0001, +0.0035] | neutral; non-inferior on both halves |
| deepening + gate vs call-style | −0.0004 [−0.0019, +0.0012] | +0.0018 [+0.0001, +0.0037] (sign-flip p 0.061) | no quality gain at 1.7B |
| gate alone vs call-style | −0.0008 [−0.0021, +0.0004] | +0.0002 | fails non-inferiority on DEV by 0.0001 |
| judge-aware vs submitted | +0.0004 | +0.0004 | below the bar (as at 0.6B) |

**CodeContests EXT-dev (108 statements).**
- Execution adds +0.034 over dense at 1.7B and +0.043 at 0.6B.
- Deepening adds exactly 0.
- The gate costs −0.0058 [−0.0171, +0.0003] at 1.7B and −0.0084 [−0.0211, −0.0007] at 0.6B.

**Executions per DEV policy at 1.7B:** 23,848 ungated call-style; 32,421 with deepening; 9,188 with deepening + gate.

**UniXcoder with the gate.** It barely prunes anything (47,132 vs 47,284 executions), so a fixed score gap depends on each encoder's score scale.

## V3-oracle — the statement parser against labels its author did not write

Pre-registration: `V3_parser_gold/prereg_oracle.md`. Result: `V3_parser_gold/result_oracle.json`.

**Sanity controls passed.**
- Negative control (every statement given another problem's tests): precision 0.0000–0.0002.
- 80% of CodeContests statements show a sample test in their text.
- Runtime: 1 minute.

| CONF half | stdin precision [95% CI] | recall@cap | bar |
|---|---|---|---|
| CodeContests train (true sample labels) | 0.910 [0.895, 0.925] | 0.980 | **fails 0.95** |
| APPS train (official tests; a lower bound) | 0.938 [0.919, 0.953] | 0.958 | **fails 0.95** |

- **By source:** AtCoder 1.000, Aizu 0.977, Codeforces 0.885, CodeChef 0.847.
- **HackerEarth 0.0.** Its labels carry a "SAMPLE" artefact and are unusable, and the parser also takes its input-format prose as samples.
- **Without HackerEarth:** 0.919.
- **Call-style:** this oracle is uninformative (APPS official tests often omit the statement's examples).
- **Fixes:** only from DIAG failures, judged once on CONF; that is a separate pre-registration.

## Stress test (summary; details in `STRESS_TEST.md`)
- **Sparse effects.** Every effect rests on few statements: 23 of 2,500 for call-style, 6–10 for deepening + gate. Decisions now report a sign-flip permutation test and the number of changed statements alongside the bootstrap.
- **Call-style is robust.** It is positive in 100% of 200 re-splits, but reaches +0.003 in only 86%.
- **The gate is not.** Its gold exclusion depends on the dataset, so it is off.
- **Contamination claims corrected (F1).** Neither the APPS nor the CodeContests exposure of F2LLM-v2 is known per problem.

## R2 — encoder choice, quality side (latency pending)

Pre-registration: `R2_encoder_choice/prereg.md`. Result: `R2_encoder_choice/result_quality.json`.
Pipeline = call-style + deepening to 50, no gate. The 0.6B stores were
re-recorded under the same code as the 1.7B stores (`*_r2*`).

| 0.6B − 1.7B | DEV | TUNE |
|---|---|---|
| dense | −0.0274 [−0.0337, −0.0212] (350 statements differ) | −0.0229 |
| call-style | −0.0209 | −0.0144 |
| **pipeline** | **−0.0169 [−0.0218, −0.0120]**, sign-flip p < 0.001 | **−0.0134 [−0.0182, −0.0086]** |

- **Rule 1 fails.** 0% of 200 re-splits are non-inferior at −0.002, so 0.6B does not replace 1.7B automatically.
- **Post-processing narrows the gap from −0.027 to −0.017** but does not close it.
- **CodeContests EXT-dev:** −0.0055 [−0.0357, +0.0257], so no veto. Recall@50: 0.991 vs 1.000.
- **Confound check.** Examples differ on 3 statements, and only in the ordering of Python sets (STRESS_TEST F11). Timeout rates: 0.55% vs 0.51% (DEV) and 0.65% vs 0.54% (TUNE).
- **Decision:** under rule 4, the measured trade-off goes to the user once the CPU latency (Kaggle CPU machine) is in.
