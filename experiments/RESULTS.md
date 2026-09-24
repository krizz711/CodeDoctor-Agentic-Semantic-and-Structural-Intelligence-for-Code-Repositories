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
  in the corpus, dense NDCG@10 on these unseen statements is 0.937 (higher
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
+0.005 to +0.010 on the dense pass. Interaction with execution: pending
(`f2llm06b_all_dev_hubcc` store).

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
call-style verification, the dense gate at 0.20 (cost), CPU-time limits.
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

## POST-7 — adaptive deepening (DEV; TUNE replication pending)

When nothing in the first 20 verifies (861 of 2,500 DEV statements), execute
ranks 21–50 too. Increment over call-style verification:

| variant | Δ NDCG@10 vs call-style [95% CI] |
|---|---|
| deep verifications promoted to rank 1 | **+0.0047 [+0.0019, +0.0079]** |
| deep verifications placed at rank 2 (behind the dense top-1) | +0.0035 [+0.0017, +0.0055] |

Positive in every family. Cost: +25,830 runs (+87% executions on this DEV;
fewer with a stronger first pass, which leaves fewer statements unverified).
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

## E7 — an encoder that predates APPS (UniXcoder, contamination control)

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
| + dense gate 0.20 | +0.0001 [−0.0006, +0.0007] | −0.0003 [−0.0011, +0.0004] | 11,447 (vs 29,760) |
| + deepening to 50 | +0.0047 [+0.0019, +0.0079] | +0.0025 [+0.0006, +0.0047] | 55,590 |
| **+ deepening + gate** | **+0.0033 [+0.0012, +0.0057]** | **+0.0026 [+0.0006, +0.0048]** | **14,948** |

Deepening with the gate meets the rule on DEV and replicates on TUNE, at
**half the executions** of today's ungated top-20. It is in
`config/candidate_v2.yaml`.

Hubness × execution on TUNE: dense +0.0030 [+0.0002, +0.0056], but with
call-style execution +0.0020 [−0.0005, +0.0047] — the interval spans zero on
the half where β was fitted. It meets the letter of the replication rule (same
sign, not significantly negative) but not its spirit; **it stays off**, wired
and documented, to be switched on only if it clears the rule at 1.7B.

## Shipped vs. not, in one table

| change | verdict |
|---|---|
| call-style verification | ships (DEV +0.0071, TUNE +0.0053) |
| adaptive deepening + dense gate | ships (DEV +0.0033, TUNE +0.0026, half the executions) |
| parser fixes (`Note`, `Example` headings) | ships (code; 40/217 CodeContests statements fixed) |
| sandbox rewrite, parallelism, corpus cache, CPU-time limits | ships (engineering; verdict-identical on Linux) |
| hubness correction | off — did not replicate with execution on TUNE |
| judge-aware comparator | off — +0.0021, below the bar |
| learned fusion | rejected; tiered variant +0.002 (post-hoc, below the bar) |
| cascade | rejected with 160M; ~20% saving estimated for 0.6B→1.7B on stdin data (unmeasured) |
| metamorphic sample amplification | rejected (+0.0002 APPS, +0.003 n.s. CodeContests) |
| static pruning, input-consumption verdicts | rejected (weak or harmful) |
| lineage ranking (Bonus) | in code; exec@1 0.167 → 0.953 on example-failing mutants (upper bound) |

All DEV/TUNE numbers use F2LLM-v2-0.6B pools; see `HANDOFF.md` for the 1.7B
re-measurement and the single logged test-split run.
