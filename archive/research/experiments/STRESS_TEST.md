# Methodological stress test (2026-09-24)

A read-only attack on our own methods, run before the round-2 decisions were
finalised. Numbers come from `scripts/stress_analysis.py` →
`experiments/stress_test_result.json`, unless a source is named. **V** =
verified here; **J** = judgment.

## F1. Contamination claims: wrong or unsupported (V)
- **F2LLM-v2's training table** ([arXiv 2603.19223](https://arxiv.org/abs/2603.19223), Table 8; searched in the PDF text) names neither APPS nor CodeContests.
  It does include:
  - **OpenCodeReasoning-2** (16,632 samples), whose problems come from "TACO, APPS, CodeContests" (train and valid splits; not CodeContests test), per its dataset card;
  - **xCodeEval NL2Code** (51,072 Codeforces pairs).
- **Our README attributed APPS training to F2LLM via the C2LLM report.** That report lists APPS for C2LLM only (§4.1).
  - *Corrected:* APPS and CodeContests exposure is plausible, and unverified per problem.
- **"Uncontaminated CodeContests" was wrong.** 47 of the 108 EXT-dev statements come from the valid split.
- **"UniXcoder predates APPS" was wrong.** APPS appeared in May 2021, UniXcoder in March 2022.
  - UniXcoder was pre-trained on C4 and CodeSearchNet (paper §4). It is still a valid encoder with no APPS training.
- **No contamination signal in our data (S3).** Valid-split problems score *lower* on dense than test-split ones: 1.7B 0.701 vs 0.736; 0.6B 0.660 vs 0.742.

## F2. Effects rest on very few statements (V; S1)
| 1.7B comparison | half | mean Δ | bootstrap lower bound (5 seeds × 20k) | sign-flip p | statements changed |
|---|---|---|---|---|---|
| call-style − submitted rule | DEV | +0.00410 | +0.00211 … +0.00214 | < 0.0001 | 23 |
| call-style − submitted rule | TUNE | +0.00318 | +0.00152 … +0.00154 | 0.0003 | 23 |
| deepening + gate − call-style | DEV | −0.00035 | −0.00194 … −0.00191 | 0.72 | 6 |
| deepening + gate − call-style | TUNE | +0.00180 | +0.00005 … +0.00010 | **0.061** | 10 |
| gate − call-style | DEV | −0.00075 | −0.00220 … −0.00218 | 0.31 | 5 |
| deepening − call-style | TUNE | +0.00156 | −0.00009 … −0.00004 | 0.10 | 8 |

- **Seed stability:** the bootstrap bounds barely move with the seed (±0.00005).
- **The sparsity problem:** with 5–23 changed statements, a percentile bootstrap can exclude zero where an exact-style test does not.
  - R1's "TUNE deepening + gate, interval excludes zero" is **fragile** (sign-flip p = 0.061).
- **Rule change:** decisions now report the permutation test and the number of changed statements (CLAUDE.md, "No shortcuts").

## F3. Split sensitivity (V; S12: 200 random halves of the 5,000 statements, 1.7B)
| change | pooled Δ | halves positive (lower > 0) | halves ≥ +0.003 | halves non-inferior (lower > −0.002) |
|---|---|---|---|---|
| call-style | +0.00364 | 100% | 86% | 100% |
| deepening + gate | +0.00072 | 5% | 0% | 98% |
| gate alone | −0.00028 | 3% | 0% | **82%** |
| deepening alone | +0.00113 | 10% | 0% | 100% |

## F4. The dense gate (V + J)
- **Gold exclusion depends on the dataset (S2, V).** The gate (gap > 0.20) keeps the correct solution out of execution for:
  - CodeContests EXT-dev: 4 of 107 statements (both encoders);
  - APPS DEV: 1 of 1,439 (0.6B) and 3 of 1,455 (1.7B).

  Fisher exact p = 1.0e-4 and 6.2e-4.
- **The NDCG harm rests on 2–4 statements per encoder (S3), concentrated in the CodeContests *test* split** (the part OpenCodeReasoning-2 excludes):
  - test split: −0.0117 (0.6B) and −0.0105 (1.7B);
  - valid split: −0.0040 (0.6B) and +0.0004 (1.7B).
- **The rule was met on paper.** At 1.7B the gate passed the *letter* of R1's pre-registered EXT rule (−0.0058, interval spans zero).
- **Decision (J, post-hoc):** gate **off**. The reasons:
  - the exclusion-rate mechanism;
  - the significant loss at 0.6B;
  - FAQ 4, "your approach should ideally generalize and not overfit on a dataset";
  - F3's 82%.
- **Cost:** executions go back to ungated (1.7B DEV: 23,848 instead of 9,188). A scale-free replacement (rank- or percentile-based) would need its own pre-registration.

## F5. Construct validity (V + estimate)
- **The splits differ:** APPS test is ~99% stdin statements (0.9% call-style; round-1 statement survey), against 57.6% call-style in train.
- **At 1.7B the v2 changes barely touch the stdin family:** DEV 0.9598 → 0.9598 (call-style), 0.9595 (deepening + gate).
- **Estimate:** a test-split run of `candidate_v2` changes the screening number by about 0. v2's measured value is on call-style problems and on weaker first passes (0.6B, UniXcoder).

## F6. Adaptive reuse of EXT-dev (V)
- **EXT-dev has fed E3, E8, R1 and R2,** so it is no longer a clean held-out set.
- **EXT-lockbox (109 statements) is untouched.** It is the only valid final check on non-APPS data, used once after the configuration is frozen.

## F7. The parser misses the pre-registered bar, but mostly because the oracle's labels are wrong (V; S7 + a post-hoc breakdown)
- **Precision on the confirmation half:**
  - CodeContests CONF 0.910;
  - without HackerEarth (malformed labels, 16 extractions) 0.919;
  - APPS CONF ≥ 0.938.
  - All below the pre-registered 0.95.
- **By source:** Codeforces 0.885 (n = 980 extractions), CodeChef 0.847, Aizu 0.977, AtCoder 1.000.
- **Call-style:** the APPS-based oracle is uninformative (official tests often omit the statement's examples). A different labelled source is needed.
- **Revised after a post-hoc breakdown.** "Robust" was premature. On CodeContests CONF, 117 of the 151 unconfirmed extractions have exactly the public test's *input* and the statement's own *output*: a valid answer that differs from DeepMind's stored one (multi-answer problems, float formatting).
  - Counted as correct, precision is 0.980 (CONF) and 0.968 (DIAG).
  - The oracle's assumption that "public tests = statement examples" fails on multi-answer problems. This was not anticipated when the oracle was pre-registered.
- **What remains real.** Interactive statements, and a few examples missing from the public tests or mis-split. HackerEarth prose and HTML artefacts do not occur in APPS at all.
- **Fixes (V3b).** Interactive statements are skipped, behind a default-off switch. Brace arrays are read as lists (F11).
  - Measured once on CONF: precision 0.9103 → 0.9134; recall 0.9795 → 0.9776, inside the pre-registered tolerance.
  - Details in `V3_parser_gold/prereg_fixes.md`.

## F8. Measurement gaps (V)
- **Timeout sensitivity.** Outcome stores recorded before 2026-09-24 have no per-run durations, so it can't be measured on them.
  - *Fixed:* `ExecutionOutcome.seconds` and a `secs` field in stores.
- **Replay fidelity.** It had been checked only for the legacy ranking (150/150) and execution counts (30/30).
  - *Closed:* `scripts/check_replay_fidelity.py` on 150 seeded 1.7B DEV statements gives **150/150 identical top-10s for call-style and for call-style + deepening** (`R1_17b/replay_fidelity_dev.json`).

## F9. Process (V)
- **Timestamps.** Pre-registrations were not committed before their data was replayed, so their times can't be verified from git.
- **Late addenda.** Some addenda were written after data was *recorded* (never after it was replayed).
- **Rule change:** commit each pre-registration before replay (CLAUDE.md, "No shortcuts").

## F10. What held up (V)
- **Deduplication (S6):** 107 of 108 EXT-dev statements have max Jaccard < 0.1 against every APPS statement.
- **Parser matcher:** the negative control gives 0.0000–0.0002.
- **Pools:** R1's sanity and headroom checks pass.
- **Vectors:** the 1.7B vectors reproduce on another Kaggle machine (min cosine 0.99996).

## F11. Found by R2's confound check: brace arrays parsed as sets (V)
- **The symptom.** After re-recording, 3 of 5,000 statements still "differed" between stores (DEV q1865; TUNE q206, q2927).
- **The cause.** The call-style parser reads LeetCode's C-style brace arrays (`Array = {"10","0001",...}`) as Python **set** literals. A set's text form follows the per-process hash order, so the stored examples differ in order only.
- **Why it is a real bug, not cosmetic.**
  - The argument has the wrong type, and it loses order and duplicates.
  - The sandbox child re-evaluates the set under its own random hash seed, so a candidate's verdict on such a statement can change from run to run.
- **Scope:** 7 of 1,333 call-style statements in APPS train (0.5%): q1865, q4407, q4659, q206, q2927, q3095, q4905.
- **Fixed (V3b).** Brace displays are read as lists in source order, so parsing no longer depends on the hash seed.
  - Regression tests in `tests/test_parser_fixes.py`, including a 5-seed determinism check.
  - The stores were not re-recorded for these 9 statements. Their effect is bounded by 9/2,500 per half.

## The independent reviewer's claims, re-checked in code (V)
All confirmed:
- `_read_config` turned any error into `{}`, i.e. execution silently off. *Fixed.*
- A spawn failure became a silent non-match. *Fixed:* counted, warned about, and reported in the audit.
- `resolve_workers` used logical cores. *Fixed:* physical cores.
- The runner `eval`s the call spec. *Now a stated exception.*
- `>=` requirements.
- No fork-bomb or file-write sandbox tests.
- Few atomic writes.
- Four `except Exception: pass` in `app/server.py`.
- The best-effort test-split log.
- The full suite runs only on ubuntu/3.11.
- The CLAUDE.md contradictions. *Resolved.*
