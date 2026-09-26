# E1 result: the blank-line fallback is adopted into candidate_v2

- **Pre-registration:** `PREREG.md` (commit 62e9445).
- **Decision run:** attempt 1, 2026-09-26 05:38–05:44 (344 s), HEAD b8f1769, 3.5 GB free RAM, 0 spawn failures. Log: `E1_runlog.txt`; full numbers: `E1_result_dev.json`.
- **Command:** `python scripts/exp_e1_blank_fallback.py --split dev --live`, after recording the three sidecars with `record_blank_fallback.py`.

## Decision: every rule and control passes, so it is adopted
| pre-registered rule | threshold | DEV | pass |
|---|---|---|---|
| 1. Δ stdin NDCG@10 (882) | ≥ +0.003 | **+0.0078** (0.95993 → 0.96771) | yes |
| 2. 95% bootstrap CI | excludes 0 | [+0.0031, +0.0131] | yes |
| 3. sign-flip p (exact, two-sided) | < 0.05 | 0.00098 | yes |
| 4. statements changed | ≥ 6 | 12 (11 up, 1 down) | yes |
| 5. all-family Δ | ≥ 0 | +0.0028 [+0.0011, +0.0047] | yes |
| 6. rank-1 golds displaced by a fallback-only non-gold | ≤ 1 | 0 | yes |

| control | required | got |
|---|---|---|
| S1: unmodified replay | 0.95327 / 0.95993 | exact |
| S3: 200 raw runs re-executed (includes all 10 recorded timeouts) | ≥ 198 identical | 200 |
| S4: live top-10 = replay, switch on (and off) | ≥ 45/46 | 46/46 (46/46) |
| Negative control: another statement's collapsed input | ≤ 1 gold, Δ ≤ 0 | 0 golds, Δ 0 |
| Retry control: the same input again, not collapsed | descriptive | 0 golds, Δ 0 |
| Sensitivity: untracked fallback runs rejected | descriptive | identical (no untracked acceptances) |

**Secondary:**
- MRR@10: stdin +0.0085, all families +0.0030.
- The frozen rule plus the fallback: stdin +0.0052 (8 up, 0 down).
- Cost: 26.2 extra runs per affected statement (46 of 2,500 DEV statements); added wall-clock per affected query p50 −0.24 s, p95 +1.1 s at 6 workers.
- The two affected statements whose text calls blank lines meaningful (q816, q1233) are unchanged at 1.0.

## What it means
- Correct programs failed only because the scraped samples render each line followed by a blank line. The official judge inputs have none: on TUNE, 24/24 equal the collapsed sample.
- **The retry control shows the gain is due to collapsing, not a second chance.**
- The mechanism behaves as designed: 38 gold recoveries.
- **The one loss (q720):** the sample answer is the single number `1`. Nothing verifies in the top 20, so deepening runs, and two unrelated candidates (ranks 21 and 27) read the whole collapsed input and print `1`. The gold falls from rank 3 to 5. This is the known short-output coincidence of sample verification, which the consumption gate cannot stop when a program reads everything.

## Adversarial check of the adoption (CLAUDE.md)
- **FAQ:**
  - "Dynamic analysis, such as execution of code is also in scope and allowed, however keep in mind that your approach should ideally generalize and not overfit on a dataset." The rule is generic (a sample with blank lines gets one re-run without them), and it is inactive on cleanly rendered statements.
  - "regex parsing … well within scope."
  - Agrees.
- **Strongest case against:**
  - It repairs an APPS rendering artefact, which fires on 1.8% of DEV statements and 2/108 CodeContests statements.
  - Its value on the unseen set depends on that set's rendering, which is unknown.
  - **Resolution:** it costs nothing where it does not fire, and the claim is scoped to "raises DEV stdin NDCG@10"; no screening gain is projected.
- **Pre-mortem: it hurts on the unseen set if:**
  - (a) blank lines are meaningful there. The fallback runs only after a raw failure; the TUNE and DEV meaningful cases are unchanged.
  - (b) short outputs coincide more often. q720 shows this; the rank-1 displacement count is 0 here.
- **Traps checked:**
  - Detector and gate chosen on TUNE, decided on DEV: yes.
  - Construct validity of the stdin denominator: fixed before DEV, all-family reported.
  - Few statements: 12 changed, exact test.
  - Timing and load: S3 200/200 and S4 46/46.
  - Leakage: none. No test split; qrels used only to score.

## What changes
- `config/candidate_v2.yaml`: `execution_blank_line_fallback: true`.
- `config/frozen.yaml` is untouched; the screening number stays 0.96558.
- Using this in the submission needs a logged test-split run of the whole candidate configuration. That is a final decision for the team.
- Arm B (judge-like stdin and `exit`) is a separate fidelity fix. It is not done in this cycle; it is listed as open in `SUMMARY.md`.
