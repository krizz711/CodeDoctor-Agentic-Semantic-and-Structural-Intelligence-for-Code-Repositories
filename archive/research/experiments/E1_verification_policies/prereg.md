# E1 — verification policies over one recorded execution pass

Written **before** any result of this experiment was seen (2026-09-24).

## Data

- Encoder for the candidate pool: F2LLM-v2-0.6B (`f2llm06b`), encoded locally.
  The 1.7B artifacts are not available on this machine; this is a stated
  limitation, not a choice. The 0.6B pool is *weaker* than the shipped one,
  which is closer to the unseen-set condition (no APPS familiarity advantage
  for the encoder is assumed away, but a weaker first pass is simulated).
- Split: APPS **train** split only, all query families, seeded-random
  (`make_development_split(match_family=None, seed=20260919)`): TUNE and DEV
  halves. **DEV** is the gate. The test split is not touched.
- Store: `scripts/record_outcomes.py --model f2llm06b --split dev --exec-depth 20`,
  replay: `scripts/replay_outcomes.py`.

## Validity check (must pass before any policy is read)

`replay_outcomes.py --check-legacy 150`: replaying the shipped rule reproduces a
live `apply_execution_rerank` top-10 on 150 DEV queries with examples. If any
mismatch cannot be explained by timing (timeouts near the 2 s limit), the
replay numbers are not reported.

## Arms (control = `legacy`, the shipped rule without numeric tolerance)

| arm | hypothesis | primary family |
|---|---|---|
| legacy+numeric1e-6 | as shipped; expected ≈ neutral on DEV-all | stdin |
| judge-aware | statement ε, case rule, any-answer *multiset* → more gold verifications on the stdin family, no precision loss | stdin |
| judge-aware+shape | adds a weak "same output shape" tier for any-answer statements placed *below the dense top-1* | stdin |
| call-style | function-style candidates verified by calls → large gain on kata/leetcode families | kata, leetcode |
| call+extended-stdin | adds non-dashed stdin layouts | all |
| all | everything above except the shape tier | all |

## Metrics

Primary: NDCG@10. Secondary: MRR@10, R@1, R@10. Reported overall and per
family (competitive_stdin, kata_function, leetcode_method, unknown).

## Decision rules

- Paired bootstrap over DEV queries, 2,000 resamples; 95% percentile CI.
- Holm correction across the six arms against the control.
- **Ship** an arm only if: corrected CI excludes zero on the positive side,
  point estimate ≥ +0.003 NDCG@10, and no family shows a significant
  regression (per-family CI upper bound < 0 would block it).
- A gain must come from the mechanism claimed: call-style gains from
  kata/leetcode rows, judge-aware gains from rows whose statements carry the
  corresponding rule. Checked with the per-family table and bucket counts.
- Replication before shipping into the frozen config: the same direction on
  the TUNE half (not used for any fitting in E1) and, when available, on an
  external set (E3).
- Anything else is reported as null or negative, with its interval.

## Known threats

- 0.6B pools differ from the shipped 1.7B pools; the effect size on the
  shipped system may be smaller (stronger dense pass leaves less to fix).
- Timing: verdicts near the 2 s timeout can differ between runs and between
  worker counts; the legacy check quantifies it.
- APPS train statements may be in the encoder's training data; that affects
  the dense pass, not the execution evidence being tested.
