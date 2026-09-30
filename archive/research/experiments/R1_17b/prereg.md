# R1 — re-measure the v2 components with the shipped 1.7B encoder

Written 2026-09-24, before any 1.7B outcome store exists.

## Why
Every round-1 decision was made on F2LLM-v2-0.6B pools. The submission uses
F2LLM-v2-1.7B, and `HANDOFF.md` requires the components to be re-measured at
1.7B before any test-split run.

## Inputs
- Embeddings: `artifacts/apps/emb_f2llm17b_{corpus,qtrain}.npy`, produced by
  the Kaggle kernel `munalsingh/codedoctor-apps-encode` (published recipe,
  revision `3766d46e`, fp16). Verified here: texts identical to the local
  files, train NDCG@10 recomputed as 0.93285 (the manifest's value).
- Tiers: the same DEV/TUNE halves of APPS train (seed 20260919). No test split.
- Execution: same code and defaults as the 0.6B stores (Windows, wall-clock
  timeouts, 6 workers = physical cores).

## Hypotheses
- H1: call-style verification improves NDCG@10 over the submitted rule
  (legacy + numeric 1e-6, depth 20) at 1.7B.
- H2: adaptive deepening to 50 with the dense gate at 0.20 improves over
  call-style at 1.7B.
- H3 (cost): the gate alone is non-inferior to the ungated top 20.
- Expected, not tested: all effects are smaller than at 0.6B, because the
  stronger first pass leaves fewer statements to fix.

## Decision rules (identical to round 1)
- Paired bootstrap over statements, 2,000 resamples, 95% interval, Holm across
  the arms of this experiment.
- **Ships in the test-run configuration** if, on DEV, the corrected interval
  excludes zero and Δ ≥ +0.003, and on TUNE the direction replicates (same
  sign, not significantly negative).
- **Cost changes** ship if non-inferior: lower bound above −0.002 on both halves.
- **Declared before seeing results:** a component that is significant but
  below +0.003 at 1.7B, or non-inferior (lower bound > −0.002 on both halves),
  and that passed at 0.6B *and* on the weaker-encoder arm (UniXcoder), is
  reported as "neutral at 1.7B, kept for generalisation". It may stay on for
  the unseen-dataset configuration, labelled as such. It may not be described
  as improving the APPS score.
- Anything significantly negative on either half at 1.7B is switched off.

## Commands
`experiments/cpu_queue_17b.sh` (record DEV and TUNE, deepen both to 50), then
`scripts/replay_outcomes.py` on each store with `--check-legacy 150`.

## Addendum (2026-09-24, written before any external or hubness store at 1.7B exists)
The DEV store was recorded before this addendum; no replay of it has been run.

- **EXT-dev replication** (`experiments/README.md` requires it for quality
  changes): CodeContests EXT-dev recorded at 1.7B, deepened to 50 over the
  EXT corpus. A shipped quality change must not reverse direction there (same
  sign, or an interval spanning zero). Call-style is neutral on EXT by
  construction (no call-style statements), so this bears on deepening.
- **Versions (Bonus)**: the E5 benchmark recorded at 1.7B and reported as in
  E5. No decision depends on it.
- **Hubness at 1.7B**: β refitted on TUNE with `scripts/hubness_eval.py
  --model f2llm17b` (the same selection rule as at 0.6B), CodeContests bank
  (`artifacts/ext_cc/emb_f2llm17b_queries.npy`). Then DEV and TUNE stores are
  recorded with that β. Hubness switches on only if it meets the round-1 rule
  on DEV (with call-style execution, interval excludes zero, Δ ≥ +0.003) and
  replicates on TUNE with an interval that excludes zero. That is stricter
  than round 1's "same sign" rule, because round 1's TUNE replication was the
  weak point.
- 1.7B external vectors: Kaggle kernel `munalsingh/codedoctor-ext-encode-17b`
  (T4, fp16). The drift probe against the stored APPS vectors gave min cosine
  0.99996 (64 documents, 32 queries).

## Benchmarks (written 12:05, before any 1.7B store was replayed)
- **Replay validity:** `--check-legacy 150` on the 1.7B DEV store must give
  ≥ 149/150 identical top-10s (0.6B gave 149/150; its one miss was a load
  timeout). Below 145/150 the replay is untrusted and no number from it is
  reported.
- **Pool sanity:** dense DEV NDCG@10 at 1.7B must be ≥ the 0.6B value on the
  same statements (0.90055); on the whole train split 1.7B scores 0.93285 vs
  0.90748. Lower means broken pools, and the run is void.
- **Ceiling:** NDCG@10 with the gold moved to rank 1 whenever it sits in the
  top 20 (resp. 50) is the most any post-processing could reach. If ceiling@50
  minus legacy+numeric is below +0.003, H1/H2 cannot meet the bar at 1.7B.
  They are then reported as "no headroom" and decided on non-inferiority only.
- **Runtime:** each stage ≤ 2× the matching 0.6B stage (record ~12 min, deepen
  ~7.5 min). A stage with no progress line for 10 minutes is investigated,
  then stopped.

## Correction and a post-hoc decision (2026-09-24, from the stress test)
- EXT-dev is not "uncontaminated". F2LLM-v2 trains on OpenCodeReasoning-2,
  which draws on CodeContests train and valid; 47 of EXT-dev's 108 statements
  come from the valid split. Exposure per problem is unverified.
- At 1.7B, deepening + gate met the letter of this pre-registration's EXT rule
  (−0.0058, interval spans zero). The gate was nevertheless switched off, on
  the gate's gold-exclusion rate (4/107 on CodeContests vs 3/1455 on APPS DEV,
  Fisher p 6e-4), the significant loss at 0.6B, and FAQ 4. **That decision is
  post-hoc** relative to this pre-registration and is recorded as such
  (`STRESS_TEST.md`, F4).
