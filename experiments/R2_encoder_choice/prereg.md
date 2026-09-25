# R2 — which encoder should the submission use?

Written 2026-09-24, before any 1.7B pipeline replay has been run and before
any CPU latency has been measured.

## Why
The screening run (0.96558) encoded queries with F2LLM-v2-1.7B on a Kaggle
GPU. The FAQ asks for CPU retrieval. Nobody has measured 1.7B query encoding
on a CPU; this laptop (7.7 GB RAM) cannot hold 1.7B in fp32. The question is
whether a smaller encoder plus the CPU post-processing is as good.

## What the FAQ says (the evaluating judge's own answers)
- **FAQ 1:** "During retrieval, you should make minimal use of the GPU (ideally
  use the CPU only)." Query encoding happens at retrieval, so it has to be
  CPU-feasible.
- **FAQ 2:** "relying on a large model during retrieval is not efficient as this
  cost is incurred for every query." This bears directly on the query encoder's
  size.
- **FAQ 3:** contamination is "account[ed] for ... by using a unseen dataset."
  APPS gains that come from a larger model's memorisation may not transfer, so
  uncontaminated evidence (CodeContests) weighs more than APPS.
- **FAQ 5:** the judges re-run MTEB. The number we submit must reproduce on
  their machine, and its runtime is visible to them.
- **FAQ 6:** "we don't expect the choice of a embedding/retrieval model to have
  a significant impact." All candidates are Apache-2.0 (checked on the HF API).
- **FAQ 7:** "we would consider the time and resource utilization."
- **Silent:** whether the screening cut to the top 15 looks at anything but
  NDCG@10 (the brief says NDCG@10 and MRR). The FAQ gives no answer; this is
  not assumed either way.

## Candidates
F2LLM-v2 1.7B, 0.6B and 160M: same family, recipe and licence, and DEV
outcome stores exist or are being recorded. UniXcoder is excluded (dense
NDCG@10 0.118). 80M has no store and enters only if 160M passes.

## Measurements
- **M1 quality:** pipeline = call-style + deepening to 50 + dense gate 0.20
  (the v2 candidate), replayed on each encoder's DEV and TUNE stores; dense
  reported too. Also reported: recall@20 and recall@50 of the dense pool.
- **M2 CPU cost on this laptop** (6 physical cores, 7.7 GB RAM; recorded
  in the result): per-query encoding latency at batch 1 (p50/p95), throughput
  with the encoder's own batching, load time and peak RSS. 100 seeded TUNE
  statements, float32 (the repo's CPU dtype). A model is **not loaded** if its
  float32 weights exceed 80% of available RAM; that fact is the result, and
  bfloat16 is measured instead. Its vectors are compared with the stored GPU
  vectors for the same statements, and it is labelled as a different operating
  point. Execution cost per statement = executions (from the replay) × the
  measured per-run time, also reported per encoder.
- **M3 uncontaminated check:** CodeContests EXT-dev pipeline NDCG@10 and
  recall@50 for every encoder with EXT vectors (0.6B, 1.7B).

## Decision rule (fixed sequence; no Holm needed)
1. 0.6B against 1.7B, pipeline vs pipeline, paired by statement. Adopted if
   the Δ interval's lower bound is above **−0.002** on DEV **and** on TUNE.
   This is the repo's existing cost rule (`experiments/README.md`), not a new
   margin.
2. Only if 0.6B is adopted: 160M against 1.7B, same test. Adopt the smallest
   encoder that passes, stopping at the first failure.
3. EXT-dev: reported; an adopted encoder must not be significantly worse than
   1.7B there (interval upper bound ≥ 0).
4. **If 0.6B fails the −0.002 margin, nothing switches automatically.** The
   measured trade-off (Δ NDCG@10 against CPU seconds per query) goes to the
   user, because the FAQ does not say how screening weighs score against
   efficiency.

## Adversarial review, done before the data
1. **Strongest case for keeping 1.7B.** Screening ranks by NDCG@10 and the FAQ
   is silent on efficiency at that stage; −0.01 could cost the top-15 cut.
   → *Resolved by rule 4:* a loss beyond the pre-registered margin is never
   accepted automatically.
2. **Strongest case against the rule.** DEV/TUNE are APPS, which F2LLM was
   trained on (FAQ 3), so 1.7B's advantage there may be memorisation that the
   unseen set removes. → *Partly open:* EXT-dev is uncontaminated but has 108
   statements, too few to test a −0.002 margin. It is used as a veto (rule 3),
   not a gate.
3. **Pre-mortem: we switch to 0.6B and the hands-on set is harder, so the
   smaller model's pool misses the gold and post-processing cannot recover it**
   (the UniXcoder lesson). → Recall@50 is reported per encoder on DEV and EXT.
   A significantly lower recall@50 on EXT is flagged even if rule 1 passes.
4. **Confound: stores recorded at different times.** Parser fixes landed
   during round 1, and load at recording time affects timeouts. → Before any
   comparison, check that every shared statement has identical extracted
   examples in both stores, and report timeout counts per store. If the
   examples differ, the older store is re-recorded before comparing.
5. **My own earlier proposal.** The 0.005 margin was ad hoc and conflicted
   with the repo's −0.002 cost rule. → Replaced.
6. **Operating points.** Latency is measured on this laptop, not the judges'
   machine. Claims stay relative ("×N faster") and name the hardware. 1.7B in
   bf16 on CPU is a different operating point from the fp16 GPU vectors, and
   is labelled so.
7. **The dense gate (0.20) was tuned on 0.6B scores.** Other encoders' score
   scales may differ. → The replay applies the same gate to each encoder and
   reports executions per encoder. The gate is not re-tuned per encoder.

## Benchmarks (written before any CPU latency or pipeline comparison was run)
- **Known number:** the repo's earlier CPU figure for 0.6B is 0.35 texts/s on
  six threads (~2.9 s per text; texts not recorded). A measured 0.6B batch-1
  mean outside 0.5–2× of that is explained (statement length, threads) before
  any latency is used.
- **Fidelity:** CPU float32 vectors must match the stored GPU vectors for the
  same statements at median cosine ≥ 0.999. Otherwise the CPU operating point
  differs from the one whose quality was measured, and the quality numbers do
  not transfer.
- **Pointlessness stop:** a model whose first 10 statements average over 60 s
  each is stopped there. At that rate the 3,765 test queries would take more
  than 60 hours, which is not a CPU retrieval system under FAQ 1/2/7, and more
  statements would not change that conclusion.
- **Confound check before any quality comparison:** identical extracted
  examples for shared statements across the stores being compared
  (adversarial item 4).

## Correction (written before any R2 data; found by checking the brief itself)
"Silent" above was wrong. The FAQ is silent on how screening ranks, but the
brief is not: "We will rank your solutions on the P0 submission goal
(Retrieval Accuracy) based on their performance on the test split ... This
will be a competitive screening meaning we will be selecting the top
submissions for hands-on evalution." So screening ranks by accuracy, and
efficiency counts at hands-on (brief: the demo should show "how fast your
solution is"; FAQ 2 and 7). This strengthens adversarial item 1 (keep the
strongest APPS score for screening) and makes rule 4 — never trade score
beyond the margin without the user — more important, not less.

## Confound check result (before any R2 comparison)
Extracted examples differ on 2 of 2,500 statements per half between the
round-1 0.6B/160M stores and the new 1.7B stores (DEV q3978, q1865; TUNE
q206, q2927). Timeouts: 0.62% of runs in the round-1 0.6B stores against
0.51–0.54% in the 1.7B stores, recorded under heavier load. As
pre-registered, the older stores are re-recorded (`experiments/cpu_queue_r2.sh`,
new files `*_r2*`; the round-1 stores are kept for provenance). 160M is
re-recorded only if 0.6B passes rule 1.

**Deviation (M2):** the CPU benchmark runs on a Kaggle CPU machine, not this
laptop. With the desktop apps open, the laptop had 1.4 GB of RAM free: not
enough for even the 160M model with activation headroom. The machine is
recorded in the result.

## Addendum after the stress test (written before any `_r2` store is replayed)
- **Correction:** "uncontaminated" CodeContests above is wrong. EXT-dev is
  only partly protected (47/108 statements from the CodeContests valid split,
  which OpenCodeReasoning-2, a source in F2LLM-v2's training mix, draws on).
  M3 is reported as a non-APPS check, not an uncontaminated one.
- **Pipeline compared:** call-style + deepening to 50, **no gate** (the gate
  is off; STRESS_TEST F4). Dense and call-style without deepening are
  reported too.
- **Tests, because effects rest on few statements (F2):**
  - Paired bootstrap (2,000) **and** a paired sign-flip permutation test.
  - The number of statements whose rank changes.
  - Non-inferiority requires the bootstrap lower bound above −0.002 on DEV and
    on TUNE, **and** at least 95% of 200 random re-splits of the pooled 5,000
    statements non-inferior (normal approximation per half).
- **Recording note:** the 0.6B `_r2` stores were recorded while the code
  changed underneath them. The recording step used the old code; the
  deepening steps loaded new code that only adds a `secs` field and
  spawn-failure counting. Verdict logic is unchanged. The stress-test analysis
  (one core) ran during `rec_06b_dev_r2`. Timeout rates are compared before
  any comparison is trusted.

## Outcome (2026-09-25)
- **Rule 1 (0.6B vs 1.7B):** fails. DEV −0.0169 [−0.0218, −0.0120], TUNE −0.0134 [−0.0182, −0.0086]; 0% of 200 re-splits non-inferior. Rule 2 (160M) is therefore not run.
- **Rule 3 (EXT veto):** not triggered (−0.0055 [−0.0357, +0.0257]).
- **M2 (CPU cost, Kaggle CPU, 2 cores):** 1.7B 7.83 s/query and 8.5 GB; 0.6B 2.79 s and 3.9 GB; 160M 0.48 s and 2.1 GB.
  - Benchmarks: the known number passes, fidelity passes for float32, and bfloat16 fails fidelity.
- **Rule 4:** nothing switches automatically. The trade-off goes to the user.
