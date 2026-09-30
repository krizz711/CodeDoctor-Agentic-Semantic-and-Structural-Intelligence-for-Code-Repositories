# P0 — why execution adds nothing on the cleanest CodeContests transfer set

Written 2026-09-26 and committed **before** any of the analysis below was run.

## The anomaly (from the 2026-09-25 debate, `reference/debate-2026-09-25.md`)
- **CC-onesol** (Dr. Mehta, reproduced exactly by Dr. Ortega): the 61 CodeContests EXT-dev problems from the CodeContests *test* split (not in OpenCodeReasoning-2), with every `ccx-*` (incorrect same-problem solution) removed from the pool, so the corpus has one solution per problem, like APPS.
  - Dense (1.7B): NDCG@10 **0.9786**, R@1 **0.951**, ceiling@20 **1.0**.
  - Execution (frozen rule): **+0.0000 [+0.0000, +0.0000]**.
- **APPS DEV, stdin family** (882 statements, 1.7B): dense NDCG@10 **0.90155**, ceiling@20 **0.98299**; execution (frozen rule) **+0.0582 [+0.0462, +0.0709]**.
- Dr. Ortega ranked explaining this above every other next step: if we don't know why execution helps on APPS but not there, no transfer claim is safe.

## Reframing (before any data)
Execution can only promote a gold that dense retrieval already has in its top 20. Its gain is bounded by **headroom = ceiling@20 − dense NDCG@10**:
- CC-onesol: 1.0 − 0.9786 = **0.0214** (about 3 of 61 problems).
- APPS DEV stdin: 0.98299 − 0.90155 = **0.0814**.

So the question becomes: **why is dense retrieval so much easier on CC-onesol?**

## Hypotheses
- **H1 — effective competitor pool.** A CodeContests query effectively competes only with the 216 other CodeContests solutions. The 8,765 APPS documents differ in style and sit far away in embedding space. An APPS query competes with 8,764 same-style APPS documents. Dense retrieval degrades as same-distribution distractors pack around the query (Fang et al., *Scaling Laws for Dense Retrieval*, SIGIR 2024; arXiv 2505.24279; arXiv 2602.05062).
- **H2 — contamination.** The encoder may have seen these problems (xCodeEval covers Codeforces). Unverifiable per problem; checked only indirectly (cc-test vs cc-valid).
- **H3 — statement specificity.** CodeContests statements may be longer and more specific, making dense retrieval easier regardless of pool.

## Measurements (all from stored 1.7B vectors and recorded stores; no encoding, no APPS test data)
- **M0 (known-number control).** Recompute dense rank from stored vectors and reproduce:
  - CC-onesol cc-test dense NDCG@10 **0.9786**;
  - APPS DEV stdin dense NDCG@10 **0.90155**.

  Tolerance ±0.002 (bf16/fp16 noise). **If either fails, the whole analysis is void.**
- **M1 (mechanism, CC side).** For the 61 CC-onesol queries:
  - the share of dense top-20 slots taken by `ccd-*` documents, against their 2.4% share of the corpus (217 of 8,982);
  - the rank of the gold among `ccd-*` documents only.
- **M2 (H1 test, APPS side).** For each APPS DEV stdin query, shrink the pool to **gold + 216 random other APPS documents** (matching CC-onesol's same-source pool of 217). Recompute dense NDCG@10, R@1, ceiling@20 and headroom. 5 seeds; report the mean and range.
- **M3 (execution under the shrunken pool).** One seed. 200 seeded APPS DEV stdin statements that have parsed examples. Live frozen-rule execution on the shrunken-pool top 20: dense vs execution NDCG@10, with a paired bootstrap.
- **M4 (autopsy).** The CC-onesol cc-test statements whose gold is not at dense rank 1. For each:
  - were examples parsed;
  - does the gold verify;
  - does any other candidate verify;
  - what did execution do.
- **M5 (H2, H3, descriptive).**
  - Dense R@1 and headroom on CC-onesol cc-test vs cc-valid.
  - Median statement length (characters), APPS DEV stdin vs CC EXT-dev.

## Decision rules (fixed now)
- **H1 supported** if both hold:
  - (a) **M2:** the APPS shrunken-pool headroom (5-seed mean) is **≤ 2 × 0.0214 = 0.0428**;
  - (b) **M1:** `ccd-*` documents fill **≥ 12%** of CC-onesol top-20 slots (≥ 5× their 2.4% corpus share).

  If (a) holds but not (b), H1 is **partially** supported: pool size explains the difficulty, but the mechanism on the CC side is unconfirmed.
- **H1 refuted** if the APPS shrunken-pool headroom stays **> 0.0428**. Pool size then does not explain the gap, and H2/H3 become the leading candidates.
- **M3 prediction (confirmation, not a gate):** the execution gain on the shrunken pool is **≤ half** its full-pool value (≤ +0.029). If execution still adds ≥ +0.029 there, execution's value on APPS is **not** just "fixing what dense leaves", and that finding gets reported.

## What each outcome would mean
- **H1 supported:**
  - The "anomaly" is a ceiling effect of an easy benchmark, as Dr. Mehta suspected.
  - CC-onesol is **inconclusive** about transfer, not negative.
  - A fair transfer test needs a large **same-source** pool, e.g. CodeContests+ (11,690 problems, `ByteDance-Seed/Code-Contests-Plus`, CC-BY-4.0).
- **H1 refuted:** something about CodeContests makes dense retrieval easier independent of pool size. The transfer question stays open, and H2/H3 need direct tests.

## Negative control
M2 is repeated with the pool shrunk to gold + the **216 highest-scoring** other documents (all hard negatives kept). The headroom should stay near the full-pool value (0.0814). If it collapses too, M2's effect is an artefact of the ranking code, not of which documents are present.

## Limits (stated in advance)
- 61 statements is small. Every CC number carries a bootstrap interval.
- H2 cannot be tested per problem; the cc-test vs cc-valid comparison is indirect.
- M3 runs live on this laptop with wall-clock timeouts, which may be load-sensitive.
