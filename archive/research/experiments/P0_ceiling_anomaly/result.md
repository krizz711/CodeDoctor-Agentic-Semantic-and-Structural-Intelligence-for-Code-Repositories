# P0 result: the CC-onesol "anomaly" is a small-pool ceiling effect

- **Pre-registration:** `prereg.md`, committed in d4fb3af before the run.
- **Script:** `scripts/p0_headroom.py`.
- **Output:** `result.json` and `run_log.txt`.
- **Data:** stored 1.7B vectors, APPS train (DEV half) and CodeContests EXT-dev only. No APPS test-split file and no lockbox query was opened.

## Sanity controls (all pass, so the run is valid)
| check | expected | got |
|---|---|---|
| M0: APPS DEV stdin dense NDCG@10 from vectors | 0.90155 | **0.90155** |
| M0: CC-onesol cc-test dense NDCG@10 | 0.9786 | **0.97857** |
| Negative control: pool = gold + 216 highest-scoring others | ≈ full pool (0.0814 headroom) | **identical**, 0.08145 |

## Decision (rules fixed in the pre-registration)
| rule | threshold | measured | holds |
|---|---|---|---|
| (a) APPS headroom with gold + 216 random APPS documents, mean of 5 seeds | ≤ 0.0428 | **0.0103** (range 0.0095–0.0115) | yes |
| (b) share of CC-onesol top-20 slots held by CodeContests golds (`ccd-*`) | ≥ 12% | **14.2%**, 5.9× their 2.4% corpus share | yes |

**H1 is supported.** When an APPS query competes with the same number of same-source documents as a CC-onesol query (217), APPS becomes just as easy:
- dense NDCG@10 rises from 0.9016 to **0.9886** (R@1 0.835 → 0.975);
- headroom falls from 0.081 to **0.010**, below CC-onesol's 0.021.

## Confirmation: M3, live frozen rule, the same 200 DEV stdin statements
| pool | dense | execution | gain [95% CI] | statements up / down |
|---|---|---|---|---|
| full APPS corpus (8,765) | 0.8819 | 0.9576 | **+0.0758** [+0.0475, +0.1061] | 28 / 1 |
| gold + 216 random APPS documents | 0.9908 | 0.9982 | **+0.0074** [+0.0019, +0.0148] | 4 / 0 |

The prediction held: the gain is less than half of the full-pool value (it is 10% of it). Execution's value is repairing what dense retrieval leaves, and in a 217-document pool dense retrieval leaves almost nothing. 253 s wall, 6 workers.

## M4: the 3 CC-onesol cc-test misses
In all three, the gold solution is a correct program but fails the exact comparison against the sample. No candidate verifies, so execution cannot act.

| statement | dense rank of gold | why the gold fails |
|---|---|---|
| cc-test-175 | 2 | prints `Yes` where the sample shows `YES` (case-insensitive checker) |
| cc-test-250 | 4 | prints a different valid answer (multi-answer problem) |
| cc-test-269 | 2 | prints a different valid sign assignment (multi-answer problem) |

## M5: descriptive
- **Split comparison:**
  - cc-valid (plausibly seen via OpenCodeReasoning-2) is *harder* than cc-test: dense 0.955 vs 0.979, headroom 0.045 vs 0.021. No contamination signal, consistent with STRESS_TEST F1.
  - With 47 and 61 statements, this cannot rule contamination out.
- **Statement length:** CodeContests statements are longer (median 2,018 vs 1,640 characters).

## What this means, and what it doesn't
1. **CC-onesol is inconclusive about transfer, not evidence against it.** It is effectively a 217-document task, and at that size APPS shows the same ~zero execution gain. The debate's "+0.0000 on CC-onesol" should not be cited as a failure to generalise.
2. **Execution's value depends on corpus size.** The brief says "the actual code snippets range in the thousands". At that scale, same-source competitors crowd the gold and execution matters (+0.076 here). If the hands-on set were a few hundred snippets, expect it to add little.
3. **The execution gain on a new dataset is still unmeasured.** A fair test needs a large same-source pool. CodeContests+ (`ByteDance-Seed/Code-Contests-Plus`, 11,690 problems, CC-BY-4.0) is the candidate.
4. **The limit that remains is the checker.** Case-insensitive and multi-answer problems cannot verify under exact comparison.
   - The judge-aware comparator addresses the case rule but was below the bar on APPS (+0.0004 at 1.7B), and is off.
   - This is recorded as a known limit, not re-tested here.

## Objections considered
- **"The APPS documents aren't pushed away: 86% of CC top-20 slots are APPS documents."**
  - True, so the pre-registration's wording "pushed away" overstated it.
  - What M1 shows is that APPS documents fill the top 20 but rarely outrank the CodeContests gold (R@1 0.951 overall vs 0.984 among `ccd-*` alone).
  - The decisive evidence is M2, which does not depend on that wording.
- **"The random 216 are easier negatives than CC-onesol's."** A random draw leaves out APPS's near-duplicate problems, which CC-onesol does not contain either. That makes it the matched comparison, not a loophole.
- **"Load-sensitive timeouts."** M3 is one live run, but the two arms ran interleaved on the same statements, so load affects both alike.
