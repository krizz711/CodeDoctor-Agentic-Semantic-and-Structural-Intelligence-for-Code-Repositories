# E2–E7 pre-registration

Written 2026-09-24 **before** any of these results existed. Same statistics as
E1 (paired bootstrap, 2,000 resamples, Holm within an experiment, +0.003
NDCG@10 minimum effect, per-family no-regression, mechanism check). DEV =
all-family DEV half of the APPS train split; the APPS test split is not used.

## E2 — evidence-driven cascade (POST-1)
- Tiers: F2LLM-v2-160M first, F2LLM-v2-0.6B on escalation (the 0.6B→1.7B
  version needs the 1.7B artifacts; stated as a limitation).
- Hypothesis: most statements with examples are answered by the small tier plus
  execution; the cascade is **non-inferior** to the large tier alone
  (CI lower bound > −0.002) while paying the large encoder on a minority of
  queries.
- Ship rule (cost change): non-inferiority holds AND escalation rate < 60%.
- Reported: escalation rate, parameter-weighted query cost, executions/query.

## E3 — external replication (CodeContests EXT-dev, 108 statements)
- Hypothesis: the E1 arms that ship replicate in direction on EXT-dev; the
  extended stdin layouts matter more here (CodeContests layout).
- EXT-lockbox (109) is reported once, after the configuration is frozen.
- With n=108 the interval will be wide; replication = same sign of the point
  estimate and CI not significantly negative.

## E4 — learned evidence fusion (POST-6)
- Fit on TUNE only (sign-constrained logistic), evaluated once on DEV.
- Hypothesis: fusion beats the binary judge-aware rule, mainly by demoting
  unverified candidates that crash on input parsing and by ordering multiple
  verified candidates; gain concentrated in statements where the gold does not
  verify.
- Ship rule: quality rule vs the best binary policy of E1.

## E5 — versions (the Bonus) on the mutant benchmark
- 150 DEV lineages, 3 single-edit buggy versions + 1 correct, correct position
  randomised.
- Hypothesis (from ExecRetrieval): dense ranks a buggy version above the
  correct one in a large share of lineages; execution + lineage ranking cuts
  that share sharply and raises exec@1.
- Reported: exec@1, lineage@1, buggy_above, distinct@10 for dense /
  execution / lineage rankings. No shipping decision (the Bonus is a hands-on
  demonstration), but the numbers go into the deck only as measured.

## E6 — hubness correction (PRE-5)
- beta chosen on TUNE (cross-half bank), reported once on DEV, dense pass only.
- Prior: small effect for an APPS-familiar encoder. Ship rule: quality rule.

## E7 — contamination control (encoder that predates APPS: UniXcoder)
- Same replay arms as E1 on the UniXcoder pool.
- Hypothesis: the absolute NDCG is low (encoder never trained on APPS), but the
  *gain* from execution-based post-processing is positive and larger than on
  F2LLM, because more golds sit below rank 1 for execution to lift.
- Reported, not shipped: it measures whether the pipeline's value depends on
  contamination.
