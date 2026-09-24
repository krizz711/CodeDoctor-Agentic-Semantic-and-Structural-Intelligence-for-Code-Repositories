# CodeDoctor — Samsung PRISM GenAI Hackathon 2026, Theme 01 (code retrieval)

## Working rules

### Objectivity
- Judge every proposal on evidence, regardless of author (teammate, user, or Claude's own earlier output). Credit what is right and state what is wrong in the same review.
- Verify each factual claim against code, data, or a cited source before repeating it; label anything unverified as "unverified" or "estimate/hypothesis".
- Never present an estimate beside a measurement without labelling it. Never mix operating points (e.g. 1.7B accuracy with 0.6B latency) in one claim.
- "Implemented" ≠ "on the scored path" ≠ "improves the score". Say which one is true.
- Report negative and null results as plainly as wins, including regressions caused by our own changes.

### No shortcuts
- No metric is claimed without running it; quote the command and its output.
- Pre-register each experiment (hypothesis, metric, decision rule) before running it. No change ships on an in-sample number.
- Gate on held-out data with paired bootstrap + multiple-comparison correction; if the CI includes zero it is not an improvement.
- The APPS test split is for final reporting only, one run per frozen config, logged in data/test_split_audit_log.json. It never chooses a configuration; if it ever did, disclose it.
- Never use meta_information, the partition column, qrels/id structure, test-batch transduction, or MTEB result caches (always cache=None).
- Never skip, xfail, loosen, or delete tests to get green; never bypass hooks; paste failures verbatim.
- Read the whole code path before changing it; prefer small, reversible, tested changes.

## Brief essentials (theme1_guidelines.pdf)
- P0: rank snippets for an NL query; screened on CoIR AppsRetrieval test split via MTEB (`mteb.get_task("AppsRetrieval")`), NDCG@10 and MRR; submit the MTEB JSON (+ CSV of responses) as a GitHub Release, plus PPT and a runnable repo.
- P1: rebuild indexes/caches for any version in reasonable time. Bonus: retrieve across all versions (near-identical snippets are hard to rank).
- Generation/explanation after retrieval is out of scope. Expected to run on CPU with minimal GPU.

## Samsung jury FAQ — answers verbatim (binding)
Summary: GPU/API allowed only offline (pre/post-processing, incremental updates) and must be efficient; retrieval CPU-only; an API at query time counts as GPU; contamination allowed but must be documented; don't overfit to APPS — hands-on uses an unseen Python dataset; judges re-run MTEB; time and resources are evaluated.

1. GPU for indexing vs retrieval — "You are allowed to use the GPU during pre/post processing (including incremental updates) phase. However, we would like you to make it as efficient as possible. During retrieval, you should make minimal use of the GPU (ideally use the CPU only)."
2. Commercial/cloud APIs — "Usage of an API during the retrieval also counts as utilizing the GPU. Please keep in mind that the objective of this theme is to develop an efficient and effective code retrieval process, relying on a large model during retrieval is not efficient as this cost is incurred for every query and it is not one-time (unlike LLM usage during pre/post processing). We do not have a restriction on using closed-source or commercial APIs during the pre/post processing phase, but as mentioned earlier, these are also subject to evaluation. You should to try to make these stages as efficient as possible as well."
3. Contaminated models — "Yes, it is acceptable to use such a model. It would be nice if this fact is documented in the submission materials (PPT, GitHub repo). Please keep in mind that we would account for this contamination during the hands-on evaluation by using a unseen dataset."
4. APPS-specific heuristics — "While using static analysis methods such as regex parsing are well within the scope of the theme and expected to part of the solution, we would advise against overfitting too much on the APPS dataset as we will be doing hands-on eval with a unseen dataset (which will be representative of the APPS dataset). Dynamic analysis, such as execution of code is also in scope and allowed, however keep in mind that your approach should ideally generalize and not overfit on a dataset. An overfit system has little usability outside of it's training dataset and is mostly considered a failure."
5. Re-running MTEB — "We would be verifying the submission by re-running the MTEB evaluation. After that, we would be performing another evaluation using an unseen dataset (which would also be in Python, please ignore the statement about Javascript in the PPT)." [JavaScript demo requirement superseded.]
6. Model licences — "We would prefer embedding/retrieval models with permissible licenses, however, utilization of a non-commercial model would not lead to disqualification. It would be subject to the same screening as the other submissions. The theme focuses on pre/post processing and we don't expect the choice of a embedding/retrieval model to have a significant impact. The implications on a future PRISM worklet are out of scope of the hackathon."
7. P1 incremental vs rebuild — "For the P1 goal, an incremental approach might be more efficient over rebuilding the entire index, however, we do not have any restrictions on the approaches taken (a non-incremental approach is also allowed). However, we would consider the time and resource utilization during the evalution."
8. Sample codebase — "We would not be releasing an additional dataset/codebase. The APPS dataset will be used throughout the hackathon. We would be using an internal "unseen" dataset for hands-on evaluation, this dataset would not be released to the participants. As for the demonstration, you are free to choose any suitable repository."

## Where things live
- Architecture plan: docs/V2-ARCHITECTURE-PLAN.md
- Experiments and results: experiments/README.md, experiments/RESULTS.md, experiments/HANDOFF.md
