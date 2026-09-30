# Final submission reporting protocol — VITV / Doctors

Authorized by the user on 2026-09-30: finish main with accepted upgrades,
measure the final implementation, complete the template, and publish a release.
C4 is stopped/incomplete. This is final reporting, not another optimization sweep.

## Frozen behavior and checks

`config/submission.yaml` contains the accepted C3 candidate settings. F2LLM-v2-1.7B
revision 3766d46e7a68545ed6190c15330983f9b39ab718, CPU FP32, published prompt,
8192 tokens, EOS and normalized last-token embeddings. 0.6B is explicit opt-in.
No Q8/OpenVINO, trained fusion, hubness, cascade or metamorphic treatment is selected.
Verify default configuration and shared execution integration before test access.
Model/data/config/source hashes, exact commands, hardware, packages, timings,
raw logs and run failures are retained. No MTEB result caches. User-authorized amendment (2026-09-30): final CPU query
encoding may be checkpointed in four disjoint recipe-key hash shards, each freshly
computed for this frozen submission. Merge only matching provenance, verify complete
coverage and 30 seeded live re-encodings (minimum cosine 0.999), then score once.
No labels enter encoding or shard assignment. Encoding and scoring replay times are
reported separately; replay time is never presented as live retrieval latency.

## Final validation order

1. Unit/integration checks and isolated-environment synthetic model smoke.
2. P1: fixed 40-function Python fixture, three repetitions of full indexing and
   no-change, one-function, ten-function, add, delete and rename updates. Count
   exact cache misses/reuse; compare the final index with a fresh rebuild.
   Report actual duration and memory without imposing the old E3 hardware gates.
3. Bonus: deterministic 12-query fixture of correct and incorrect Python code
   variants across three versions, with the correct variant alternating between
   oldest/middle/newest. Include duplicate unchanged content. Use the real
   cross-version retrieval and verifier; report correct-version-first, lineage-first,
   buggy-above, distinct@10 and query latency. This is a functional synthetic
   diagnostic, not an estimate of unseen-dataset accuracy. Retain earlier 150-query
   synthetic results as historical only; do not relabel them as final live evidence.
4. Freeze runtime code/config and commit. Run the full AppsRetrieval test once with
   the official MTEB entrypoint, CPU queries and cache=None. Emit authentic MTEB
   JSON and predictions, and record test access before calling the evaluator.
   A completed run is reported regardless of whether the historical score improves.
   No tuning from test results. Do not run EXT-lockbox.

Hosted work uses an isolated Linux environment on the available Kaggle account.
Capture CPU/ISA/RAM first; cap any indexing operation at 30 minutes, P1/Bonus
at 3 hours, each of four CPU query shards at 11 hours, and final P0 scoring at 11 hours. Failures are blocked/incomplete, never replaced
by smaller datasets or historical results. Setup failures may be repaired before
final evaluation; preserve all attempts. Do not automatically repeat a completed
final test evaluation. Offline documents may be cached with full recipe provenance;
cache misses must still work. Timing is machine-specific and excludes no hidden
query transport or preprocessing cost.

## Final-decision review

Alternative: keep the old frozen release. It has a measured test artifact but omits
accepted fixes in hands-on paths. Chosen: ship the accepted settings with one final
measurement, preserving the historical configuration/result separately.
Alternative: 0.6B default. It is cheaper, but prior DEV non-inferiority failed
(delta -0.0169); use 1.7B given P0 priority and expose the cheaper mode explicitly.
Failure risks: statement-specific heuristics may fail on unseen inputs; worked
examples can admit false positives; code fragments may not execute standalone;
long queries cost CPU time; synthetic version fixtures exaggerate verification's
coverage. No guarantee of rank, generalization or speed follows from these checks.
Cheapest falsifying checks are the shared-verifier, no-example stability, duplicate
content, deepening, cache-recipe and version-isolation tests. New integration is
validated as engineering composition, not assigned an unmeasured score gain.

FAQ alignment: CPU retrieval/no API is consistent with A/B; offline caching is
allowed. Dynamic analysis is permitted but requires generalization caveats.
The final MTEB command supports judge re-runs. P1 costs are measured rather than
called sub-second. Python fixtures follow the unseen-dataset clarification.
Model license and benchmark exposure disclosures remain explicit. The FAQ does
not prescribe a model size, a latency threshold, or a P1/Bonus aggregate score.

## Relevant FAQ evidence (verbatim)

The questions are our team's; the answers are from the judge who evaluates this submission. Reproduced exactly as received. No summary is kept: an earlier paraphrase overstated the answers ("CPU-only" for "ideally use the CPU only", "must be documented" for "It would be nice if this fact is documented"). Quote the answers themselves.

Question A: Does this constraint apply strictly to per-query inference time (live demo), or does it also apply to the one-time offline indexing of the repository? Is it acceptable to use GPU acceleration to build the initial dense vector index, provided that the actual real-time retrieval and incremental updates run purely on CPU?

You are allowed to use the GPU during pre/post processing (including incremental updates) phase. However, we would like you to make it as efficient as possible. 

During retrieval, you should make minimal use of the GPU (ideally use the CPU only).

Question B: Does this constraint prohibit the use of closed-source, commercial cloud APIs (e.g., OpenAI embeddings, Anthropic LLMs for reranking)? While calling an API technically runs on a local CPU, the heavy compute is offloaded to remote cloud GPUs. Must the solution be entirely self-contained using local, open-weights models?

Usage of an API during the retrieval also counts as utilizing the GPU. Please keep in mind that the objective of this theme is to develop an efficient and effective code retrieval process, relying on a large model during retrieval is not efficient as this cost is incurred for every query and it is not one-time (unlike LLM usage during pre/post processing).
We do not have a restriction on using closed-source or commercial APIs during the pre/post processing phase, but as mentioned earlier, these are also subject to evaluation. You should to try to make these stages as efficient as possible as well.

Question: Is it acceptable to submit a solution utilizing a model that has seen the benchmark data during its pre-training phase, provided we transparently document this and demonstrate the pipeline's generalizability on a held-out dataset during the final demo?

Yes, it is acceptable to use such a model. It would be nice if this fact is documented in the submission materials (PPT, GitHub repo). Please keep in mind that we would account for this contamination during the hands-on evaluation by using a unseen dataset.

Question: Are teams expected to provide a general-purpose codebase retrieval engine, or is it permissible to heavily engineer heuristics specifically targeting the unique structure of the AppsRetrieval dataset (e.g., regex parsing of competitive programming prompts, or building Python execution sandboxes to run the dataset examples dynamically) to maximize the P0 screening score?

While using static analysis methods such as regex parsing are well within the scope of the theme and expected to part of the solution, we would advise against overfitting too much on the APPS dataset as we will be doing hands-on eval with a unseen dataset (which will be representative of the APPS dataset). Dynamic analysis, such as execution of code is also in scope and allowed, however keep in mind that your approach should ideally generalize and not overfit on a dataset. An overfit system has little usability outside of it's training dataset and is mostly considered a failure.

Question: During the top-15 hands-on evaluation, will the jury independently re-run the AppsRetrieval MTEB evaluation script on the provided GitHub repositories to verify the submitted P0 scores and ensure runtime compliance, or is the hands-on purely focused on the Workstream B JavaScript demo?

We would be verifying the submission by re-running the MTEB evaluation. After that, we would be performing another evaluation using an unseen dataset (which would also be in Python, please ignore the statement about Javascript in the PPT).

Question: Are there specific open-source licensing requirements (e.g., Apache 2.0, MIT) for the models we utilize, given that the winning prototypes may be transitioned into Samsung PRISM worklets? Will utilizing models with non-commercial licenses lead to penalization or disqualification during the hands-on evaluation?

We would prefer embedding/retrieval models with permissible licenses, however, utilization of a non-commercial model would not lead to disqualification. It would be subject to the same screening as the other submissions. The theme focuses on pre/post processing and we don't expect the choice of a embedding/retrieval model to have a significant impact. The implications on a future PRISM worklet are out of scope of the hackathon.

Question: Does this strictly imply incremental re-indexing (where only the modified chunks are re-embedded in sub-second time), or is it expected that the entire repository index is rebuilt from scratch within a specific time limit?

For the P1 goal, an incremental approach might be more efficient over rebuilding the entire index, however, we do not have any restrictions on the approaches taken (a non-incremental approach is also allowed). However, we would consider the time and resource utilization during the evalution.

Question: The sample codebase does not appear to be attached to the current iteration of the problem statement. When will this JavaScript repository be provided to teams, or are we expected to select a substitute repository of our own choosing for the final submission demo?

We would not be releasing an additional dataset/codebase. The APPS dataset will be used throughout the hackathon. We would be using an internal "unseen" dataset for hands-on evaluation, this dataset would not be released to the participants. As for the demonstration, you are free to choose any suitable repository.
