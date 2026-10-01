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

## Local correction amendment — 2026-10-01

The user authorized the audited correction plan **locally only**. The submitted
commit `408a61f0f6e0c2a58528e88155f3dc6a4fa25a33`, local `main`, the official
tag and every published artifact remain unchanged. Work is on
`fix/submission-corrections-local`; a local commit is not a submitted amendment.
The user's subsequent clarification on 2026-10-01 permits other platforms:
“When I meant locally I was only talking about not touching github, you can
still access any other platform for this.” This authorizes the private Kaggle
continuation below. GitHub writes, release changes and retagging remain forbidden;
YouTube is outside this evaluation's scope and remains unchanged.

Before launch, the prepared private CPU notebook
`lifemeansfun/codedoctor-correction-p0` packages runtime freeze
`d2fb22a4a3e8be1ff94374dfb903c438428f74c6`. All 73 packaged files match local
documentation-only commit `3faf0e72932cf9f33fe748fdc9f4b64ab9964d51` byte for byte.
The `run.py` SHA-256 is
`0621cd21361148585624a77404f31c07bc0b9803b975c894f659ae365f245597`;
candidate configuration SHA-256 is
`9039497b871733c5236eb6060b0cf074c3d98ebeb30664c2ce711e4882f0aaf5`.
The account's notebook inventory was checked before launch: no correction-P0
notebook was listed. Preserve this run's audit log separately from historical
APPS exposure records; APPS test is previously exposed, not a fresh lockbox.

Attempt 1 (Kaggle version 1) stopped during test collection, before model/APPS
loading, because the notebook package omitted the replay and sample-evaluation
helper scripts. The sample fixture was also missing. Preserve its original
payload, setup environment, package freeze, full output checksums and raw logs
under `results/local-corrections/kaggle-attempt-1/` and
`build/local-correction/kaggle-attempt-1/`. Commit `74a9ddd` adds only those
packaging support files and an isolated-package regression test. Model, app,
encoder, evaluator and candidate configuration remain byte-identical to `d2fb22a`.
After isolated-package validation, use the single permitted infrastructure retry
as Kaggle version 2. A valid completed result will not be repeated for its score.

This continues final reporting, not a model or ranking experiment. Keep the
selected encoder, revision, instruction, pooling, sequence limits and execution
settings unchanged. Repair artifact validation, the demo's tier mismatch,
reproduction instructions and unsupported claims before freezing the corrected
runtime. Preserve all historical evidence and test-split tuning disclosures.

Before scoring, require the exact recipe-key sets of the pinned MTEB query and
corpus texts, each shard's correct hash membership, no missing/unexpected keys,
finite nonzero normalized 2,048-dimensional vectors, and matching dataset
revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. There are 8,765 corpus
records; identical text may share a vector (the recorded cache has 8,754).
Use only content and the encoder recipe for these checks; labels and IDs are
not retrieval features. Preserve the registered 30-query verification seed
`20260930`, minimum cosine `0.999`, and reject nonfinite live vectors/cosines.
Final replay must encode zero missing query keys. `cache=None` remains required.

Use disposable isolated Linux for candidate execution, with no personal files
or credentials mounted. Capture the actual platform, resources, threads,
packages, model/data/source hashes, memory, timeout and spawn-failure records.
The existing 11-hour scoring cap and one completed final run remain in force.
Do not tune or repeat a valid completed run to improve its score. An
infrastructure failure is retained and its cause reviewed before a retry.
Keep EXT-lockbox sealed. Complete synthetic negative checks and a real-model
cold Python functional check before APPS scoring; neither demonstrates unseen
accuracy. No result from a reduced dataset substitutes for final reporting.

This Mac has 8 GiB RAM, below the recorded 10.85 GB accuracy-tier peak. A local
Linux evaluation is blocked unless suitable resources become available. A
private Kaggle notebook is now authorized by the clarification above.
Never substitute lower precision, omit live checks,
or relabel the historical GPU-query score as the current CPU result.

FAQ alignment: A permits offline GPU document preprocessing; B's efficiency
concern is respected by accounting for query encoding separately. The
contamination and generalization answers require qualified disclosures and
no claim of fresh unseen accuracy. The rerun answer requires an uncached CPU
query path and working Python inputs. The license answer is a preference,
not permission to invent a repository license. The P1 answer permits either
incremental or full rebuilding; report time/resources with the fixture and
machine. The demo-repository answer permits our reproducible Python fixture.
The full relevant answers are retained verbatim below. They do not specify
post-submission amendment permission, a response-CSV schema, a numerical
latency limit or checkpointing acceptance. The checkpointing choice is the
team's previously authorized procedure, not an invented jury requirement.

The five-minute video requirement is on slide 11 of the supplied PPT template.
Prepare cut-only footage under five minutes with the original preserved. Do
not cut computation waits to imply faster execution. The public 5:46 video
remains unchanged until separate permission is received. A locally corrected
deck, video and disclosure do not change the submitted package. Human ethics
attestations and representative signature cannot be supplied by the agent.

## Completed final reporting — 2026-10-01

Private correction-P0 version2, the single infrastructure retry, completed
without changing model or ranking settings. Measured commit:
`18c4b6cbd097437bfb8498fefe572cddb658bc42`; run.py SHA-256:
`f5fbe53ca17746b765bd8f36fb88227442ecbe245fc90f25e23ef9da5c7780f2`.
Configuration SHA-256 remains
`9039497b871733c5236eb6060b0cf074c3d98ebeb30664c2ce711e4882f0aaf5`.

Accepted result: **NDCG@10 0.96583; MRR@10 0.9582509116971268;
Recall@10 0.98805**. Full3,765-query/8,765-document coverage, zero query/document
fallback,30 seeded live checks (minimum/median cosine1.0), zero spawn failures,
official JSON and raw prediction/CSV validation passed. Linux validation passed
273 tests with one optional-fixture skip; the real1.7B cold Python functional
check passed. All56 P0 and39 cold-check source records match the local candidate.
Later evidence/documentation commits are compatible only if those hashes match.

The P0 runner took2,609.959 seconds with9,405,206,528-byte peak process-tree RSS;
this excludes the four earlier CPU encoding jobs and is scoring replay with30
live checks. Do not report it as full-live retrieval latency. Relative to
historical GPU P0, NDCG/recall rose slightly while MRR fell slightly. No tuning,
new run, significance claim or attribution to individual experiments follows.
APPS was previously exposed; the hosted audit is preserved separately from
historical access. EXT-lockbox remains sealed.

Authentic [results and reproduction commands](../../results/final/p0/README.md),
[acceptance](../../results/local-corrections/kaggle-attempt-2/acceptance.json)
and both failed/successful attempts are retained. Recompute evidence inspection
against the precise proposed commit before publication. GitHub main, official
tag and release remain unchanged; Samsung approval and explicit user permission
are still required for any GitHub mutation. Human video/disclosure review and
unexecuted Docker/remote-CI checks remain outside this completed CPU evaluation.

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
