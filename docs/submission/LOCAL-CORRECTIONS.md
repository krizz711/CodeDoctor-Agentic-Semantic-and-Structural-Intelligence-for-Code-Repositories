# Local submission correction record — 2026-10-01

This is an engineering correction and evidence reconciliation, not a new model
experiment. The user authorized local work only. The submitted `main` and tag
`PRISM_GENAI_HACKATHON_Y2026` stay at
`408a61f0f6e0c2a58528e88155f3dc6a4fa25a33`. No local commit, notebook preparation,
render or validation changes the judged submission. On 2026-10-01 the user
clarified that other platforms may be used: the prepared private Kaggle CPU
evaluation is authorized. GitHub changes remain forbidden; public video unchanged.

## Requirements and decisions

The authoritative sources are the four-page `theme1_guidelines.pdf`, the complete
verbatim FAQ in `CLAUDE.md` and `PROTOCOL.md`, the supplied 12-slide template, and
the official portal screenshot provided by the user. The portal requires the
exact tag and says its commit is judged; the template specifies a maximum
five-minute video. The theme PDF requires both response CSV and MTEB JSON, PPT,
and runnable repository instructions. FAQ efficiency guidance permits GPU
preprocessing, including updates, while preferring minimal GPU retrieval,
ideally CPU. It does not prescribe a numerical latency threshold.

Keep the selected 1.7B model, pinned revision, CPU FP32 query path, instruction,
EOS, normalized last-token pooling, sequence handling and accepted C3 execution
settings. Do not retrain, quantize, tune on test scores, weaken the decision
dataset or use an old score as the new candidate's result. EXT-lockbox remains
sealed. The registered four-shard procedure, 30 live re-encodings, seed20260930
and minimum cosine0.999 remain unchanged.

## Implemented corrections

| Finding | Correction | Evidence boundary |
| --- | --- | --- |
| Apps demo loaded 0.6B arrays while default queries used1.7B | Tier-matched arrays, full recipe/model/revision checks, checksums and row identity; invalid/legacy artifacts fail visibly | Unit checks; real cold1.7B validation still requires a larger machine |
| Query replay could fill gaps live or accept invalid vectors | Strict P0 requires completed, compatible, disjoint shards and exact key coverage; dimensions, finiteness, nonzero normalized vectors and live fidelity are checked | Synthetic negative tests; authentic downloaded stores require a recorded preflight |
| Corpus manifest did not prove task/text identity | Pinned dataset/revision/count/fingerprint and exact recipe-key checks | No new scoring result follows from cache validation |
| Result/CSV validation could accept mismatched task or rounded scores | Strict official task/revision/test checks; raw prediction ID/metric recomputation and full float CSV precision | Authentic current P0 remains absent until one authorized run completes |
| Declared Torch version and environment differed | Portable Torch2.8.0 recipe, aligned installs; measured freezes retained as evidence, not installation locks | Fresh Mac installation and tests; Linux execution and Docker build are separate checks |
| Slides/README overstated speed, correctness and pipeline scope | Correct fixture medians; label historical GPU P0 and constructed Bonus; distinguish scored pipeline from workbench and C++ prototype | Historical raw measurements stay unchanged |
| Official AI form was replaced and claimed an agent-generated signature | Restore original six-section form, fill supported facts, retain per-feature fields, leave human confirmations/signature unsigned | Team must verify authorship, consent and final attestation |
| Video exceeded maximum and omitted parts of the claimed demo | Preserve original; make cut-only local preview and exact cut-review record | Human narration review and missing demonstrations cannot be replaced by ASR or trimming |

## Verification and evidence

Detailed raw local records are in `build/local-correction/`; durable small
validation records are copied to `results/local-corrections/`. These include the
tested package freeze, command outcomes, the first failing suite and final
verification. The first suite used the pre-existing Torch2.14 environment and
found four outdated CLI fixtures missing task metadata. Those fixtures were
corrected without weakening runtime validation. A fresh Python3.12 environment
installed the declared Torch2.8/MTEB2.21/encoder stack and passed607 tests,
six platform skips, six model/data deselections and five subtests. Lint,
compilation, C++ build/tests, sample evaluator and historical official JSON
validation passed. Later source changes need their own recorded regression
checks before the local freeze is considered ready.

Independent static review subsequently found two run blockers: MTEB2.21 can
unload `task.dataset` before CSV validation, and explicit shard inputs contain
nested `Path` objects that were not JSON serializable. Query/corpus IDs and copied
qrels are now retained before evaluation, after logged access intent; setup time
is recorded. Environment arguments normalize paths recursively. Final CLI,
launcher, logging, CSV and cold-check regressions passed43 tests in the portable
environment. No labels were added to retrieval features or tuning.

The frozen correction commit `d2fb22a` was then tested from a clean local clone:
608 passed, 7 skipped, 6 deselected, 5 subtests. The extra skip needs a local
corpus-text artifact absent from the clean clone; the other six are Mac platform
checks. Authentic downloaded-cache preflight passed for8,754 document vectors
and3,765 disjoint CPU query vectors, with matching provenance and old source
hashes. It did not access fresh query text or labels. The first local wrapper
failed on a report-field typo after checking the vectors; its log is preserved
and a single corrected retry produced the report. Exact pinned-task text
coverage, real-model forwards, Linux checks and final P0 remain pending.

The prepared notebook is `build/local-correction/kaggle-p0`, private CPU-only,
packaged from clean commit `d2fb22a`. Version 1 was uploaded successfully on
2026-10-01 following the user's clarification and is running at
https://www.kaggle.com/code/lifemeansfun/codedoctor-correction-p0 . Launch evidence
is retained in `results/local-corrections/kaggle-attempt-1/`; no result exists
until its checks and official evaluation complete. Read-only `git ls-remote`
confirmed both public main and the official
tag still resolve to submitted408a61f after all local work.

All12 revised slides were rendered and visually inspected both through Artifact
Tool and the PDF export. Package, geometry, reference font and import checks
passed. This does not assert native Microsoft PowerPoint validation. The
disclosure preserves all41 original paragraphs, original section geometry and
every package part except `word/document.xml`; it is rendered and inspected.

Four existing CPU query jobs were downloaded read-only from Kaggle. Recorded
counts are971/882/975/937 (total3,765); CPU encoding elapsed seconds are
11,100.779 /10,147.842 /12,613.804 /13,987.302. Their environment metadata records
source823aa87, a common configuration and dependency provenance. The previous
audit verified52 recorded source files against submitted408a61f. These are
completed encoding jobs, not a final official CPU score or a live single-query
latency claim. Preserve SQLite inputs, logs and manifests; do not recommit large
caches or silently re-encode a valid negative result.

## Resource and completion boundaries

The Mac has8 GiB physical RAM, below the retained10.85 GB hands-on peak RSS.
No heavy accuracy-tier run is certified on this Mac. Existing Kaggle sessions
record31.35 GiB RAM and4 logical CPUs. A distinct private notebook can run
focused Linux tests, the real-model cold Python check, exact cache/shard gates,
30 live checks and one complete official scoring pass. The user's 2026-10-01
clarification authorizes that Kaggle upload and launch. Cold check results are
functional evidence on constructed new code, not unseen-accuracy estimates.

Before any publication, complete the remaining real-model/Linux/final-P0 checks,
verify source and artifacts, review the video content and AI form, and obtain
both the team's amendment decision and explicit user authorization for the
specific remote actions. No agent may sign human ethics declarations or infer
Samsung permission from silence. Existing release, tag and public video remain
unchanged. GitHub CI and a Docker build on the corrected source are unverified
until actually executed; local checks are not remotely green CI.

## Prepared continuation

After the local source freeze, build without `--allow-dirty`:

```sh
python scripts/build_submission_kernel.py --owner lifemeansfun --stage p0 \
  --kernel-name codedoctor-correction-p0 --cold-python-first \
  --out build/local-correction/kaggle-p0
```

The notebook stays private, disables GPU query inference, consumes the existing
document and four query-kernel outputs, and uses an isolated virtual environment.
It runs tests and the cold check before touching the final evaluation. Its
launch is authorized by the protocol amendment above. Preserve any failed attempt and its
cause. Never repeat a valid completed final run to improve the result.
