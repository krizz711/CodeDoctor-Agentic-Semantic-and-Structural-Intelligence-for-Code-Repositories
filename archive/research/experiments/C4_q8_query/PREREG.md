# C4-E1 preregistration: F2LLM-v2-1.7B Q8_0 query encoder

Written 2026-09-29 **before any C4 decision run**. This file must be committed
before encoding decision queries. Implementation and dry tests may precede it.
No result is claimed here.

## Question and context

Can a pinned llama.cpp Q8_0 copy of the **same** F2LLM-v2-1.7B query model
reduce CPU retrieval cost while preserving the current candidate ranking?
This tests one treatment, with F32 GGUF only as a conversion control. The
stored document vectors, document encoder, scorer and execution verifier stay
fixed. C3-E2 OpenVINO's repeat was void under S2 and stopped under addendum A3;
its treatment observations are not a valid negative benchmark for this method.

Baseline: `v3` at `7d64a803fc50f23582d566575e1190f540b216d2`,
`config/candidate_v2.yaml` SHA256
`7f4b1d195b17a6a6f6f00ad43ddee074f7b0f784ed54e5b13d3d91548e3374c9`.
The baseline includes accepted C3 blank fallback and judge-like I/O fixes.
The proposed change is experimental and default off. `config/frozen.yaml` and
the submitted 0.96558 score remain unchanged.

The strongest alternative is keeping current FP32 Torch. It has known quality
and avoids conversion/runtime complexity. Q8_0 is worth adopting only if its
measured cost improvement clears all gates. A key failure mode is tokenizer,
EOS or pooling mismatch; another is that a faster forward pass still leaves
transport, execution or memory too costly. The cheapest falsifying check is
100 TUNE queries, exact token IDs and F32 control, then Q8 fidelity and time.

## Jury wording relevant to this decision

FAQ A: “You are allowed to use the GPU during pre/post processing (including
incremental updates) phase. However, we would like you to make it as
efficient as possible. 

During retrieval, you should make minimal use of the GPU (ideally use the CPU
only).” This experiment uses CPU retrieval; offline indexing may use GPU.

FAQ B: “Usage of an API during the retrieval also counts as utilizing the GPU.
Please keep in mind that the objective of this theme is to develop an efficient
and effective code retrieval process, relying on a large model during retrieval
is not efficient as this cost is incurred for every query and it is not
one-time (unlike LLM usage during pre/post processing).
We do not have a restriction on using closed-source or commercial APIs during
the pre/post processing phase, but as mentioned earlier, these are also subject
to evaluation. You should to try to make these stages as efficient as possible
as well.” The local llama.cpp process is not a remote service, but 1.7B query
inference remains a recurring cost that this experiment must reduce.

FAQ on unseen data: “While using static analysis methods such as regex parsing
are well within the scope of the theme and expected to part of the solution,
we would advise against overfitting too much on the APPS dataset as we will be
doing hands-on eval with a unseen dataset (which will be representative of the
APPS dataset). Dynamic analysis, such as execution of code is also in scope and
allowed, however keep in mind that your approach should ideally generalize and
not overfit on a dataset. An overfit system has little usability outside of
it's training dataset and is mostly considered a failure.” Using the same
model and verifier reduces mechanism changes; DEV has prior exposure and is a
regression check, not fresh unseen confirmation.

FAQ on re-run: “We would be verifying the submission by re-running the MTEB
evaluation. After that, we would be performing another evaluation using an
unseen dataset (which would also be in Python, please ignore the statement
about Javascript in the PPT).” The scored MTEB entrypoint must accept the
experimental adapter, but no final test run is authorized here.

FAQ P1: “For the P1 goal, an incremental approach might be more efficient over
rebuilding the entire index, however, we do not have any restrictions on the
approaches taken (a non-incremental approach is also allowed). However, we
would consider the time and resource utilization during the evalution.” Cache
miss indexing remains the original document path and will be smoke-tested;
its offline cost is reported separately. The FAQ gives no numeric latency or
RAM threshold. The gates below are team-selected practical thresholds.

## Fixed intervention and data boundary

Use the project-pinned `codefuse-ai/F2LLM-v2-1.7B` revision
`3766d46e7a68545ed6190c15330983f9b39ab718`. Convert it ourselves to F32
GGUF and Q8_0 with llama.cpp release `b10269` (prefix `1c3c967`; resolve full
commit). Record conversion command, checkpoint/source/binary/GGUF hashes and
package versions. One model, one quantisation treatment, no tuning sweep.

Use a persistent application-managed localhost llama.cpp process, CPU only,
GPU offload 0, no query/prefix-result reuse. Include tokenization and transport
in timing. Preserve the current instruction prefix, HF token IDs including
terminal `<|im_end|>` (151645), 8192-token cap, last-token pooling, L2-normalised
2048-dimensional vectors and ordered input. Fail closed on errors; no silent
Torch fallback. Document encoding/cache keys use the original encoder.

Use only APPS-train TUNE and DEV assignments. Stage 1 samples 100 seeded TUNE
queries (`20260929`) plus synthetic Unicode, whitespace, empty-content and
boundary cases. Stage 2 uses all 2,500 TUNE and all 2,500 DEV queries. DEV has
prior exposure. Reused EXT-dev is only a warning/veto: a wholly negative 95%
interval blocks adoption but a positive result is not fresh confirmation.
APPS test and EXT-lockbox stay sealed. Before data access, audit the exact file
paths: `get_train_split()` is not sufficient isolation because its loader also
reads test files. Use an explicit train-only artifact allowlist.

## Sequential gates and statistics

1. **Conversion control.** Exact token-ID parity on stage-1 inputs and edge
   cases. Fresh FP32 Torch on the same CPU versus F32 GGUF: median cosine
   ≥0.9999, minimum ≥0.999. A failed F32 control is void.
2. **Treatment screen.** Q8_0 versus fresh Torch FP32: median cosine ≥0.995,
   minimum ≥0.98. Q8_0 must not be slower or use more process-tree peak RSS on
   the stage-1 warm run. Failure is a valid negative treatment result if
   controls pass. Similarity alone cannot approve adoption.
3. **Ranking.** Score all TUNE and DEV query vectors against existing document
   vectors with the **current** candidate configuration. Execution records can
   be reused only with matching parser and execution settings; fill other
   required pairs live. Incomplete replay cannot pass. Live versus replay on
   100 seeded TUNE statements must match at least 99 top-10 rankings, with
   every mismatch investigated. Report dense and final NDCG@10, MRR, recall,
   per-family results and changed-statement counts. A shuffled-query-vector
   control must fall below 0.05 dense NDCG@10, or investigate alignment.
4. **Quality decision.** On each of TUNE and DEV, the paired 95% bootstrap
   lower bound for final-pipeline ΔNDCG@10 must exceed −0.002 (2,000 resamples,
   seed 7, statement as pairing unit). Report a paired sign-flip test as a
   diagnostic; nonsignificance is not non-inferiority. Reject statistically
   significant family regressions after Holm correction at 0.05. Inspect the
   count of changed statements and the attainable effect before interpreting
   confidence intervals.
5. **Performance.** One Kaggle CPU session; same seeded 100 DEV queries in
   four alternating blocks of 25, repeated three times. Load one model at a
   time, equal effective CPU threads, fixed warm-up, no competing heavy job.
   Measure tokenization through returned vector and full retrieval through
   final ranking. Report cold startup separately; mean/p95, CPU/ISA, threads,
   and total process-tree peak RSS. Require ≥1.5× mean encoding speedup,
   ≥25% lower mean end-to-end latency, no worse end-to-end p95, and ≥30% lower
   peak RSS. Paired bootstrap lower confidence bounds must clear both speed
   thresholds. All prior gates must pass.

Any failed baseline/alignment control makes the run void. One retry is allowed
only for a documented infrastructure/control failure after a specific fix. Do
not repeat a valid negative result to seek a better outcome. If full evidence
cannot be collected, mark blocked/incomplete; do not shrink the decision set
or relax thresholds. Budget 36 hours of work, reserving 12 for packaging;
conversion/setup troubleshooting ≤4 hours and experiment execution ≤16 elapsed
hours. These are caps, not predicted runtimes.

## Delivery and evidence

Preserve raw logs, per-query results, attempted runs, commands, exit status,
artifact hashes, data exposure, versions and machine metadata. Prepare a
ready-to-run Kaggle notebook using private artifacts. Run focused adapter,
config, recipe/cache and pipeline tests and real-model smoke checks on Linux
and local Mac where feasible. Claims name the machines actually measured.
If all gates pass, package GGUF, checksum, runtime pin and one-command launcher,
and verify uncached document indexing followed by accelerated retrieval.
Candidate adoption does not change the submitted score. Final APPS test,
release and merge remain a team decision.

Result: pending. Protocol commit: pending until this file is committed.
