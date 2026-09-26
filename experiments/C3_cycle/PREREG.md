# Cycle 3: three experiments, pre-registered

- **Written:** 2026-09-26. Committed **before** any treatment in this file is run on its decision data.
- **Evidence behind the choice:**
  - `research_reports.json`: three researchers, one per scoring goal;
  - `research_scripts/`;
  - the skeptic review (`skeptic_review.md`, `skeptic_reports.json`);
  - an independent review of this plan (`prereg_reviews.json`). Every REQUIRED amendment from the two reviewers (statistics/bias; scope/feasibility) is adopted, and most RECOMMENDED ones.
- **Committed together with this file** (the analysis is frozen before DEV is replayed):
  - `scripts/exp_e1_blank_fallback.py`, `scripts/record_blank_fallback.py`, `scripts/c3_stats.py`, `scripts/union_replay.py`;
  - the diffs to `app/pipeline/execution_rerank.py` and `scripts/replay_outcomes.py`;
  - `tests/test_blank_line_fallback.py`.
- Analysis code changed after this commit is listed in an addendum **before** its output is read.

## Why these three: the scope they serve
| # | experiment | brief / FAQ wording it serves |
|---|---|---|
| E1 | Verifier fidelity: re-run samples with blank lines collapsed | P0 screening ("performance on the test split of the CoIR apps dataset"); FAQ: dynamic analysis is "in scope … your approach should ideally generalize" |
| E2 | A CPU-quantised 1.7B query encoder, plus a single-machine end-to-end cost | Brief: "expected on run on CPU with minimal GPU resource utilization"; the demo should show "how fast your solution is"; FAQ A: "(ideally use the CPU only)"; FAQ B: "relying on a large model during retrieval is not efficient" |
| E3 | P1: incremental re-index that works in the demo, measured end to end | Brief: "rebuild any indexes, caches, etc for any version/change in a reasonable amount of time"; FAQ: "we would consider the time and resource utilization during the evalution" |

## Rejected in the research phase (not run again)
| idea | evidence | where |
|---|---|---|
| Map a small query encoder into the 1.7B document space: ridge, fused, unpaired (vec2vec) | The 0.6B map scores DEV 0.89618, below native 0.6B (0.90056); fused 0.90765 (−0.0203 vs 1.7B). The negative control collapses to 0.001. **Reproduced independently** (identical to 5 decimals). | `research_scripts/qmap.py` |
| torch dynamic int8; weight-only int4 | cosine 0.21; 0.53–0.60 (0.6B) | `int8_probe.py`, `wq_probe*.py` |
| 0.6B → 1.7B cascade | 56–57% escalation; 7–9% saving | `cascade_*.json` |
| Dense Rocchio PRF | worse at every setting on TUNE | `prf.py` |
| Crash demotion, perturbation tie-break, identifier bonus | −0.012; 0.892 → 0.853; null or harmful | `soft.py`, `perturb.py`, `ident2.py` |
| Offline LLM spec cards (PRE-4) | Prior art shows ~0 gain for strong LLM embedders; the target class is ~0% of the test split | report C3 |
| Bonus near-miss tie-break (MBR-exec) | pilot 0.46 vs chance 0.33 (n = 33); **queued for the next cycle**, not rejected | `pilot_nearmiss*.json` |

## Common protocol (applies to E1–E3)
- **Data:**
  - APPS **train** only: DEV and TUNE halves, seed 20260919. **DEV decides**; TUNE was used for design.
  - **No APPS test-split file is opened.** The EXT-lockbox stays sealed.
  - CodeContests EXT-dev is used only descriptively or as a veto: it has been reused adaptively (STRESS_TEST F6), so it never supports a claim.
- **Metrics:**
  - NDCG@10 is primary, the screening metric.
  - MRR@10 is reported for every arm (the brief measures "NDCG@10" and "MRR").
  - With one relevant document per query, per-statement NDCG and MRR changes share their sign.
- **Statistics:**
  - Paired bootstrap over statements: 2,000 resamples, seed 7, 95% percentile interval.
  - Paired sign-flip permutation test, **two-sided**, on the sum of paired differences T. p = (flips with |T\*| ≥ |T|, including the identity) / (number of flips). Exact enumeration when ≤ 20 statements change; otherwise 100,000 random flips, seed 7.
  - Statements up and down are always printed next to Δ.
  - Each experiment has exactly **one** primary test. E2's configurations are tested in a fixed sequence, and only the first to pass the TUNE gate reaches DEV.
  - E1 is superiority, E2 non-inferiority, E3 descriptive. These are separate questions, decided separately.
- **Re-runs:**
  - Each decision run is performed **once**. A run voided by a sanity control or stop rule may be repeated **at most once**, only after the cause is written down and fixed in a commit made before the repeat.
  - Every attempt is logged. The first non-void attempt is the result. A non-void run is never repeated to change its outcome.
  - After two void attempts the experiment stops and reports the blocker.
- **Seeds:**
  - bootstrap and sign-flip: 7;
  - E1 placebo derangement: 20260926, over the sorted affected qids;
  - E1 S3 sample: 20260926;
  - E2 DEV halving and DEV timing sample: 20260926.
- **Order and isolation:**
  - E1 runs first, then E2's local steps (`fill_missing`, replay), then E3 Part 3 last.
  - No two of these, and no other heavy process, run concurrently on the laptop: E1 and `fill_missing` use wall-clock timeouts, and E3 is a timing run. Kaggle work may overlap with laptop work.
  - E2's baseline is candidate_v2 **as of this commit** (blank fallback off), whatever E1 decides.
- **Baselines** (reproduced exactly by `scripts/union_replay.py` and re-checked independently by the statistics reviewer):

| DEV (2,500) | NDCG@10 | MRR@10 | stdin family (882) NDCG@10 | stdin MRR@10 |
|---|---|---|---|---|
| dense 1.7B | 0.92792 | 0.91414 | 0.90155 | – |
| frozen rule (submitted), replayed | 0.94846 | 0.93940 | 0.95978 | 0.95331 |
| **candidate_v2 as replayed** (call-style + deepening to 50, numeric 1e-6) | **0.95327** | **0.94487** | **0.95993** | **0.95312** |

  TUNE: candidate_v2 0.95849, stdin 0.96362. The replay does not model `execution_skip_interactive` (2 of 5,000 statements) or CPU-time limits; both are live-only.
- **A run that misses a sanity control is void, not "inconclusive".**
- **Nothing here edits `config/frozen.yaml`.** New behaviour sits behind default-off switches. Adopting anything into the screening submission is a separate final decision, which needs its own logged test-split run.

### Seen before this pre-registration (disclosure)
- **Research and skeptic phases:**
  - **DEV:**
    - blank-line prevalence and gold-crash counts, which used DEV qrels (35/824 interleaved, 29/35 gold crashes);
    - the loss decomposition, the query-map and cascade replays (both rejected), and the E5 lineage reproduction.
  - **TUNE:**
    - E1's live effect. The **any-blank detector** and the **consumption gate** were chosen on TUNE, after the q1090 regression was seen.
    - My implementation was then checked on TUNE against the skeptic's independent code (`E1_result_tune.json`). It gives exactly the skeptic's numbers: 28 gold recoveries, 9 non-gold, 9 up / 0 down, placebo 0.
  - **EXT-dev:** E1 arm B probes (7/9 crashing golds fixed).
- **Myself and the reviewers:**
  - DEV baselines;
  - DEV's affected-statement count for E1 (46; 44 stdin), baseline on them, ceilings, and how many golds are already at rank 1;
  - P0's DEV analyses (unrelated treatments).

---

## E1: verifier fidelity (P0)

**Hypothesis.** When a stdin sample input contains blank lines, some correct programs crash or mismatch because of them. APPS's official inputs for these problems contain none: 24/24 TUNE cases equal the collapsed sample. Re-running a failed candidate once on the collapsed input adds evidence and raises NDCG@10.

**Treatment (arm A), a default-off switch `execution_blank_line_fallback` (`ExecutionConfig.blank_line_fallback`):**
- **When it fires.** For a statement whose stdin examples are used by the policy (`legacy_examples > 0`), consider each of the **first two** stdin examples whose input has a whitespace-only line inside `stdin.strip("\n")`. A candidate whose raw run on that example does not match gets **one** re-run on the input with every whitespace-only line removed.
- **Raw verifications are never removed.**
- **Consumption gate.** A fallback match counts only if the tracked input consumption covers every non-blank character (`read ≥ len(collapsed.rstrip())`, as in the replay's `_consumed_exactly`). Untracked fallback runs are accepted and counted.
- **Where it applies.** The same rule runs live (`verify_candidate`) and in replay: a sidecar written by `record_blank_fallback.py`, and `Policy.blank_fallback`.

**Primary metric.** DEV **stdin family** NDCG@10 (882 statements): candidate_v2 as replayed, with vs without arm A, paired by statement. Baseline **0.95993**.
- **Why stdin (construct validity, fixed before DEV).** The screening split is ~99% stdin statements (STRESS_TEST F5), and arm A acts only on stdin examples.
- **What the bar means on this scale.** The +0.003 bar was set for all-family deltas. On the 882-statement stdin family it corresponds to a net summed gain of 2.65 NDCG@10 units, i.e. +0.00106 on the all-family scale: about 2.8× lower in absolute terms than the bar earlier decisions used.
- **Reported beside the primary:** the all-family Δ, its CI and its p.
- **What the stdin Δ is not:** a forecast of the screening delta. The prevalence of blank-line samples in the test split is unknown and is not measured (the file stays closed).

**Decision rule.** Adopt into `candidate_v2.yaml` (switch on; frozen untouched) iff **all** hold:
1. Δ stdin NDCG@10 ≥ **+0.003**.
2. The 95% bootstrap CI excludes 0.
3. Two-sided sign-flip p < 0.05.
4. At least 6 statements change. This is implied by rule 3: with 5 changed statements, even all up, exact p = 2/32 = 0.0625.
5. All-family Δ ≥ 0.
6. At most 1 statement in which a fallback-only non-gold verification displaces a gold that was at rank 1.

**Claim scope.** A pass licenses only "raises DEV stdin NDCG@10 by X (n up / m down)", described as a fallback for blank lines in rendered sample inputs (seen in APPS's CodeChef-rendered statements).
- No screening or test-split gain is projected.
- The README keeps the screening number of the frozen configuration until a logged test-split run of an adopted configuration exists.

**Sanity controls:**
- **S1:** the unmodified replay reproduces DEV 0.95327 / stdin 0.95993 exactly.
- **S2:** passes iff, on TUNE, gold recoveries are 28 ± 1, statements up 9 ± 1, and statements down ≤ 1. This is already run: 28, 9 and 0, reported in `E1_result_tune.json`.
- **S3:** 200 raw runs are drawn (seed 20260926) from the **affected** DEV statements' used examples, including every recorded raw timeout. It passes iff ≥ 198/200 re-executed verdicts match the recorded ones. A re-run that times out counts as "no match" and is not re-run.
- **S4, live = replay:** for every affected DEV statement, the live `apply_execution_rerank` top-10 with the switch on equals the replayed top-10 in ≥ n − 1 statements.
  - The **replay** is the primary number; live is a control.
  - If the tolerated mismatch falls on a statement whose NDCG changes in the primary, the decision must also hold with that statement scored from its live ranking. Otherwise E1 is not adopted.

**Negative control (must fail).** The same machinery, where every affected example's input is the collapsed **first** example of a **different** affected statement: a derangement over the sorted affected qids, seed 20260926 (`record_blank_fallback.py --placebo`). It is judged against the statement's own expected output. It must recover ≤ 1 gold and give Δ ≤ 0. If it "improves", the mechanism claim is void.

**Retry control (descriptive; it attributes the gain).** The same machinery with the **uncollapsed** input: one extra run per candidate that received a fallback run. The gain credited to collapsing is the treatment's recoveries minus the retry control's. Any gold or rank-1 the retry control moves on its own is listed.

**Power and ceiling:**
- 46 affected DEV statements (44 stdin); baseline NDCG@10 on them 0.781.
- Of the 44 stdin statements, 31 golds are already at rank 1. Of the other 13, one gold is outside the 50-candidate pool, so **at most 12 statements can go up**.
- Reachable ceiling on the stdin family: **+0.0092** (8.09 units, 3.06× the bar). The +0.0103 figure counts the out-of-pool gold, which no fallback can reach.
- Expected: +0.004 to +0.008 from at most 12 upward changes. This is an estimate from TUNE, where the detector and the gate were chosen, so it is optimistic (winner's curse). TUNE's reachable stdin ceiling was +0.0057.

**Runtime and stop rule.**
- About 2–4k fallback runs; expected under 5 minutes at 6 workers (the recorder's count).
- Void if the run exceeds 15 minutes or `spawn_failures() > 0`.
- Pre-flight: ≥ 3 GB free RAM, recorded.

**Secondary (descriptive):**
- **Frozen rule + arm A;** MRR@10.
- **Sensitivity:** the primary recomputed with untracked fallback runs **rejected**. If the decision differs, that is stated next to the result.
- **Cost:** extra sandbox executions per affected statement, and the added wall-clock per affected query (p50/p95 at 6 workers, from S4's live runs), reported next to Δ in any README line.
- **Meaningful blank lines:** the fallback's outcome on every affected statement whose text mentions "blank line" or "empty line", on DEV and EXT-dev.
- **EXT-dev.**

**Arm B (fidelity fix, no score claim; not a decision experiment).**
- **What it does:**
  - stdin is given to the child as a temp-file handle rather than a pipe;
  - `exit` and `quit` are bound to `sys.exit` in the runner's builtins (not by prepending source).
- **Default-off switch:** `execution_judge_like_io`.
- **Order:** E1's primary and its controls run with arm B **off**.
- **Accepted only if both hold:**
  - New sandbox tests (fstat idiom, bare `exit`, `exit('msg')` → rc 1, a `from __future__` file) pass on the Windows, ubuntu and macOS CI jobs.
  - A DEV re-run with arm B on shows 0 statements with lower NDCG. The re-run covers every recorded stdin run that did not verify, for candidates whose code matches the exit/quit or `fstat(0)`/`os.read(0)` idioms (the selection in `skeptic_scripts/armb_tune.py`), and is scored by replay. Run count and runtime are reported; stop at 30 minutes. Candidates outside the selection are not re-run, and the result says it covers only them.
- EXT-dev's 7/9 is in-sample and descriptive.

---

## E2: a CPU-quantised 1.7B query encoder, with a single-machine end-to-end cost (FAQ A/B)

**Hypothesis.**
- **Quality:** an OpenVINO int8 build of F2LLM-v2-1.7B, used **for queries only**, is non-inferior in DEV pipeline NDCG@10. The documents keep the stored vectors.
- **Efficiency:** it is clearly cheaper on CPU than float32 torch.

**Runtime.**
- OpenVINO through sentence-transformers `backend="openvino"`. It keeps the HF tokenizer (its `<|im_end|>` post-processor), last-token pooling and normalisation; `_verify_recipe_applied` is reused.
- llama.cpp is the fallback only if OpenVINO cannot export or run. That switch would be an addendum committed before any DEV encoding.

**Where it runs: two kernel pushes, both on the Kaggle CPU class R2 used (2 physical cores).**
- **Kernel A:**
  - Runs export, S1–S6, NEG, the TUNE gate, REF/REF06 timings and the end-to-end block.
  - **Inputs:** the 100 R2 TUNE query texts with their stored 1.7B/0.6B vectors, the recorded top-50 pools of those statements, and the corpus texts. They are delivered as a **private** Kaggle dataset, never embedded in the script.
  - The IR is saved as kernel output.
  - The gate result is **committed locally before kernel B is pushed**.
- **Kernel B:**
  - Uses kernel A's IR (as a kernel source). It must reproduce kernel A's TUNE-100 vectors at min cosine ≥ 0.99999, or the run is void.
  - Then it times the adoption block (below) and encodes the DEV (2,500) and EXT-dev (108) query texts, which come from the same private dataset.
- **Data rules:** no qrels, no test-split query file and no lockbox file are uploaded. The dataset's file list is committed with the result.
- **Isolation:**
  - Each arm is timed in its **own fresh subprocess** that loads only that arm's model (for OpenVINO, the saved IR; never the exporting process).
  - Export and nncf compression run in a separate process. Their time and peak RSS are reported as a pre-processing cost, never as query-time RSS.
- **Recorded per session:**
  - CPU model string and `/proc/cpuinfo` flags (avx2, avx512f, avx512_vnni, avx_vnni, amx);
  - torch threads measured inside `CodeEncoder.encode`;
  - OpenVINO `INFERENCE_NUM_THREADS` and `DYNAMIC_QUANTIZATION_GROUP_SIZE` of the compiled model;
  - `pip freeze`.

**Arms:**

| arm | what | used for |
|---|---|---|
| REF | torch float32, 1.7B | speed baseline; known number 7.83 s ± 25% (R2) |
| REF06 | torch float32, 0.6B | demo-tier speed comparator (R2: 2.79 s) |
| C0 | OpenVINO float32 export | sanity: must reproduce the stored vectors |
| T1 | OpenVINO: nncf int8 **sym, group 32** weights; the plugin's default dynamic activation quantisation | first candidate |
| T1b | T1 with `down_proj` and `v_proj` unquantised (ignored scope) | only if T1 fails the TUNE gate |
| NEG | T1 with the terminal EOS removed | negative control |

If nncf/optimum-intel rejects group size 32 for 8-bit weights, T1 is INT8_SYM per-channel (group −1) and T1b keeps that mode. The substitution is recorded, not chosen.

**Timing protocol (for every ratio).**
- REF and T are timed in the **same kernel session**, interleaved in blocks over the same 100 queries (REF06, T, REF, T, REF06, …), with one warm-up query discarded per arm.
- ratio = mean(T) / mean(REF), with a paired bootstrap 95% CI over queries.

**TUNE gate.** On the 100 seeded R2 TUNE statements; this is the only place a configuration is chosen.
- **Fidelity:** cosine to the stored vectors, median ≥ 0.995 and min ≥ 0.98.
- **Efficiency:** (ratio ≤ 0.77 **and** CI upper bound < 1.0), **or** (peak RSS ≤ 4.5 GB **and** ratio ≤ 1.00 **and** p95 ≤ 1.10 × REF's p95).
- Try T1, then T1b. The first to pass goes to kernel B. If neither passes, stop and report the negative result.

**DEV.**
- Kernel B encodes the DEV and EXT-dev query texts with the chosen T. Locally: dense scores against the stored corpus vectors, then the candidate_v2 policy through `union_replay`.
- Pairs never recorded before are executed with `fill_missing` (the recorder's configuration). Its fidelity is checked on 100 recorded pairs from those statements (seed 20260926, stratified to include every recorded timeout): ≥ 99/100 identical verdicts.
- Reported: how many statements' treated rankings depend on a `fill_missing` pair, and the primary recomputed without them.

**Primary metric.** DEV all-family **pipeline** NDCG@10 (candidate_v2 as replayed), quantised queries vs stored 1.7B queries, paired. Baseline **0.95327**.

**Decision rule, quality.** Non-inferior iff **all** hold:
1. The 95% bootstrap CI lower bound of Δ > **−0.002**.
2. There is no significant decline: not (Δ < 0 and two-sided sign-flip p < 0.05).
3. The stdin-family (882) Δ point estimate is ≥ −0.002, with its CI and up/down counts reported. This guards the construct the screening split consists of.
4. The EXT-dev Δ CI upper bound is ≥ 0. This is a veto only.

A CI above 0 is **never** reported as a gain: a pure-noise perturbation produced +0.0014 [+0.0003, +0.0027] on TUNE. A failure is reported as "non-inferiority not shown (a null treatment fails this bound an estimated 13–36% of the time)", not as "int8 is worse".

**Decision rule, outcome.**
- **Adoption timing.** Measured on 100 seeded DEV query texts (seed 20260926), timed in kernel B with the interleaved protocol **before** the DEV encode. The bulk DEV encode is never used for timing.
- **Adopt as the CPU operating point for 1.7B** (switch `query_backend: openvino` in the candidate config; screening unchanged) iff quality is non-inferior **and** efficiency holds on the adoption block: (ratio ≤ 0.67 and CI upper < 1.0) **or** (peak RSS ≤ 4.5 GB and ratio ≤ 1.00 and p95 ≤ 1.10 × REF's p95).
- **Adoption also requires engineering:**
  - `_LOADED_MODELS` in `app/retrieval/encode.py` keyed on backend as well;
  - a test, failing before the fix, proving that with `query_backend: openvino` the document encoding and `document_cache_key` use the torch model.
- **Also replace 0.6B as the demo tier** iff, in addition:
  - mean latency ≤ 1.1 × REF06 in the same session;
  - and the same holds on the demo laptop (≥ 3 GB free, nothing else running, the same 100 TUNE texts).
- **Otherwise** the quantified trade-off (Δ NDCG, seconds, GB) goes to the user.

**Single-machine end-to-end cost** (Dr. Ortega's request; descriptive; in kernel A).
- For REF, REF06 and T, on the 100 TUNE statements, each stage timed per query:
  - tokenise and encode;
  - dense search over 8,765 × d;
  - the live execution pass under candidate_v2 settings on the recorded pools, workers = physical cores, run once per statement and labelled as shared across arms;
  - the total.
- Reported: p50, p95, mean and each subprocess's peak RSS.

**Sanity controls:**
- **S1:** C0 median cosine ≥ 0.9999 and min ≥ 0.999.
- **S2:** REF within ±25% of 7.83 s.
- **S3:** every query's token ids end in exactly one 151645.
- **S4:** nncf is installed and the exported config shows the intended weight mode; a silent float32 fallback fails.
- **S5:** the A0 replay reproduces 0.95327.
- **S6:** after installing openvino/nncf/optimum-intel, `transformers==4.57.6` and `sentence-transformers==5.0.0` (the R2 pins) are still installed, or the run is void. The exact openvino, nncf, optimum and optimum-intel versions are pinned in the kernel builder before kernel A runs.
- **Negative control:** NEG median cosine < 0.9.

**Stop rules.**
- Stop before kernel B if the TUNE gate fails.
- If the DEV encode projects over 6 h from its first 50 queries, the decision to halve is made from the projected time alone, before any DEV NDCG exists. The subset is the first 1,250 qids after sorting the DEV qids and shuffling them with `random.Random(20260926)`.

**Scope limits.**
- (a) Any speed or RAM claim names the CPU model and whether avx512_vnni or AMX was present.
- (b) While `query_backend: openvino` is only in the candidate config, the README may not describe int8 query encoding as the path behind the screening number or the one the judges' MTEB re-run uses.
- (c) A default-off switch changes nothing in the judges' re-run; turning it on there is a final decision.

**Secondary (descriptive):**
- the projected query-encoding time of the judges' MTEB re-run on the measured CPU (3,765 × mean latency) for REF, REF06 and T, **labelled estimate** (R2's estimate: 8.2 h at 1.7B);
- the IR's size and how judges would obtain it (release asset, or export on their machine, with export time and peak RSS);
- the 1.7B per-document CPU rate.

---

## E3: P1, incremental re-index that works in the demo, measured end to end

The skeptic found that the demo **cannot re-index a changed folder in-process**:
- the workbench's `start_build` returns early on a built target;
- the explorer's re-index skips embeddings, which switches dense retrieval off after any edit;
- the demo embedding cache is keyed on content only (`compute_embedding_key` defaults its revisions to "v1"). A same-dimension model or recipe change serves stale vectors silently (3 failing scratch tests).

**Part 1: correctness (not an experiment).**
- One recipe-aware key function (model@revision, document prefix, max_seq, EOS, normalisation), shared by `PrePostPipelineEncoder.document_cache_key` (scored-path keys stay **byte-identical**, tested) and all five demo call sites.
- A clear error when cached vectors disagree in dimension.
- The three scratch tests promoted to `tests/` as regressions; each fails before the fix.
- Changing the key invalidates existing demo cache entries. The one-time re-warm on (R) is the full-rebuild baseline, and the demo cache is warmed before any demo recording.

**Part 2: engineering.**
- A judge-reproducible rebuild path: edit a file, trigger a rebuild in-process (a `rebuild` flag on the workbench build and its endpoint), and get dense results for the new content.
- The explorer's re-index either computes embeddings or says that dense retrieval is off.
- Tests: the edited function is retrievable and the stale text is not.

**Part 3: measurement (descriptive, with a claim rule).**
- **Operating point:**
  - F2LLM-v2-0.6B, CPU, float32, batch 1 at the call site as shipped;
  - **torch threads as the shipped code sets them** inside `CodeEncoder.encode` (`os.cpu_count()` = 12 logical here), measured and recorded;
  - this laptop (i5-11260H, 6C/12T).
- **Pre-flight:** ≥ 3 GB free RAM. Otherwise I stop and ask the user to close applications. A run started below 3 GB is not counted as a void attempt, and Part 3 is not moved to another machine.
- **Corpora:**
  - **(R)** this repo's `app/`, through the demo workbench path;
  - **(A)** the APPS corpus through the scored path's corpus cache (`--corpus-cache`). Changed documents are drawn from TUNE golds, seed 20260926.
- **Change sets, 3 repetitions each:**
  - k = 0, 1, 10 and 100 modified function bodies;
  - 1 added file; 1 deleted file;
  - 1 pure move: the file is moved with byte-identical content, so 0 encodes are expected.
- **Timed stages:** load, walk/hash, chunk, lexical build, edges, cache lookup, encode of misses, cache write, publish, and end to end from edit to first dense answer. Peak RSS is recorded.
- **Statistics:** each change set reports the median of its 3 repetitions. The slope is OLS of end-to-end time on the **number of chunks actually re-encoded** (the miss count) over all 12 k-runs. Chunks re-encoded per edit are reported (one edit can re-encode 1–3 chunks: the function, its enclosing class, the file summary).
- **Baseline:** a full rebuild at the same operating point with an empty cache, measured on (R). On (A) it is the measured per-document rate × 8,765, **labelled as an estimate**.

**Claim rule** (on (R) only; (A) is descriptive). The README and deck may say "re-embeds only what changed", with the numbers, **only if** all hold:
1. In every run the number of re-encoded chunks equals the number of chunks whose encoder input changed (0 for a delete and for a pure move).
2. The fitted slope is within ±30% of the measured per-document rate.
3. The k = 0 time with the model warm is ≤ 10 s.
4. The k = 10 end-to-end time is ≤ 0.15 × the measured (R) full rebuild.

The k = 100 row is reported but not ruled: on (R), 100 of ≈ 470 chunks is 21% of the corpus.

**Labelling.** Every P1 number states:
- the encoder (F2LLM-v2-0.6B);
- the device (CPU, float32);
- the machine (this laptop);
- the threads as measured;
- the path (workbench, explorer, or `--corpus-cache`).

No P1 timing is claimed for the 1.7B screening encoder, and a Kaggle 1.7B rate never shares a table or sentence with laptop numbers without a machine label.

**Gates (the run is void if any fails):**
- **Parity (R):** incremental vs a measured full rebuild. Cosine ≥ 0.99999 per re-encoded chunk, and 20 preset queries give identical top-10s.
- **Parity (A):**
  - unchanged documents' vectors are bit-identical to the cache;
  - changed documents' vectors match a separate fresh CPU encode of the same texts (cosine ≥ 0.99999);
  - top-10 identity ≥ 99.5% on the 2,500 TUNE queries, dense-only, using the stored 0.6B TUNE query vectors, between the incremental index and an index assembled from scratch from the same vectors plus a from-scratch lexical build.
  - Parity is never measured against the GPU-seeded vectors, whose mixed-origin top-10 identity (0.9716) is already below 99.5%.
- **Negative:** changing model/revision, prefix, max_seq, EOS or normalisation gives 100% cache misses.
- **Positive:** a pure move gives 100% hits and 0 encodes.
- **Sanity:**
  - the 0.6B batch-1 rate is within ±25% of the skeptic's 1.275 s per document;
  - the warm model load is within 2× of 13.9 s.

**Stop rule.**
- Each incremental run is killed at 10 minutes; the (R) full-rebuild baseline at 30 minutes.
- A run is void if s/doc > 3.0 (paging) or free RAM falls below 1 GB.
- After 2 void runs, stop and report the blocker.

**Secondary (descriptive):**
- the mixed-origin effect (GPU bf16 seed vectors plus CPU fp32 updates; skeptic, TUNE: top-10 order identity 0.9716, ΔNDCG@10 +3.9e-6);
- batch 1 vs the workbench's windows of 8;
- the 1.7B per-document CPU rate from E2's kernel, machine-labelled.
