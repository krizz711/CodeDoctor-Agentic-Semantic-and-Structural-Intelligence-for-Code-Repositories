# CodeDoctor v2: pre/post-processing architecture investigation and plan

## Context

**Where `main` stands (@ `d609421`):**
- The screening number, 0.96558, is F2LLM-v2-1.7B dense (0.93756) plus one post-processor: stdin execution verification.
- Brief axes 1–4 (query and snippet categorization and processing) contribute 0.
- The router, hybrid, chunker and graph are not on the scored path.

**What the jury said** (Samsung FAQ):
- pre/post-processing is what is judged;
- model choice should not matter much;
- contamination will be neutralized with an unseen, APPS-like Python set;
- retrieval must be CPU-only (a query-time LLM or API counts as GPU);
- offline LLM/GPU is fine if efficient.

Competitors can buy the same contaminated encoder. **What separates us is signals a memorizing encoder cannot supply, extracted cheaply.**

**Method used for this investigation:**
- read `main`;
- measured statement and code properties over all 8,765 APPS queries and documents (HF parquet, read into memory only);
- surveyed 2023–2026 work, sources at the end.

Anything not measured is labelled a hypothesis. The earlier findings still stand (macOS `preexec_fn` crash, the false "hybrid RRF = 0.93756" claim, unshipped efficiency levers). They are carried into Part F.

---

## Step 0: `C:\Users\Asus\manish\CLAUDE.md` (first action after approval)

**Decision: include the FAQ verbatim.**
- It is binding, it is external, and it is missing from the repo, whose docs already contradict it.
- A verbatim quote avoids paraphrase drift.

**Costs:**
- About 1.5k tokens per session.
- It applies only to sessions under `C:\Users\Asus\manish`, so clone the repo under it.
- **I will not commit it to the public repo without your OK.**

Content:

````markdown
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
````

---

## Part A: First principles, the signals a contaminated encoder cannot fake

A statement-to-solution pair carries five independent kinds of evidence. Only the first is what the encoder models.

| # | Signal | What it is | Contamination-proof? | Query-time cost | Status on `main` |
|---|---|---|---|---|---|
| S1 | Semantic | The meaning of the story and task, matched to code | **No**: the encoder family trained on APPS | 1 encoder pass | Used (the whole score) |
| S2 | Behavioral | Run the candidate on the statement's worked examples | **Yes** | ~0.08 s per candidate, parallelizable | Used, **stdin only, exact/numeric only** |
| S3 | Judging semantics | Statement rules on what counts as correct: any answer, ε-tolerance, any letter case, modulus | **Yes** | µs (regex) | **Unused.** Blanket 1e-6 tolerance only |
| S4 | Structural I/O | The input grammar in the statement vs the code's read pattern; the output format vs its print pattern | **Yes** | µs per candidate (offline AST card) | Features existed and were rejected against a 1.7B encoder; never used for execution routing |
| S5 | Intent reconstruction | An offline LLM restates each snippet as a task spec, matched NL↔NL | Mostly (generation is not recall of APPS pairs) | One extra dot product | Unused |

**Measured over all 8,765 APPS queries and solutions (train split / test split):**

| Property | Train | Test |
|---|---|---|
| Statement allows **multiple answers / "print any"** | 3.0% | **12.8%** |
| States a float tolerance | 1.0% | 2.6% |
| "Any letter case" accepted | 0.8% | 2.2% |
| Modulus answer | 3.0% | 5.8% |
| T-test-cases input | 22.7% | 24.0% |
| Call-style family (Codewars + LeetCode) | **57.6%** | ~0.9% |

Execution-verifiable today: 32% of train statements, 97.6% of test.

Corpus shapes:
- 62.1% read stdin;
- 8.6% define `class Solution`;
- 28.0% are top-level functions without stdin;
- 99.2% `ast.parse` cleanly;
- median length 332 chars, max 289k.

**What follows:**
- S3 is a free fix for a known loss class. Output mismatch is 60% of gold verification failures on DEV, "mostly multi-answer" per `docs/RETRIEVAL-PIPELINE.md`.
- S2 must learn call-style or it goes blind on a train-like distribution.
- S4 and S5 are what carry queries with no examples.

**Honest ceiling.** On the APPS test split only 236 queries miss rank 1: 111 verified-wrong, 73 nothing-verified with dense wrong, 52 gold outside the top 20. Realistic P0 gains are **≤ +0.01–0.02 NDCG**. The separating value shows up on the unseen set: a weaker dense pass, more call-style problems, possibly more multi-answer ones. So the evaluation centre of gravity must be held-out sets plus an encoder that never saw APPS (Part G).

---

## Part B: The state of the art, filtered by the FAQ

| Line of work | Representative sources | Transfers under the FAQ? | Maps to |
|---|---|---|---|
| Semantic code search by input/output examples | Stolee et al. TOSEM'14; CoSQA+ (test-driven agents) | Yes (CPU) | S2, POST-3/4 |
| Execution-based selection | CodeT, MBR-exec, **LEVER** (ICML'23, a learned verifier over execution results) | Yes: the verifier is tiny, execution is CPU | POST-6 |
| Checkers, generators, validators for competitive programming | **CodeContests+** (EMNLP'25 Findings: checkers plus 13M correct and incorrect solutions), HardTests (ICLR'26), AutoCode | Offline only | Ground truth for POST-3; external data (Part G) |
| Offline index enrichment | **EnrichIndex** (COLM'25, +10.6 NDCG@10, 293× fewer online tokens), Doc2Query++ (dual-index fusion), **CodeBridge** (FSE'25: query↔LLM-comment matching, +21–25% MRR zero-shot), PseudoBridge ('25–'26), ReCo (ACL'24 code style normalization) | **Only the offline half.** Query-side generation is excluded | PRE-4 |
| Adaptive per-query routing and cascades | Adaptive Re-Ranking (Jun 2026: 1.15–53× lower median latency), QPP-based dense/sparse selection | Yes | POST-1 |
| Hubness correction | QB-Norm (CVPR'22), DIS, DualDIS/Sinkhorn ('25) | Yes, with an inductive train-query bank only | PRE-5 |
| Efficient query towers | Query-encoder distillation ('23), KALE ('23), F2LLM-v2 80M–14B plus Matryoshka | Yes | POST-1 tiers |
| Version discrimination | **ExecRetrieval** (Sep 2026: the correct version is ranked below a buggy near-clone in 67–78% of queries) | Yes, via execution | POST-8 |
| Query-time LLM (HyDE, Query2doc, LLM rerankers, CodeRankLLM 7B) | Common | **No** (FAQ answer 2) | Excluded, and the deck says why |

---

## Part C: Proposed architecture, CodeDoctor v2

```
OFFLINE (index time; GPU/LLM allowed; content-addressed, incremental)
  corpus snippet ──► content hash ──┬─► dense vectors: small tier + 1.7B tier        (PRE-6 store)
                                    ├─► snippet card (AST): IO mode, entry+arity,
                                    │     read signature, print literals/format,
                                    │     repaired variant                             (PRE-3)
                                    ├─► spec card (optional, small code LLM):
                                    │     task / input fmt / output fmt / any-answer   (PRE-4)
                                    └─► doc-side hubness penalty (train-query bank)    (PRE-5)

QUERY TIME (CPU only)
  statement ──► PRE-1 parser: sections, examples (stdin or call), judge rules,
  │              input-shape signature, family
  ├─► PRE-2 encoder view (prose, examples routed to the verifier)
  ▼
  POST-1 evidence-driven cascade:
     small-tier encode → top-50 → POST-2 static compatibility (order and prune)
       → POST-3/4/5 parallel execution with a judge-aware comparator and
         consumption/shape instrumentation
       → confident verified hit?  yes → done
                                  no  → escalate: 1.7B encode, rescore top-50,
                                        POST-7 adaptive deepening
  ▼
  POST-6 evidence fusion (LEVER-style monotone scorer over S1–S5) → ranked list + reasons
VERSIONS: POST-8 lineage grouping over unique hashes; rank within a lineage by execution evidence.
```

All of it fits the MTEB hooks:
- `encode()` records texts and returns small-tier vectors.
- `similarity()` does the rest from caches keyed by text.

That is the same contract `app/pipeline/encoder.py` already uses.

---

## Part D: Component catalogue

For each component: the idea, the evidence, where it lives, the honest expected value, and the kill criterion. Everything lands default-off and ships only via Part G.

### Pre-processing

**PRE-1: Statement parser v2** (extends `app/pipeline/query_analysis.py` and `parse_worked_examples` at `app/pipeline/execution_rerank.py:140`)
- **Sections:** story, input spec, output spec, constraints, examples, notes.
- **Judge rules (S3):**
  - any-answer ("multiple answers / print any");
  - ε, parsed from the "absolute or relative error" clause;
  - case-insensitivity;
  - modulus.
- **Examples:** the existing stdin layouts, inline `Input:` lines, `Sample Input N`, CodeContests `Examples`, LeetCode `Input: a = …`, Markdown `**Input:**`, doctest `>>>`, and Codewars `f(x) = y` / `==` / `=>` / `->` / `returns`, `x ==> y`, `name = … output = …`.
- **Input-shape signature:** line count, token types, and a T-test-case pattern.
- **Value:** it enables POST-3/4 and routing. Adds nothing by itself.
- **Kill if:** extraction precision on the V3 gold set is below 0.95.

**PRE-2: Encoder view, examples routed to the verifier.** Encode prose sections only. Measured: examples plus notes are about 20% of test-query characters. This is a CPU-cost lever.
- **Keep only if** non-inferior (Part G), because the encoder was trained on raw statements. The prior is roughly neutral to a slight loss.

**PRE-3: Snippet card, static AST offline** (extends `app/pipeline/code_analysis.py`)
- I/O mode (stdin script, `Solution` method, or functions).
- Entry point and arity.
- Read signature: sequence and count of `input()` / `split()` / loops over T.
- Print signature: literal tokens, `%.Nf` formats, `% MOD`.
- **A repaired variant:** wrap body-only documents. That fixes 19 DEV gold crashes (`'return' outside function` or an unbound `nonlocal`).
- **Value:** cheaper execution (prune incompatible candidates) plus S4 features.

**PRE-4: Spec card, offline LLM (the CodeBridge / EnrichIndex half that is compliant)**
- A small local code LLM (1.5–3B, GPU, batched) or a cheap API writes, per snippet: a story-free task statement, the input format, the output format, the any-answer flag, and algorithm tags.
- It is embedded as a second vector field, with a fusion weight fitted on TUNE.
- **Hypothesis:** little lift on APPS (the encoder already knows APPS), more lift on unseen data and with non-APPS encoders. **Judge it on the external sets and the encoder-robustness arm.**
- **Report the offline cost:** GPU-hours, tokens, dollars.
- **Kill if** there is no significant gain on external data.

**PRE-5: Doc-side hubness penalty**
- CSLS/DIS with a bank of APPS-train queries, never test queries. Test-batch transduction is refused as benchmark-structure exploitation.
- Offline vector; query time costs one subtraction.
- **Prior:** small.

**PRE-6: Content-addressed artifact store** (reuse `app/indexing/cache.py`)
- One hash maps to vectors, snippet card and spec card.
- A new version computes only new hashes.
- This is **P1 on the scored path**, and it lets judges re-run without a GPU for documents.

### Post-processing

**POST-1: Evidence-driven cascade (the real agentic layer)**
- The repo measured that a 0.6B top-50 rescored by 1.7B is lossless: 0.93761 against 0.93756 (`docs/RETRIEVAL-PIPELINE.md`).
- It then concluded the cascade "saves nothing", because both document indexes must be built. **Under the FAQ that conclusion is wrong:** document indexing is offline and allowed, and only query-time cost counts.
- **Policy:** small tier → execute → escalate to a 1.7B query encode only when no confident verification exists.
- **Hypothesis:** most test queries resolve without the 1.7B encode (0.6B recall@20 is 0.979; gold verification is about 0.8). CPU query cost drops by a large fraction at non-inferior NDCG. Measure the escalation rate, the latency distribution, and NDCG.
- Also try F2LLM-v2-330M or 160M as the first tier.

**POST-2: Static compatibility gate.** Order and prune the execution list using PRE-1 × PRE-3:
- a stdin statement against a function-only snippet;
- a read-signature line count against the example-input structure.
- **Value:** fewer executions and higher precision. **Kill if** gold recall in the executed set drops (measured by replay).

**POST-3: Judge-aware comparator ladder** (replaces the blanket rule in `run_candidate`)
1. exact;
2. ε-numeric, with ε from the statement;
3. case-insensitive, only if the statement allows it;
4. **shape or multiset match, only when the any-answer flag is set:** same line count, same token types, and the same multiset where applicable;
5. call-style `literal_eval` equality.

- Each rung carries its own evidence strength, consumed by POST-6. It does not use one fixed bonus.
- **Validate the rungs against real CodeContests+ checkers:** measure precision and recall of our heuristic "any-answer" rung against the official checker verdicts.

**POST-4: Call-style harness**
- **Callee resolution order:** a name mentioned in the statement, then a `Solution` method, then a single top-level function; arity must match.
- Calls run inside the sandbox, arguments parsed with `literal_eval`.
- **Low-entropy guard:** at least 2 passing examples with at least 2 distinct expected values, because bool/0/1 outputs collide.
- **Value:** about 0 on the APPS test split; decisive on a train-like distribution (57.6% call-style).
- **Kill if** promotion precision is below 0.80.

**POST-5: Consumption and shape instrumentation (out-of-the-box signal)**
- Wrap `sys.stdin` in the runner to record whether the candidate consumed all example input, hit EOF early, or left input unread. Record the output shape too.
- A right-problem program almost always consumes the input exactly. That holds even on multi-answer problems where exact output fails.
- A wrong-problem program often under- or over-reads.
- **Hypothesis:** a strong graded signal for the 12.8% multi-answer class and for crash triage (parse crash vs logic crash). Validated by replay: distributions for gold against non-gold.

**POST-6: Evidence fusion (LEVER-style) replaces the binary "+bonus, never demote" rule**
- A small **monotone-constrained** logistic or GBDT scorer over:
  - dense score, rank, and gap to the top;
  - the comparator rung reached and the fraction of examples passed;
  - consumption/shape flags and the crash class;
  - static compatibility;
  - spec-card similarity, if PRE-4 ships;
  - hubness.
- Trained on TUNE (APPS-train, all families) plus external train (CodeContests+). Few features. Monotone constraints: never "passing more examples → lower score".
- It subsumes tie-breaks among multiple verified candidates, soft promotion (rank 2 for deep hits), and mild demotion on structural incompatibility.
- The 9-feature reranker failed because its features restated what the encoder knows. **Execution-outcome features do not.**
- **Kill if** it does not beat the current binary rule on DEV and on external data after correction.

**POST-7: Adaptive deepening.** When nothing in the top 20 reaches a strong rung, execute ranks 21–50 within dense gap 0.20. On test, 52 golds sit outside the top 20 and recall@100 is 0.996, so the upper bound is about +0.005. The value is higher when dense is weaker.

**POST-8: Bonus, lineage ranking across versions**
- Retrieve over unique hashes once, not once per snapshot. This replaces the uncalibrated cross-snapshot max in `app/indexing/versioned.py:294`.
- Group lineages with MinHash.
- Rank within a lineage by execution rung, then diff-token affinity to the query, then a recency prior.
- **This is the answer to ExecRetrieval's finding.** Benchmark: DEV golds plus single-edit mutants, and real incorrect near-miss solutions from CodeContests+. Report exec@1 for dense-only against the pipeline.

---

## Part E: Combinations, measured rather than guessed ("execute once, replay many")

**INFRA-1: Outcome store.** Extend `scripts/instrument_execution.py` so that, per (query, candidate, example), it stores:
- exit class and crash type;
- consumption flags;
- raw normalized stdout, or the call-style return repr;
- output shape.

Run it once over the union of top-50 pools from all encoders. After that, **every comparator rung, gate, fusion rule, deepening policy and cascade threshold is a replay taking seconds**, with no re-execution.

The repo already proved replay exactness for gates, reproducing the control to 0.00e+00.

**INFRA-2: Factorial design instead of ad-hoc combos**
- **Factors:**
  - encoder: F2LLM-1.7B, F2LLM-0.6B, and a non-APPS encoder such as EmbeddingGemma-300M or bge-small;
  - PRE-2 on/off;
  - PRE-4 on/off;
  - PRE-5 on/off;
  - POST-3 rung set;
  - POST-4 on/off;
  - POST-5 on/off;
  - fusion: binary vs POST-6;
  - POST-7 on/off.
- Use a fractional factorial (resolution IV) to estimate main effects and two-way interactions on DEV.
- Confirm the top configurations on EXT-dev.
- **Key interaction hypotheses** (to pre-register):
  - (a) PRE-4 helps more with the non-APPS encoder than with F2LLM;
  - (b) POST-6 gains mostly come via POST-5 features on any-answer statements;
  - (c) POST-1 escalation rate × POST-2 pruning determines CPU cost with no NDCG interaction.

**INFRA-3: External data**
- CodeContests(+) validation and test Python solutions (stdin, with official checkers).
- A call-style set: HumanEval+MBPP and/or LeetCode solutions.
- Deduplicated against APPS with MinHash on statements and exact-hash on code.
- Split into EXT-dev and a sealed EXT-lockbox.
- Check CodeContests+ licence terms before use.

---

## Part F: Prerequisites (from the earlier review, required for Parts D and E)

**Sandbox rework.**
- Remove `preexec_fn`, whose `setrlimit` crashes macOS and is unsafe in threaded parents (`app/pipeline/execution_rerank.py:221`).
- Replace it with a runner that applies rlimits in a try-block inside the child, with no limits on Windows.
- Catch `SubprocessError` (line 270).
- Cap stdout.
- Report distinct crash, timeout and mismatch classes.
- This runner is also where POST-4 and POST-5 live.

**Parallel execution.** A thread pool over subprocesses in `app/pipeline/encoder.py:166`, results written back in dense order. Ship the DEV-neutral dense gate at 0.20.

**Scored-path plumbing:**
- PRE-6 cache with `--corpus-cache`;
- `--query-device cpu`;
- the router (POST-1) inside `PrePostPipelineEncoder`, with reason strings in `execution_audit`.

**Hygiene:**
- restore `config/retrieval.yaml`, and `copy.deepcopy` at `tests/test_config_freeze.py:22`;
- clamp `top_n` at `encoder.py:183`;
- accumulate texts across MTEB's 50k-document chunks.

**Honesty in the README and deck:**
- disclose GPU/CPU use;
- a single operating-point table;
- document contamination;
- disclose that the test split chose depth 20 and numeric comparison;
- numeric tolerance does not survive multiple-comparison correction (p = 0.017 across about 7 arms), so do not claim it;
- redraw the architecture as measured → shipped or rejected;
- make the repo and notebook visible to judges.

---

## Part G: Verification, testing and validation (applies to every component)

**V1. Data tiers**

| Tier | Contents | Use |
|---|---|---|
| TUNE | APPS-train | Fitting only |
| DEV | APPS-train held-out, per family | Gates |
| EXT-dev | External sets | Replication gate |
| EXT-lockbox | Sealed external | Reported once after freeze |
| APPS test | MTEB test split | Final report only, once per frozen config, logged |

- **Encoder-robustness arm:** every gate is also run with a non-APPS encoder. Pre/post-processing must add value independent of contamination, which is the jury's own framing.
- **Recency control:** external problems newer than F2LLM-v2, where solutions exist. If none exist, say so.

**V2. Statistics**
- **Pre-registration:** commit `experiments/<id>/prereg.md` before running.
- **Tests:** paired bootstrap (at least 2,000 resamples) plus a permutation test, with Holm correction within each factor family.
- **Minimum important difference:** +0.003 NDCG.
- **Quality ship rule:** corrected CI excludes zero, and replicated on EXT-dev.
- **Cost ship rule:** non-inferiority margin −0.002.
- **Mechanism check** (via `scripts/audit_execution_rerank.py`): the gain must land in the targeted bucket.
- **Independence:** the implementer does not run the gate.

**V3. Parser gold set.** About 250 statements across formats, labelled by a non-author, with a 50-item double annotation. Report precision and recall per layout. Property-based fuzzing: never raises, no empty extractions. The test-split parse count stays at 3,676 or more.

**V4. Verifier validation:**
- Rates per rung, stratified by rank and family: gold-verification recall, false-promotion rate, promotion precision.
- **POST-3 rungs against CodeContests+ checkers.**
- **Negative controls:** shuffled query↔example pairs; at least 1,000 random programs.
- **Adversarial controls:** `print("YES")`, echo, and constant-return programs must not pass on problems with at least 2 distinct outputs.
- **Mutation suite:** kill rate of worked examples on gold mutants.
- **Sandbox safety suite:** loop, fork bomb, memory hog, stdout flood, file writes, network, partial exit, reading past EOF.

**V5. Engineering:**
- unit and property tests;
- ranking invariants (monotone evidence, no NaN, row length preserved);
- MTEB-wrapper tests (forced small corpus chunks, corpus smaller than the depth);
- replay exactness: INFRA-1 replay equals a live run;
- a golden top-10 file on 200 queries;
- a CI matrix of ubuntu, macOS and windows × Python 3.11 and 3.12, plus a nightly weights/dataset job;
- determinism: parallel equals sequential, and a per-OS verification-count table;
- pinned versions, revision SHAs, cache checksums, `cache=None`;
- clean-room reproduction by a teammate who did not write the change.

**V6. Efficiency:**
- recorded hardware, cold vs warm, three repetitions;
- per-stage p50/p95/p99 latency, peak RSS, executions per query, and the cascade escalation rate;
- offline GPU-hours, tokens and dollars (PRE-4, PRE-6);
- incremental re-index time against the changed fraction.

**V7. Compliance tests:**
- the query path under `CUDA_VISIBLE_DEVICES=""`;
- queries never touch the corpus cache;
- the result JSON carries the git SHA, a config hash and a cache checksum;
- the test-split access log is checked in CI.

**V8. Red team.** A non-author tries to make a wrong candidate verify: trivial outputs, odd formats, huge inputs, contradictory examples. Findings are fixed or disclosed.

**V9. Reporting.**
- Every arm is reported, including null and negative ones.
- Measured and estimated values are labelled.
- Claims are scoped: nothing "generalizes" without EXT-lockbox evidence.

---

## Explicitly not recommended

- **Query-time LLM, HyDE, Query2doc, or LLM/cross-encoder rerankers:** FAQ answer 2.
- **Larger encoders:** FAQ answers 2 and 6.
- **Fine-tuning the encoder on APPS:** that is model work, not pre/post-processing, and it deepens contamination.
- **Test-batch transduction** (Sinkhorn over test queries, one-to-one assignment): it exploits benchmark structure.
- **Equal-weight RRF with BM25:** BM25 scores about 0.01 on APPS, and the measured fusion was harmful.
- **Reviving the 9-feature reranker as it was.**
- `meta_information`, `partition`, or the qrels identity.

## Critical files

- `app/pipeline/query_analysis.py`: PRE-1
- `app/pipeline/code_analysis.py`: PRE-3
- `app/pipeline/execution_rerank.py`: runner, POST-3/4/5
- `app/pipeline/encoder.py`: POST-1/2/6/7 on the scored path
- `app/pipeline/reranker.py`, `app/pipeline/tuning.py`: reuse for the POST-6 fit, with monotone constraints
- `app/indexing/cache.py`: PRE-6
- `app/indexing/versioned.py`: POST-8
- `scripts/instrument_execution.py`, `scripts/compare_arms.py`, `scripts/audit_execution_rerank.py`, `scripts/dev_eval.py`: INFRA-1/2
- new `scripts/eval_beir.py` and `experiments/`
- `config/frozen.yaml`, `.github/workflows/ci.yml`
- `README.md`, `docs/RETRIEVAL-PIPELINE.md`, `docs/SUBMISSION-DECK.md`

## End-to-end verification

1. `pytest` is green on the OS matrix.
2. The INFRA-1 replay equals a live run.
3. Every shipped component has a prereg plus a result JSON meeting V2, including the non-APPS-encoder arm.
4. EXT-lockbox is reported once after freeze.
5. One final `scripts/evaluate_mteb.py` run, then `validate_submission.py`, the provenance test, and the clean-room reproduction.
6. The V6 latency and cost table is regenerated for the shipped config.
7. The headline numbers agree across the JSON, CSV, README and deck.

## Sources

- LEVER: https://arxiv.org/abs/2302.08468
- Stolee et al.: https://dl.acm.org/doi/10.1145/2581377
- CoSQA+: https://arxiv.org/html/2406.11589v5
- CodeContests+: https://aclanthology.org/2025.findings-emnlp.299.pdf
- HardTests: https://openreview.net/pdf?id=v3SzGCfAXN
- AutoCode: https://arxiv.org/html/2510.12803v1
- EnrichIndex: https://arxiv.org/abs/2504.03598
- Doc2Query++: https://arxiv.org/abs/2510.09557
- CodeBridge: https://arxiv.org/abs/2504.07740
- PseudoBridge: https://arxiv.org/abs/2509.20881
- ReCo: https://aclanthology.org/2024.acl-long.75/
- Adaptive Re-Ranking: https://arxiv.org/abs/2606.25249
- QPP-based dense/sparse selection: https://arxiv.org/pdf/2109.10739
- QB-Norm: https://arxiv.org/pdf/2112.12777
- DualDIS: https://arxiv.org/html/2508.02538
- Query-encoder distillation: https://arxiv.org/pdf/2306.11550
- KALE: https://arxiv.org/pdf/2304.01016
- ExecRetrieval: https://arxiv.org/abs/2609.01865
- CoRNStack: https://arxiv.org/abs/2412.01007
- C2LLM (APPS in training data): https://arxiv.org/html/2512.21332v1
- F2LLM-v2: https://huggingface.co/codefuse-ai/F2LLM-v2-1.7B
- CPython macOS `setrlimit`: https://github.com/python/cpython/issues/84698
- MTEB 2.21 search wrapper: https://github.com/embeddings-benchmark/mteb/blob/2.21.0/mteb/models/search_wrappers.py
