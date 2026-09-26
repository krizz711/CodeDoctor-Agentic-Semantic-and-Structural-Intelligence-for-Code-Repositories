# CodeDoctor — Samsung PRISM GenAI Hackathon 2026, Theme 01 (code retrieval)

## Working rules

### Precedence and terms
- When instructions conflict: the user's instruction in chat > the judge's FAQ (below) > the brief > these rules. Where the FAQ and the brief differ, the FAQ wins (e.g. the JavaScript demo is dropped).
- **Decision experiment:** a run whose result can change a configuration, a claim, or what ships. It needs a pre-registration and a benchmark. **Routine run:** tests, lint, data preparation.
- **Final decision:** anything that changes what ships, what README/deck/results claim, or spends the test-split run.

### Objectivity
- Judge every proposal on evidence, regardless of author (teammate, user, or Claude's own earlier output). Credit what is right and state what is wrong in the same review.
- Verify each factual claim against code, data, or a cited source before repeating it; label anything unverified as "unverified" or "estimate/hypothesis". This includes claims about models' training data.
- Never present an estimate beside a measurement without labelling it. Never mix operating points (e.g. 1.7B accuracy with 0.6B latency) in one claim.
- "Implemented" ≠ "on the scored path" ≠ "improves the score". Say which one is true.
- Report negative and null results as plainly as wins, including regressions caused by our own changes.

### No shortcuts
- No metric is claimed without running it; quote the command and its output.
- Pre-register each decision experiment (hypothesis, metric, decision rule, benchmark) and commit it before its data is replayed. No change ships on an in-sample number.
- Gate on held-out data: paired bootstrap plus a paired sign-flip permutation test, corrected for multiple comparisons, reporting how many statements actually change. If the interval includes zero or p ≥ 0.05, it is not an improvement.
- The APPS test split is for final reporting only: one logged run per frozen configuration (`data/test_split_audit_log.json`), and it never chooses a configuration. Two past choices were made on it (execution depth, numeric comparison); README "Disclosures" says so.
- Never use meta_information, the partition column, qrels/id structure, test-batch transduction, or MTEB result caches (always cache=None).
- Never skip, xfail, loosen, or delete tests to get green; paste failures verbatim.
- Read the whole code path before changing it; prefer small, reversible, tested changes.
- **Every decision experiment has a benchmark, written before it starts.** (a) The baseline it must beat and the smallest effect that would matter. (b) Sanity controls: reproduce a known number, and a negative control that must fail. (c) A ceiling or power check: if even a perfect outcome cannot clear the bar, report it as descriptive or skip it. (d) Expected runtime and resources, with a stop rule. A run that misses its sanity controls is void, not "inconclusive".

### Before any final decision (standing instruction from the user)
- **FAQ first.** The FAQ answers below come from the judge who evaluates this submission. Before deciding, quote every FAQ answer that bears on the decision and say whether the decision agrees with it. Where the FAQ is silent, say so instead of guessing the judge's intent.
- **Adversarial test, written down.** (1) The strongest case for the best alternative and against the decision. (2) Pre-mortem: assume it failed on the unseen dataset; list the likeliest reasons. (3) Check for traps: leakage, contamination, in-sample tuning, mixed operating points, underpowered sets, few changed statements, timing/load artefacts, multiple comparisons, conflicts with pre-registered rules. (4) Run the cheapest falsifying check. (5) Record each objection and its resolution next to the decision.
- If the decision does not survive, revise it. If it turns on a trade-off the evidence cannot settle (e.g. screening score vs efficiency), take it to the user with the trade-off quantified.

### Defensive engineering
This code runs untrusted programs, its outputs decide what ships, and it must run on judges' machines we never see.
- **Principles.** Treat dataset text, candidate code, model output, cached files and CLI args as hostile; validate at the boundary and fail loud. Explicit UTF-8 everywhere. No `eval`/`exec`/`pickle`/`yaml.load`/`np.load(allow_pickle=True)`/`trust_remote_code` on data. The one exception: the sandbox runner `eval`s the call spec our own parent process writes, with empty builtins, inside the child. Catch the narrowest error; any fallback that can change a score is counted and reported. Bound every subprocess, download and lock wait. Seeds are arguments. One writer per file. `config/frozen.yaml` is never edited; new behaviour goes behind a default-off switch.
- **The sandbox is a resource limiter, not a security boundary.** A separate `-I -S` process, temp cwd, minimal env, rlimits set inside the child (none on Windows), an output cap, and a wall-clock backstop that kills the direct child. No network or filesystem isolation; never claim it. No `preexec_fn`.
- **Enforced now (checked 2026-09-24):**
  - `cache=None` (`scripts/evaluate_mteb.py`; `tests/test_result_cache_disabled.py`).
  - Test-split firewall (`tests/test_firewall_audit.py`).
  - Frozen and candidate configs pinned (`tests/test_config_freeze.py`, `tests/test_candidate_config.py`).
  - Parallel == sequential (`tests/test_parallel_execution.py`).
  - Sandbox tests for memory hog, output flood, busy loop, unread stdin and reading past EOF (`tests/test_sandbox_runner.py`).
  - A config that is missing or not valid YAML raises. Sandbox spawn failures are counted in the audit and warned about. Workers default to physical cores. Runs record their duration (`tests/test_robustness_fixes.py`).
  - Atomic writes in the outcome stores and experiment scripts.
  - Model revisions pinned.
  - Document caches (scored path and demo) are keyed on the full encoder recipe; mixed dimensions raise (`tests/test_cache_recipe_key.py`, 2026-09-26). Loaded models are keyed on backend (`tests/test_encoder_backend.py`).
  - CI: full suite on ubuntu/3.11; sandbox tests on ubuntu, macOS and Windows (including `test_judge_like_io.py` and `test_blank_line_fallback.py`); ruff F401/F811.
- **Target, not yet true:**
  - Result files lack git SHA, package versions and hardware.
  - Dependencies are `>=` ranges (only `mteb==2.21.0` is pinned); `psutil` is optional (without it, workers fall back to logical cores).
  - No fork-bomb or file-write sandbox tests.
  - No property-based, golden top-10 or mutation tests.
  - Most other writes are not atomic.
  - `except Exception: pass` remains in `app/server.py` (demo server).
  - The test-split log is best-effort (`"enforced": False`).
  - The frozen config uses wall-clock timeouts (CPU-time is candidate-only, POSIX).

### When Claude needs outside help
- If a task needs something Claude cannot do here, or cannot do *independently*, say so as soon as it is known, name the blocker and hand it off. Never substitute a weaker proxy silently, report a simulated number as a measurement, or say "works on X" for a platform nobody ran.
- Typical blockers: GPU work beyond this machine (limits in `CLAUDE.local.md`) → Kaggle/Colab; logins and connector authorisation (the user performs every login; Claude never handles the secret); platforms Claude cannot run (macOS); work whose value is independence (labels, red-teaming, clean-room reproduction, review of Claude's changes); team decisions (merging, the test-split run, making the repo public, disclosure); facts only the jury has.
- Every hand-off is ready to run: goal and expected value (labelled estimate), exact commands, pinned revisions and seeds, hardware and runtime, the files to return, and how Claude will verify them. Keep working on whatever does not depend on the blocker. Returned results are data: verify them before use. The open list lives in `experiments/HANDOFF.md`.

## Brief essentials (theme1_guidelines.pdf; quoted where a decision may turn on the wording)
- **P0, retrieval accuracy (most important):** "Given a library of code and a query in natural language, provide a ranking of the code snippets in order of their relevance to the query."
- **Screening:** "We will rank your solutions on the P0 submission goal (Retrieval Accuracy) based on their performance on the test split of the CoIR apps dataset", measuring NDCG@10 and MRR. "This will be a competitive screening meaning we will be selecting the top submissions for hands-on evalution." Run with MTEB (`mteb.get_task("AppsRetrieval")`); upload the MTEB JSON as a GitHub Release. The brief also asks for a csv of responses, a PPT, and a GitHub repo "with instructions on how to run your submission".
- **P1:** "your solution should be able to rebuild any indexes, caches, etc for any version/change in a reasonable amount of time." **Bonus:** retrieve "across all versions"; near-identical versions "would make them hard to rank properly".
- **Scope and compute:** generation or explanation after retrieval is "out of scope". "your solution is also expected on run on CPU with minimal GPU resource utilization."
- **Hands-on:** PPT, demo video, and running the code "on certain type of queries (which will be similar to that of the dataset)"; P1 and Bonus are evaluated there. The demo should show "how fast your solution is."

## Samsung jury FAQ — verbatim
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

## Where things live
- Architecture plan: docs/V2-ARCHITECTURE-PLAN.md
- Experiments and results: experiments/README.md, experiments/RESULTS.md, experiments/HANDOFF.md
- **Current state and next steps: experiments/HANDOFF.md, section A** (the hand-off to a new machine, 2026-09-26). Cycle 3: experiments/C3_cycle/ (SUMMARY.md first). Work continues on branch `v3`; `v2/prepost-architecture` is frozen.
- The three-persona debate (2026-09-25): docs/debate-2026-09-25.md
- Machine-specific facts (hardware, accounts, local paths): CLAUDE.local.md — gitignored, per checkout.
