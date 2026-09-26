# Hand-off

## A. Moving to another machine: the state on 2026-09-26 (branch `v3`)

Work moves to an agent on another laptop, which will have only GitHub. Everything that agent needs is in this repo; nothing depends on the old laptop.

### A1. Branches
| branch | state | rule |
|---|---|---|
| `main` | the screening submission (0.96558), untouched | never push to it (team decision) |
| `v2/prepost-architecture` | frozen at `9982be8`; CI green on all 8 jobs | **do not touch** (user, 2026-09-26) |
| `v3` | = `v2` + this hand-off (docs only) | continue here |

### A2. Audit of `v2/prepost-architecture` at `9982be8` (read-only, 2026-09-26)
- **CI:** green on all 8 jobs (the main job, the recipe registry, and the sandbox on ubuntu/macOS/Windows × Python 3.11/3.12). The last three pushes were all green.
- **Local:** `ruff check app scripts tests --select F401,F811,F821,E9` is clean. `pytest tests` gives 492 passed, 5 skipped (Windows; the skips are Linux-only rlimit tests).
- **`config/frozen.yaml` is byte-identical to `main`.** The submitted configuration was never edited. Every cycle-3 change sits behind switches that default to off, and they are on only in `config/candidate_v2.yaml`.
- **Secrets:** a scan of tracked files for GitHub, Hugging Face and Kaggle tokens and API-key patterns finds nothing. `CLAUDE.local.md`, `artifacts/`, `build/` and `.claude/` are gitignored.
- **Size:** no tracked file is over 5 MB. `data/outcomes/` holds 27 MB of recorded stores.
- **Stale docs (open; changing claims is a team decision):**
  - `README.md` does not yet mention cycle 3: E1, arm B, the demo-cache fix, or `rebuild=true`.
  - The disclosure line Dr. Ortega asked for, about test-qrels-informed sizing in `docs/V2-ARCHITECTURE-PLAN.md:270`, is still missing; the user deferred it.
  - `RESULTS.md` and `CLAUDE.md` are updated on `v3` in this hand-off.

### A3. Cycle 3: what is done, what is not
Full detail: `experiments/C3_cycle/` (`SUMMARY.md` first, then `PREREG.md` with addenda A1–A3).

| item | status | what is left |
|---|---|---|
| **P0:** why execution added +0.000 on CodeContests-onesol | **done**: a small-pool ceiling effect (`P0_ceiling_anomaly/result.md`) | – |
| **E1:** blank-line fallback | **done, adopted** in `candidate_v2`: DEV stdin +0.0078 [+0.0031, +0.0131] | – |
| **E1 arm B:** judge-like I/O | **done, accepted** (DEV 0 changes; CI green on 3 OSes); on in `candidate_v2` | – |
| **E2:** int8 OpenVINO 1.7B query encoder | **Run 1 VOID** (S2: Kaggle gave a slower CPU; addendum A3). **Run 2 (the one permitted repeat) finished on Kaggle; its output has NOT been read.** | step B1 below |
| **E3 Part 1:** recipe-aware demo cache key | **done** (tests) | – |
| **E3 Part 2:** in-process `rebuild=true` | **done** (tests) | – |
| **E3 Part 3:** P1 timing | **not run**: it needs ≥ 3 GB free RAM, and the old laptop had 0.2–0.6 GB | step B2 below |

### A4. Next steps, in order, with commands
**B1. Evaluate E2 run 2 (Kaggle kernel `munalsingh/codedoctor-e2-ov-a`, version 2).**
```bash
kaggle kernels output munalsingh/codedoctor-e2-ov-a -p build/kernel_e2a/out2
```
- The download includes a ~1.7 GB IR under `e2/ir_*`. The result is `e2/e2_A.json`.
- On Windows, `cd` into the target folder and pass `-p .`, because the CLI mangles relative paths.
- **Apply `PREREG.md` mechanically, in this order:**
  - **Is it void?** Void if S2 fails (`gate_T1.REF_within_25pct_of_7.83s`), S1 (`S1_C0_reproduces_stored`), S3, S6 (`pins`) or NEG (`NEG_fails_as_required`).
    - If void: **E2 stops.** Report the hardware blocker (A3); there are no more repeats.
  - **Otherwise, the gate:** `chosen` is `null` when neither T1 nor T1b passed.
    - Expected from the void run: fidelity fails for both (median cosine 0.982 and 0.991 against the 0.995 bar), and both are slower (1.43×, 1.31×).
    - If it fails: **E2 ends negative.** Write `E2_result.md` covering fidelity, speed ratio with CI, peak RSS, CPU model and ISA flags, and the single-machine end-to-end costs (`search_cost`, `execution_pass`, the REF/REF06 latencies, `doc_rate_1.7B_fp32`). These are descriptive and machine-labelled; they are also Dr. Ortega's requested single-machine cost.
  - **Only if a T passes (unexpected):**
    - commit the gate result;
    - `python scripts/build_e2_kernels.py kernel --stage b --arm T1` (or `T1b`), then `cd build/kernel_e2b && kaggle kernels push -p .`;
    - then `python scripts/exp_e2_replay.py --kernel-out <kernel-B out>/e2 --arm T1 --out experiments/C3_cycle/E2_result_dev.json`. Its known-answer control is already passed: Δ = 0 on stored vectors.
- **Never report an int8 CI above 0 as a gain** (PREREG E2).

**B2. E3 Part 3: the P1 timing run** (`scripts/exp_e3_p1_timing.py`). It runs last; nothing else heavy may run at the same time.
- **Machine change needs an addendum first.** The pre-registration says Part 3 runs on "this laptop" (i5-11260H) and "is not moved to another machine". Moving it is a legitimate team decision, but write addendum A4 in `PREREG.md` **before** running:
  - the new machine (CPU, cores, RAM);
  - that every P1 number will carry that machine label;
  - that the sanity band ("0.6B batch-1 within ±25% of 1.275 s/doc") refers to the old laptop and is re-anchored to the new machine's own measured batch-1 rate *before* the timing runs, or reported as a known mismatch.

  Decide which of these two before running, and say so in A4.
- **Pre-flight:** ≥ 3 GB free RAM.
- **Commands:** `python scripts/exp_e3_p1_timing.py --dry --out build/e3_dry.json` checks the mechanics (not a result). Then `python scripts/exp_e3_p1_timing.py --out experiments/C3_cycle/E3_result.json`.
- **Claim rule:** see PREREG E3. `R_claim.claim_allowed` decides whether the README may say "re-embeds only what changed".

**B3. Then:**
- update `C3_cycle/SUMMARY.md` and the `RESULTS.md` cycle-3 section with E2 and E3;
- push `v3` and check CI (`gh run watch <id> --exit-status`).

**B4. Next cycle** (not started):
- the Bonus near-miss tie-break (MBR-exec on mutated sample inputs; pilot 0.46 vs chance 0.33; `C3_cycle/research_scripts/pilot_nearmiss.py`);
- the Markdown statement normaliser and the disclosure line, both deferred by the user;
- the README update.

### A5. Setting up the new machine
1. **Code:** `git clone`, then `git checkout v3`. Python 3.11+ virtual env, `pip install -r requirements.txt` plus CPU `torch`. For the E2 kernels, the OpenVINO pins live in `scripts/build_e2_kernels.py`.
2. **`CLAUDE.local.md`** (gitignored; write a new one for that machine) should record:
   - hardware: RAM, cores, GPU;
   - GitHub account `krizz711`, via `gh auth login` by the user;
   - Kaggle account `munalsingh`: the user sets up CLI credentials; the agent never opens `~/.kaggle`;
   - "check CI after every push";
   - local artifact paths.
3. **Artifacts (gitignored).** Regenerate them; never commit them.
   - **`artifacts/apps/`:**
     - Texts, ids and qrels: `python scripts/encode_apps.py --tier interactive --device cpu`. This also writes the 0.6B vectors (slow on CPU; a GPU is allowed for this step by FAQ A).
     - 1.7B vectors: the output of Kaggle kernel `munalsingh/codedoctor-apps-encode`. Verify against the sha256 prefixes in the old manifest: corpus `2d2b8938b80a4699`, qtrain `9e19cdc2f32be6af`.
     - This script also writes **test-split** files (`texts_queries_test.jsonl`, `qrels_test.json`). **Never open them;** they are only for the one logged test-split run.
   - **`artifacts/ext_cc/`:** `python scripts/build_external_set.py --out artifacts/ext_cc`, then the 1.7B vectors from Kaggle kernel `munalsingh/codedoctor-ext-encode-17b` (`scripts/build_ext_encode_kernel.py`). The EXT-lockbox (`splits.json` key `ext_lockbox`) stays sealed.
   - **`artifacts/oracle/`:** `codeparrot/apps` train (`apps_train.jsonl`, sha256 `45e82ef2…`) and a CodeContests train conversion (see `scripts/parser_oracle.py`).
   - **Shortcut:** the private Kaggle dataset `munalsingh/codedoctor-c3-e2-inputs` holds the APPS corpus texts, the 1.7B/0.6B corpus vectors, the DEV query texts and the TUNE-100 subset (`experiments/C3_cycle/E2_dataset_manifest.json` lists them with sha256).
4. **Checks that the artifacts are right, before any experiment:**
   - `python scripts/replay_outcomes.py --store data/outcomes/f2llm17b_all_dev_deep50.jsonl.gz` must print dense 0.92792, `legacy+numeric1e-6` 0.94846 and `call-style+deep50` 0.95327. On the `..._tune_deep50` store: 0.93779 / 0.95375 / 0.95849. This needs `artifacts/apps/qrels_train.json`.
   - `python scripts/p0_headroom.py` (about 5 minutes, live execution included) recomputes dense ranks **from the stored vectors**. Its M0 lines must print 0.90155 (APPS DEV stdin) and 0.97857 (CC-onesol).

### A6. Traps met this cycle (so they are not rediscovered)
- **Kaggle:**
  - The CPU class is **heterogeneous**: an AMD EPYC 7B12 on one run, an Intel Xeon @ 2.20 GHz on the next, both AVX2 without VNNI. Record `cpu_model` and ISA with every timing.
  - Datasets are not mounted at `/kaggle/input/<slug>`; the E2 kernels search for them.
- **nncf 3.4 rejects group-wise int8** ("INT8 modes require per-channel quantization"). OpenVINO's CPU plugin also quantises **activations** dynamically (group 32), so weight-only fidelity numbers do not transfer.
- **F2LLM-v2's EOS is `<|im_end|>` (151645); `<|endoftext|>` is the pad token.** The tokenizer's post-processor appends EOS whatever the flag says.
- **macOS:** `fstat` on a pipe reports the bytes already buffered (not 0 as on Linux/Windows). That was a wrong assumption of ours, found by CI.
- **Wall-clock timeouts are load- and memory-sensitive.** Decision runs pre-flight ≥ 3 GB free RAM and run alone.
- **Windows consoles are cp1252:** set `PYTHONIOENCODING=utf-8` before printing Δ.
- **Scoring new dense orders:**
  - Use `scripts/union_replay.py` together with `app.pipeline.search.build_candidate_pool` on the **raw** stored vectors. Re-normalising them changes near-tie order: DEV 0.92777 instead of 0.92792.
  - Only fill pairs the policy uses (`scripts/exp_e2_replay.py`).

### A7. Context that used to live outside the repo
- **The three-persona debate (2026-09-25):** now in `docs/debate-2026-09-25.md`. The moderator's ranked next steps, and where each stands:
  1. Markdown normaliser: deferred by the user.
  2. Disclosure line: deferred.
  3. The CodeContests ceiling anomaly: **done** (P0).
  4. Single-machine encoder + execution cost: E2 kernel A's end-to-end block; valid only if run 2 is not void.
  5. CodeT/MBR agreement and PRE-4 enrichment: MBR is the next-cycle Bonus item; PRE-4 was rejected in research.
  6. Quantisation: E2.
- **The brief (`theme1_guidelines.pdf`)** is the organisers' document and is not committed. Its essentials and the judge's FAQ, verbatim, are in `CLAUDE.md`.

## B. Earlier hand-off (2026-09-24): the 1.7B re-measurement, and what needs the team

**Status 2026-09-24:** steps 1 and 2 are done. The 1.7B vectors came from the
team's Kaggle kernel and were verified; the re-measurement is in `RESULTS.md`,
section R1, and is stress-tested in `STRESS_TEST.md`. Step 3 is open. The
stress test estimates it would change the screening number by about 0 (F5),
and it waits on the encoder decision (R2, `R2_encoder_choice/`).

Round-1 results in `experiments/RESULTS.md` were measured with F2LLM-v2-0.6B
pools on the APPS train split. The steps below were written for the 1.7B
re-measurement and are kept as the reproduction recipe.

## 1. Offline artifacts (GPU allowed by the FAQ)

```bash
# document vectors, content-addressed: every later run (and the judges' re-run)
# encodes only queries, on CPU
python scripts/encode_apps.py --tier accuracy --device cuda --tag f2llm17b --splits corpus,qtrain
# hubness banks (PRE-5) -- same encoder as the run that uses them
python scripts/build_hubness_bank.py --texts artifacts/ext_cc/texts_queries.jsonl \
    --tier accuracy --out artifacts/hubness/bank_cc_f2llm17b.npy --device cuda
python scripts/build_hubness_bank.py --texts artifacts/apps/texts_queries_train.jsonl \
    --tier accuracy --out artifacts/hubness/bank_appstrain_f2llm17b.npy --device cuda
```

On a T4 the frozen `attention_budget` works; on smaller GPUs pass
`--attention-budget 16000000 --max-batch 16`. Builds without flash attention
(Windows wheels) get the GQA fix in `app/retrieval/encode.py` automatically.

## 2. Re-measure on DEV/TUNE with 1.7B pools (still no test split)

```bash
python scripts/record_outcomes.py --model f2llm17b --split dev  --workers 6
python scripts/record_outcomes.py --model f2llm17b --split tune --workers 6
python scripts/replay_outcomes.py --store data/outcomes/f2llm17b_all_dev.jsonl.gz --check-legacy 150
python scripts/replay_outcomes.py --store data/outcomes/f2llm17b_all_dev.jsonl.gz
python scripts/deepen_outcomes.py --store data/outcomes/f2llm17b_all_dev.jsonl.gz --to 50
python scripts/record_outcomes.py --model f2llm17b --split dev --workers 6 \
    --hubness-bank artifacts/hubness/bank_cc_f2llm17b.npy --hubness-beta 0.3 \
    --out data/outcomes/f2llm17b_all_dev_hubcc.jsonl.gz
```

Expect smaller effects than with 0.6B: a stronger first pass leaves fewer
statements for execution, deepening or hubness to fix. Decide with the same
rules (paired bootstrap, +0.003 minimum, TUNE replication). Use
`--workers` ≤ physical cores and prefer `timeout_mode: cpu` on Linux —
wall-clock timeouts change verdicts under load.

## 3. One logged test-split run with the chosen configuration

Edit `config/candidate_v2.yaml` to switch on only what passed at 1.7B
(add `hubness_bank: artifacts/hubness/bank_cc_f2llm17b.npy` only if hubness
passed), then:

```bash
python scripts/evaluate_mteb.py --config config/candidate_v2.yaml --device cpu \
    --corpus-cache build/corpus_cache_f2llm17b.sqlite --predictions build/predictions
python scripts/validate_submission.py appsretrieval_results.json
```

The first run with an empty cache encodes the corpus (use `--device cuda` for
that one run only if CPU is too slow; queries are then encoded on the same
device — say so in the README). That run is appended to
`data/test_split_audit_log.json`; it replaces the headline number everywhere
(README, deck, CSVs) and nothing is tuned after it.

Code changes that affect the test split regardless of configuration, all
disclosed in the README: the `Note`/`Example` heading parser fixes (changes one
test statement's extracted example, `q7726`) and the sandbox rewrite
(verdict-identical on Linux in a 1,844-run check).

## What the judges' machine needs

- Linux or macOS or Windows: the sandbox now runs on all three (CI matrix).
- The corpus cache from step 1 (ship it as a Release asset) so the re-run needs
  no GPU for documents.
- For the unseen dataset: use the APPS-train hubness bank (its answers are not
  in their corpus), not the CodeContests one.

## What needs the team (Claude cannot do these here)

Ranked by value to the project. Anything marked *estimate* has not been measured.
When an item closes, strike it here and note the result in `RESULTS.md`.

1. ~~**1.7B artifacts on a GPU box.**~~ **Closed 2026-09-24.** They already
   existed: the team's Kaggle kernel `munalsingh/codedoctor-apps-encode` had
   encoded corpus and train queries with the pinned revision. Verified here:
   texts identical, train NDCG@10 recomputed as 0.93285 (the manifest's value),
   sha256 in `artifacts/apps/manifest.json`. The test-query file was not
   imported. The external sets are being encoded by
   `scripts/build_ext_encode_kernel.py` (Kaggle), with a drift probe against
   the stored vectors.
2. ~~**Let Claude run the GPU jobs itself.**~~ **Closed.** The Kaggle CLI on
   this machine already had credentials (account `munalsingh`); Claude pushes
   kernels and fetches outputs without reading them. The token pasted in chat
   was not used and should be revoked.
3. ~~**GitHub access.**~~ **Closed 2026-09-25.** The GitHub CLI is installed and logged in as `krizz711` (device login, approved in the built-in browser). The first look at CI showed that this round's pushes `463d747`, `8c4f7da` and `32e0852` had **failed** the main job. `1107297` fixed that (pytest collected a `*_test.py` script; pandas imported at module level). All 8 jobs are now green, including the sandbox on macOS/Windows/Ubuntu × Python 3.11/3.12.
4. **Parser gold labels (V3) — replaced by external labels.** No teammate is
   available, and Claude's own labels would share the parser author's blind
   spots. `scripts/parser_oracle.py` scores the frozen parser against labels
   written by other people for other purposes: APPS train's official tests
   and CodeContests train's `public_tests` (prereg:
   `V3_parser_gold/prereg_oracle.md`). The human sheet stays available if a
   non-author turns up.
5. ~~**Questions only the jury can answer.**~~ **Closed: the FAQ is the final
   word.** What it implies: the unseen set is "representative of APPS" (FAQ 4),
   so statements with worked examples are likely but not guaranteed. The
   pipeline must therefore hold up with and without examples. It does not
   degrade when examples are missing (dense order is kept), and every claim is
   scoped to statements that have examples.
6. **Red team and clean-room reproduction (V8, V5) — open.** With no second
   person, the closest honest substitutes are: (a) a clean-room run on a fresh
   Kaggle machine from a fresh clone, which tests reproducibility of the
   environment, not reviewer independence; (b) an adversarial suite written
   from published attack classes rather than from the verifier's code. Both
   will be labelled as not independent.
7. **Team decisions.** Merging into `main`; spending the single test-split run
   (only after item 1); whether the verbatim FAQ in `CLAUDE.md` and
   `docs/V2-ARCHITECTURE-PLAN.md` stays when the repo goes public; making the
   repo and the Kaggle notebook visible to judges.
8. **Only if round 2 pursues it:** PRE-4 spec cards (a small code LLM writes a
   task description for each snippet, offline). It is the one planned component
   aimed at queries without examples. Its value is a hypothesis. Claude would
   write the script and a ready notebook; the run needs a GPU.

No model training is on this list: no measurement so far points to one, and
fine-tuning the encoder on APPS is excluded by the plan (it is model work and
deepens contamination).
