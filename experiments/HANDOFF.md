# Hand-off: the 1.7B re-measurement, and what needs the team

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
3. **GitHub access — still open in this session.** The GitHub connector was
   authorised on claude.ai (account `lifemeansfun`), but this session still
   lists `plugin:engineering:github` as needing sign-in: open the session's
   MCP servers (`/mcp`, or Connectors) and sign in there. Pushing works through
   Git Credential Manager; reading CI and opening PRs/Releases need the
   connector or `gh`. The CI result for `v2/prepost-architecture` is still
   unseen.
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
