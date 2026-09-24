# Hand-off: what needs the 1.7B encoder (Kaggle/Colab GPU)

Everything in `experiments/RESULTS.md` was measured with **F2LLM-v2-0.6B**
pools on the APPS **train** split, because the laptop GPU (4 GB) cannot run
the 1.7B encoder. The shipped system uses 1.7B. Before anything here changes
the submission, three things must happen on a GPU box, in this order.

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
