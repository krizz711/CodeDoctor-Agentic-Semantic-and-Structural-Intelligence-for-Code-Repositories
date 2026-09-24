# Experiments — CodeDoctor v2 pre/post-processing

Every result in this folder was produced by a command below, on data tiers that
keep tuning and reporting apart. Hypotheses and decision rules are written in
each experiment's `prereg.md` **before** its results; results are appended
afterwards, including null and negative ones.

## Data tiers

| tier | what | used for |
|---|---|---|
| TUNE | APPS train split, all families, seeded half (`make_development_split(match_family=None, seed=20260919)`) | fitting anything |
| DEV | the other half | ship / kill decisions |
| EXT-dev | CodeContests valid+test, 108 statements (`scripts/build_external_set.py`) | replication |
| EXT-lockbox | the other 109 CodeContests statements | reported once, after a freeze |
| APPS test | the MTEB AppsRetrieval test split | final report only, logged |

No script in this folder reads the APPS test split.

## Pipeline of commands

```bash
# 0. dense artifacts (offline; a GPU is allowed here by the FAQ)
python scripts/encode_apps.py --tier interactive --device cuda --splits corpus,qtrain \
    --attention-budget 16000000 --max-batch 16                           # F2LLM-v2-0.6B
python scripts/encode_apps.py --model-name codefuse-ai/F2LLM-v2-160M \
    --revision 4ffe22c31406fc321b65a05b15563d38275ee51f --tag f2llm160m --device cuda \
    --splits corpus,qtrain                                               # cascade first tier
python scripts/encode_baseline.py --model microsoft/unixcoder-base --tag unixcoder   # no-APPS control

# 1. execute once: record every candidate's behaviour (stdout, exit, consumption, calls)
python scripts/record_outcomes.py --model f2llm06b --split dev  --exec-depth 20
python scripts/record_outcomes.py --model f2llm06b --split tune --exec-depth 20

# 2. replay many: every comparator / promotion policy, per family, with bootstrap CIs
python scripts/replay_outcomes.py --store data/outcomes/f2llm06b_all_dev.jsonl.gz --check-legacy 150
python scripts/replay_outcomes.py --store data/outcomes/f2llm06b_all_dev.jsonl.gz \
    --report experiments/E1_verification_policies/result_dev.json

# 3. learned verifier (fit on TUNE, report on DEV)
python scripts/fit_evidence_fusion.py --tune data/outcomes/f2llm06b_all_tune.jsonl.gz \
    --dev data/outcomes/f2llm06b_all_dev.jsonl.gz

# 4. cascade (small tier first, escalate only when nothing verifies)
python scripts/simulate_cascade.py --small data/outcomes/f2llm160m_all_dev.jsonl.gz --small-params 0.16 \
    --large data/outcomes/f2llm06b_all_dev.jsonl.gz --large-params 0.6

# 5. hubness correction of the dense pass (beta fitted on TUNE)
python scripts/hubness_eval.py --model f2llm06b

# 6. external replication (CodeContests, deduplicated against APPS)
python scripts/build_external_set.py
python scripts/encode_jsonl.py artifacts/ext_cc/texts_queries.jsonl --mode query \
    --out artifacts/ext_cc/emb_f2llm06b_queries.npy --device cuda
python scripts/encode_jsonl.py artifacts/ext_cc/texts_corpus.jsonl --mode document \
    --out artifacts/ext_cc/emb_f2llm06b_corpus.npy --device cuda \
    --reuse artifacts/apps/texts_corpus.jsonl artifacts/apps/emb_f2llm06b_corpus.npy
python scripts/record_outcomes.py --external artifacts/ext_cc --model f2llm06b
python scripts/replay_outcomes.py --store data/outcomes/f2llm06b_external_ext_dev.jsonl.gz \
    --qrels artifacts/ext_cc/qrels.json

# 7. versions (the Bonus): correct code among single-edit buggy versions
python scripts/build_version_benchmark.py --queries 150
python scripts/record_outcomes.py --external artifacts/versions_apps --model f2llm06b
python scripts/eval_versions.py --store data/outcomes/f2llm06b_external_ext_dev.jsonl.gz

# 8. parser gold set (labelled by someone who did not write the parser)
python scripts/parser_gold_set.py sample
python scripts/parser_gold_set.py score experiments/V3_parser_gold/labelled.jsonl
```

## Decision rules (all experiments)

- Paired bootstrap over statements, 2,000 resamples, 95% percentile interval;
  Holm correction across the arms of one experiment.
- Quality change ships only if the corrected interval excludes zero, the effect
  is at least +0.003 NDCG@10, no family regresses significantly, and the
  direction replicates on EXT-dev.
- Cost change ships if non-inferior: interval lower bound above −0.002.
- The gain must come from the mechanism claimed (per-family and bucket counts).

## Known limitations of this round

- The shipped encoder is F2LLM-v2-1.7B; this machine (RTX 2050, 4 GB) cannot
  encode it, so pools here come from F2LLM-v2-0.6B. Effects on the shipped
  system may be smaller (a stronger first pass leaves less to fix); the
  1.7B arms must be re-run where the 1.7B artifacts exist.
- GPU half-precision embeddings carry ~1e-3 relative noise (bf16 vectors do
  not normalise to exactly 1.0); near-tied candidates can swap between runs.
