<p align="center">
  <img src="docs/assets/logo.png" alt="CodeDoctor" width="180" />
</p>

<h1 align="center">CodeDoctor</h1>

**Agentic semantic and structural intelligence for code repositories.**
Samsung PRISM GenAI Hackathon 2026 · Theme 01.

> Ask a repository a question in English. Get back ranked code with the
> evidence for why each result ranked where it did — which passes ran, which
> were skipped, and what each one contributed to the score.

| | |
|---|---|
| **CoIR AppsRetrieval, NDCG@10** | **0.96558** |
| Dense pass alone (frozen recipe) | 0.93756 |
| Published score for the same encoder | 0.93692 |
| Tests | 279 passed, 1 skipped |
| Cold index, 7,132-line codebase | 1.0 s, lexical, no model |
| Identifier lookup | 0.4 ms |

---

## 1. Two products, one pipeline

Theme 01 is judged twice, and the two halves want opposite things.

**The screening half** is a leaderboard number on CoIR AppsRetrieval — 8,765
Python programs, 3,765 queries, exactly one correct answer each. Queries are
competitive-programming problem statements. Accuracy is everything; latency is
irrelevant.

**The demo half** is somebody pointing the tool at a repository and typing a
question. Here latency is everything: an answer that takes fifteen seconds is
a worse product than one that takes two milliseconds and is slightly less
certain.

The same three passes serve both. What changes is **which of them run**, and
that decision is made per query by a router.

```
                        natural-language query
                                  │
                         ┌────────┴────────┐
                         │  query router   │  reads cheap signals, plans
                         └────────┬────────┘
                                  │
        ┌───────────────┬─────────┴────────┬──────────────────┐
        ▼               ▼                  ▼                  ▼
   exact symbol       BM25            dense vectors      structural
   inverted index   postings        F2LLM-v2 encoder    call/import graph
     ~0.1 ms         ~2 ms            seconds (CPU)        ~1 ms
        └───────────────┴─────────┬────────┴──────────────────┘
                                  ▼
                    reciprocal rank fusion (k=60)
                                  │
                                  ▼
                     execution verification
              run the candidate against the query's
              own worked examples, in a sandbox
                                  │
                                  ▼
                          ranked code + evidence
```

## 2. Quick start

```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --port 8077
```

Open <http://127.0.0.1:8077/>, pick **Open a folder…**, paste a path. Any
local directory with Python, JavaScript or TypeScript in it.

It is searchable in well under a second, because chunking and the lexical
index do not need the encoder. Embeddings attach behind it while you are
already typing queries.

## 3. The router is the interesting part

Running every pass on every query makes the cheap case cost as much as the
expensive one. A question naming `authenticate_user` is answered exactly by an
index lookup — and then, in a naive pipeline, waits on an embedding that
cannot change the answer.

So the router plans, looks at what the cheap passes produced, and escalates
only when that evidence is weak. Measured on the same corpus, same session:

| query | total | dense |
|---|---|---|
| `createOption` | **0.4 ms** | skipped — *rank 1 is the identifier the query named* |
| `how does it decide an option is required` | 26,126 ms | ran — no identifier to match on |

Every decision carries a reason string, and the UI renders them. A planner
whose choices you cannot inspect is an if-statement with better branding.

The gate that matters most is execution verification, because it needs worked
examples in the query text — and that turns out to be a property of the
**dataset**, not of the wording:

| corpus | queries carrying worked examples |
|---|---|
| CoIR AppsRetrieval | 3,676 / 3,765 — **97.6%** |
| CodeTransOceanDL | 0 / 180 — **0.0%** |
| CodeTransOceanContest | 0 / 221 — **0.0%** |

`parse_worked_examples()` answers that in microseconds, so the pass is cheap
to skip and expensive to run blind.

## 4. What each pass is actually worth

Measured on held-out DEV with a paired bootstrap, not asserted:

| stage | NDCG@10 | verdict |
|---|---|---|
| Dense only | 0.89659 | baseline |
| + execution verification, top-5 | 0.93207 | |
| + execution verification, top-10 | 0.94061 | |
| + execution verification, top-20 | 0.95486 | **+0.0583, 95% CI [+0.0457, +0.0702]** |
| + nine-feature reranker | +0.0048 | **CI spans zero — rejected** |

The reranker did not clear its own significance gate, so **it ships as the
identity configuration**. The code is still there; the weights are not applied.
That is a deliberate choice and `config/reranker.json` records it.

Depth 20 rather than 10 is also measured, not assumed: it recovers 25 golds
sitting at dense ranks 11–20 and costs 13 to false verification, netting +8
rank-1 hits. It is affordable precisely because the deeper half verifies at
0.80% against 8.41% for ranks 1–10.

## 5. Does it generalise, or is it tuned to the benchmark?

The fair question, and the repository answers it by running the **unchanged**
pipeline against tasks it was never tuned on:

```bash
python scripts/blind_eval.py --task CodeTransOceanDL --tier interactive
```

`scripts/blind_eval.py` imports `build_mteb_encoder`, `get_pipeline_config`
and `get_tier_model` from the submission path rather than reimplementing them.
The only thing that differs is the dataset handed to `mteb.evaluate`.

| task | NDCG@10 | notes |
|---|---|---|
| CodeTransOceanDL | 0.35442 | 47 min on CPU; execution pass could not fire (0/180) |

**This is not presented as a win.** On that task the reranker is the identity
and no query carries worked examples, so the pipeline reduces to its dense
pass — there is nothing of this project's contribution left in the number. The
honest claim is narrower: the frozen recipe transfers to a task shaped nothing
like APPS without collapsing or erroring, and where a pass cannot contribute it
is reported as skipped rather than quietly credited.

> **A trap worth knowing about.** `mteb.evaluate` defaults to
> `cache=ResultCache(~/.cache/mteb)` with `overwrite_strategy="only-missing"`.
> Our encoder advertises the upstream model meta, so on a machine that has
> pulled MTEB's public results repository the evaluator returns a *published
> leaderboard result* without running a forward pass — in ~11 seconds, wearing
> the base encoder's score. Both evaluation paths pass `cache=None`, and
> `tests/test_result_cache_disabled.py` fails if anyone removes it.
> **If an evaluation finishes in seconds, the number is not yours.**

## 6. Repository-side capabilities

**P1 — version-aware retrieval.** Each commit is a separate snapshot. A query
pinned to commit K sees that commit's chunks and nothing else; an unknown
commit raises rather than silently falling back to HEAD. Every result set is
checked for cross-commit leakage before it is returned.

**Bonus — retrieval across all versions.** Searching every snapshot naively
makes things worse: a function that survived ten commits returns ten
near-identical rows. Results are grouped by *canonical identity* — file path
plus symbol name, which is what survives a version change — and each group
collapses to its best-scoring variant, carrying the versions it appears in.
Measured on a real repository: **5 distinct snippets in the top 5, against 3
when ungrouped.**

**Structural retrieval.** tree-sitter parses Python, JavaScript, TypeScript and
TSX into structure-aware chunks with exact source spans, and extracts
call/import edges. "What calls `verify_candidate`" is a graph query, not a text
match.

## 7. Evaluation integrity

This is the part that is unusual, so it is worth being explicit.

- **Leakage firewall.** `DevelopmentHarness.get_test_split()` raises during
  development. Access needs an explicit authorisation reason and is recorded in
  `data/test_split_audit_log.json`.
- **No improvement claimed without a paired bootstrap CI excluding zero.** The
  reranker is the proof: it made the number go up and was rejected anyway.
- **Frozen configuration.** `config/frozen.yaml` pins the model *and its
  revision SHA*, the prefixes, `add_eos_token` and `max_seq_length`. Two tests
  assert the file against the published recipe.
- **Columns that would have leaked, refused.** `meta_information` is identical
  for 500/500 query–gold pairs and `partition` removes 57% of the candidate
  pool for free. Both are documented and unused.

## 8. Reproducing every number

```bash
# stage-by-stage ablation with paired bootstrap
python scripts/run_ablation.py

# the submission artifact itself
python scripts/evaluate_mteb.py --device cpu --output appsretrieval_results.json

# generalisation suite on tasks never tuned on
python scripts/make_blind_notebook.py     # -> Kaggle notebook, GPU

# the whole acceptance pipeline
python scripts/run_all.py
python -m pytest -q
```

## 9. Layout

```
app/
  agent/router.py         which passes a query needs, and why
  retrieval/hybrid.py     exact + BM25 + dense, RRF fusion
  retrieval/encode.py     the frozen F2LLM recipe
  pipeline/execution_rerank.py   sandboxed verification
  indexing/versioned.py   snapshot isolation, cross-version collapse
  parser/chunker.py       tree-sitter chunking with exact spans
  workbench.py            the service behind the demo
config/frozen.yaml        the submission recipe, pinned by revision SHA
scripts/blind_eval.py     same pipeline, unseen datasets
web/dist/index.html       the workbench UI, no build step, no CDN
```

## 10. Known limits

- **CPU encoding is slow.** 0.35 texts/s for the 0.6B tier on six threads. The
  router exists partly because of this; a GPU removes it (MTEB's published
  `evaluation_time` for the same encoder over the whole APPS corpus is 74.9 s).
- **The execution pass is APPS-shaped by construction.** See §3. It is the
  largest single contributor to the screening score and contributes nothing to
  a general repository question.
- **Running the test suite with the demo server up can fail** on an 8 GB
  machine — two 2.4 GB encoders do not fit. Stop the server first.
