# CodeDoctor — Agentic Code Intelligence

> Samsung PRISM GenAI Hackathon 2026 · **Theme 01: Agentic Code Intelligence**
> Approach in depth: [`docs/RETRIEVAL-PIPELINE.md`](docs/RETRIEVAL-PIPELINE.md) ·
> System record: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
> Deck: [`docs/CodeDoctor-PRISM-Theme01.pptx`](docs/CodeDoctor-PRISM-Theme01.pptx)

Given a library of code and a natural-language query, rank the snippets by
relevance. CodeDoctor does it in three passes over a frozen CPU encoder, and
reports honestly what each pass bought — including the one that bought nothing.

**Result: NDCG@10 = 0.96558** on the CoIR AppsRetrieval test split (MTEB
2.21.0, 3,765 queries), against **0.93756** for the frozen encoder alone.
The scored artifact is [`appsretrieval_results.json`](appsretrieval_results.json).

---

## 1. Quick start

Python 3.11+.

```bash
pip install -r requirements.txt        # runtime
pip install -r requirements-eval.txt   # MTEB, torch, sentence-transformers
pytest -q -m "not model"               # 212 tests, no weights needed, ~20s
```

The two deselected tests load real encoder weights (a multi-GB download). Run
the full suite with plain `pytest -q` when you want them.

Four tests carry a `data` marker: they exercise the evaluation harness against
the real CoIR AppsRetrieval split. They run once `artifacts/apps` is populated
(step 1 of §6) or once `requirements-eval.txt` is installed so the split can be
downloaded; otherwise they skip and say which of the two to do. The harness
raises rather than substituting placeholder rows, so a green run never means
the metrics were computed on synthetic data.

### Reproduce the submission artifact

```bash
python scripts/evaluate_mteb.py --device cpu --predictions build/predictions
python scripts/validate_submission.py appsretrieval_results.json
```

This runs the real pipeline through `mteb.evaluate`, so `appsretrieval_results.json`
is MTEB's own output — no score is ever assembled by hand. It also writes
`submission_predictions.csv` and `submission_metrics.csv`, the readable ranked
responses behind that JSON.

> **This command executes Python that came from the dataset.** The third pass
> verifies candidate snippets by running them against the worked examples in
> the problem statement. Each run is a fresh `python -I -S` subprocess in a
> throwaway directory, with a stripped environment, a 2-second wall-clock
> timeout, and an address-space cap where the platform supports one. It never
> runs in-process and never runs anything you did not download from
> `CoIR-Retrieval/apps`.
>
> It is **on by default** so that this command reproduces the submitted number
> exactly. It is also the only pass that cleared the significance gate
> (+0.0440 NDCG@10 at top-10, +0.0583 at top-20, both p = 0.0000 on held-out
> DEV). To turn it off:
>
> ```bash
> python scripts/evaluate_mteb.py --no-execution   # reports the dense-only score
> ```

### See it answer a query

```bash
python scripts/encode_apps.py --tier interactive     # one-time index build
python scripts/demo_query.py --interactive
```

Each answer prints the query's detected family, the ranked snippets, *why*
each one moved (per-feature contributions), and a per-stage latency breakdown.

### Developer service and UI

```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000   # then open http://localhost:8000
docker build -t codedoctor . && docker run -p 8000:8000 codedoctor
```

---

## 2. What the solution does

```
                    query text                       code corpus
                         |                                |
              ┌──────────▼──────────┐        ┌────────────▼────────────┐
   PASS 1     │  categorize (1,2)   │        │   categorize (3,4)      │
              │  family · identifiers│        │  AST facts · shape      │
              │  literals · cleanup  │        │  VERIFIED vs CANDIDATE  │
              └──────────┬──────────┘        └────────────┬────────────┘
                         │                                │
                    ┌────▼────────────────────────────────▼────┐
                    │  frozen dense encoder (F2LLM-v2, CPU)    │
                    │  published recipe, pinned by test        │
                    └────────────────────┬─────────────────────┘
                                         │  candidate pool (top 50)
                    ┌────────────────────▼─────────────────────┐
   PASS 2 (5)       │  linear reranker over 9 features         │
                    │  gated on held-out DEV — did NOT clear   │
                    │  the bar, ships as identity (see §3b)    │
                    └────────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
   PASS 3 (5)       │  execution verification — the only pass  │
                    │  that cleared the bar. ON; --no-execution │
                    └────────────────────┬─────────────────────┘
                                         ▼
                                   ranked snippets
```

Numbers in parentheses are the improvement axes named in the Theme 01 brief.

| Module | Role |
|---|---|
| [`app/pipeline/query_analysis.py`](app/pipeline/query_analysis.py) | query family, named identifiers, literals, boilerplate stripping |
| [`app/pipeline/code_analysis.py`](app/pipeline/code_analysis.py) | AST facts per snippet, regex fallback labelled `CANDIDATE` |
| [`app/pipeline/features.py`](app/pipeline/features.py) | the nine rerank features, corpus-side IDF, hubness |
| [`app/pipeline/reranker.py`](app/pipeline/reranker.py) | the second pass; identity config = exact dense passthrough |
| [`app/pipeline/tuning.py`](app/pipeline/tuning.py) | coordinate ascent on NDCG@10, weights capped at ±2.0 |
| [`app/pipeline/splits.py`](app/pipeline/splits.py) | distribution-matched development splits |
| [`app/pipeline/execution_rerank.py`](app/pipeline/execution_rerank.py) | sandboxed execution verification — runs dataset code; see the warning in §1 |
| [`app/pipeline/encoder.py`](app/pipeline/encoder.py) | the MTEB-facing two-pass encoder |
| [`app/retrieval/encode.py`](app/retrieval/encode.py) | the frozen encoder, two tiers, quadratic batch planner |

---

## 3. Three things worth knowing about this task

**The published recipe is worth ~2.8 NDCG points.** An earlier build of this
repository scored 0.87603 with a model whose published score is 0.90446. The
gap was configuration: F2LLM pools with **last-token** pooling, so truncating a
sequence cuts the EOS and the pooled vector becomes an arbitrary interior
token — a wrong vector, not a degraded one.
[`tests/test_published_recipe.py`](tests/test_published_recipe.py) pins the
prefix, the EOS flag and the 8192-token cap against MTEB's own model registry,
and the encoder probes the *loaded* tokenizer and refuses to run if the EOS is
missing. With the recipe applied, both tiers reproduce their published scores:
0.90427 for the 0.6B (published 0.90446) and **0.93756** for the 1.7B
(published 0.93690).

**The train and test splits are different datasets.** From query text alone:

| split | competitive-stdin | kata | leetcode | unknown |
|---|---|---|---|---|
| train (5000) | 34.1% | **43.0%** | 14.6% | 8.2% |
| test (3765) | **98.4%** | 0.1% | 0.8% | 0.8% |

The corpus is ordered by source and the test split is a contiguous tail of it.
A contiguous half-split of train scores 0.209 on one half and 0.623 on the
other, so a "held-out" number taken that way reports distribution shift rather
than generalization. Development splits are seeded-random and filtered to the
family the scored split is actually made of.

**Attention memory is quadratic.** Document lengths run from 225 tokens on
average to 289,002 at the tail. Batching by count, or even by a linear token
budget, asks for tens of gigabytes on that tail — this corpus OOMed at 32 GiB
and then at 12 GiB before the budget was made quadratic in sequence length.

---

## 3b. What each pass is actually worth

Held-out DEV — 853 competitive-stdin train queries the weights never saw,
first pass = F2LLM-v2-1.7B, every delta from a 1000-resample paired bootstrap:

| stage | NDCG@10 | MRR@10 | R@1 | R@10 | Δ vs dense (95% CI) |
|---|---|---|---|---|---|
| dense only | 0.89659 | 0.87560 | 0.8277 | 0.9613 | — |
| + feature reranker | 0.90139 | 0.87997 | — | 0.9613 | +0.0048 [−0.0009, +0.0109] · **rejected, inside noise** |
| + execution, top-5 | 0.93207 | 0.92293 | 0.9050 | 0.9613 | +0.0355 [+0.0269, +0.0438] |
| + execution, top-10 | 0.94061 | 0.93382 | 0.9168 | 0.9613 | +0.0440 [+0.0342, +0.0532] |
| + execution, top-20 | **0.95486** | **0.94730** | **0.9273** | **0.9777** | **+0.0583 [+0.0457, +0.0702]** |

**Our own nine-feature reranker did not clear its own gate.** Against a weak
encoder it is worth +0.037; against F2LLM-v2-1.7B it is worth +0.0048 with a
confidence interval that includes zero, so `scripts/tune_reranker.py` shipped
the identity configuration and recorded why. A 1.7B code-trained encoder
already models identifier overlap and answer shape well enough that restating
them as hand-built features buys nothing.

Two more first-pass ideas were measured and rejected the same way — dual-encoder
fusion (+0.0038, p = 0.164, and double the inference cost) and pseudo-relevance
feedback (neutral at best, significantly negative at α ≥ 0.4). The full grid is
in `data/first_pass_variants.json`.

**One thing did clear the gate, and it was not a cleverer reranker.** Execution
counted a solution printing `1.0000000000000002` as wrong against a sample
showing `1.000000000`. Comparing numeric output as numbers rather than as text
is worth **+0.00468 NDCG@10 on DEV, 95% CI [+0.00117, +0.00891], p = 0.017** —
and an instrumented run confirms the mechanism: **29 golds recovered against 2
new false verifications**, so precision rises rather than falls. Six gating
arms were rejected or neutral before this one. `numeric_tolerance` ships
default-off; the submitted configuration is unchanged.

**And one prediction of ours that the gate caught.** Stopping execution at the
first verified candidate provably cannot change which document ranks first —
R@1 came back bit-identical at 0.9273 — so we expected it to be free. It is
not: when the *gold* would have verified later in the list, early exit never
reaches it and the gold is left below the documents that did verify. R@10 falls
0.9777 → 0.9754, −0.00216 overall with a CI excluding zero, rejected. A dense
gate at 0.20 fared better, running 39% of the candidate budget for a quality
difference inside the noise band. Both ship default-off; neither touched the
submitted configuration. Full table and method in
[`docs/RETRIEVAL-PIPELINE.md`](docs/RETRIEVAL-PIPELINE.md), arm data in
`data/arms/`.

Recall@10 holds at 0.9613 through top-10 — execution reorders the head of the
list, it cannot retrieve what the dense pass missed. At top-20 it *does* move,
to 0.9777, because execution then reaches candidates ranked 11–20 and promotes
verified ones into the top ten. The pool ceiling is 99.30% and nothing goes
past it. Cost is linear in depth: 0.42 / 0.83 / 1.78 s per firing query.

---

## 3c. What it scores on the split that counts

DEV picks the configuration; the test split reports it. Same pipeline, same
frozen weights, scored by `mteb.evaluate`:

| test-split stage | NDCG@10 | MRR@10 | R@10 | hit@1 | sandbox |
|---|---|---|---|---|---|
| dense only, F2LLM-v2-1.7B | 0.93756 | 0.92370 | 0.97955 | — | — |
| + execution, top-10 | 0.96085 | 0.95449 | 0.97955 | 0.93519 | 5,839 s |
| + execution, top-20, exact comparison | 0.96543 | 0.95837 | 0.98619 | 0.93732 | 10,686 s |
| + numeric comparison (**shipped**) | **0.96558** | **0.95857** | **0.98619** | **0.93758** | 11,920 s |

The final step is **not** claimed as a gain: paired over the same 3,765
queries it is +0.00015, 95% CI [−0.00046, +0.00081], p = 0.630 — four queries
better, three worse, 3,758 unchanged. It ships as the higher measured value
under a more correct comparison rule. On held-out DEV the same change was
+0.00468 with an interval excluding zero; it did not transfer, because after
execution the test split has almost no headroom left.

**Both depths were run, because DEV did not settle it.** DEV's dense pass
leaves 3.9% of golds outside the top 10; the test split's leaves 2.0%. That
halves the prize for searching deeper while leaving the risk — every extra
candidate is another chance to promote a wrong document that happens to pass
the same worked examples — unchanged. Predicting the winner from DEV would
have been guessing, so we measured both.

Depth 20 won by +0.00458. Against depth 10 it recovers 25 golds from dense
ranks 11-20 and loses 13 answers to false verification, for +8 rank-1 hits.
It is affordable for a reason worth stating: candidates at dense ranks 11-20
verify at **0.80%**, against **8.41%** for ranks 1-10. A document the dense
pass ranks 11th is rarely a program that passes the query's worked examples,
so the deeper half of the pool is roughly ten times cleaner than the shallow
half. The cost is wall clock — 10,686 s of sandbox time against 5,839 s.

Reproduce both decompositions from the shipped predictions:

```bash
python scripts/audit_execution_rerank.py --predictions submission_predictions.csv
```

---

## 4. What the solution deliberately does not use

The raw CoIR APPS files carry more than the retrieval system is given:

- `meta_information` holds the original problem URL and, for kata problems,
  the **starter code** — which names the exact function signature of the answer.
- a `partition` column marks each document `train` or `test`, which would halve
  the candidate set for free.
- the qrels are the identity map `q<N> -> d<N>`.

MTEB hands the encoder `text` and nothing else, and the pipeline is built to
match: it records the texts it was given during `encode()` and reads nothing
else in `similarity()`. Each of those three would raise the reported score
without improving retrieval.

---

## 5. Evaluation integrity

- **Frozen config.** The submission path reads `config/frozen.yaml`, never the
  mutable `config/retrieval.yaml`;
  [`tests/test_config_freeze.py`](tests/test_config_freeze.py) proves the
  insulation by mutating the live file and asserting the evaluation path is
  unaffected.
- **Leakage firewall, and what it actually covered.**
  `DevelopmentHarness.get_test_split` raises unless the firewall is lowered
  with an accepted reason, and enforces a two-access lifetime budget against
  `data/test_split_audit_log.json`. That is true, and until 2026-09-19 it was
  also misleading: the runs that produced every reported test-split number
  went through `scripts/evaluate_mteb.py`, which never called the harness and
  never logged. The control was real and did not cover the path that mattered.
  The log claimed one access; six had happened.
  `scripts/evaluate_mteb.py` now records every test-split evaluation, and
  `scripts/backfill_test_split_audit.py` reconstructed the missing history
  from the run artifacts — each reconstructed row is marked `BACKFILLED` with
  the file it came from, because an audit log that quietly acquires plausible
  entries is worth less than one that admits a gap. On this path the firewall
  **discloses rather than refuses**: a budget that silently blocked a
  submission run would be a worse failure than one that reports honestly.
- **Nothing ships inside the noise band.** Every component is gated on a
  1000-resample paired bootstrap on held-out DEV. If the 95% interval includes
  zero, `scripts/tune_reranker.py` ships the identity configuration and says so.
- **Weight cap.** No non-dense feature may exceed ±2.0, so the dense score
  always still matters. Unbounded, coordinate ascent ran one weight to −6.31.
- **Negative control.** A planted-noise feature buys in-sample NDCG in
  [`tests/test_pipeline_reranker.py`](tests/test_pipeline_reranker.py) — which
  is exactly why the held-out gate exists.

---

## 6. Reproducing every number

```bash
# 1. dump texts + embeddings once (GPU optional; CPU works, slowly — see below)
python scripts/encode_apps.py --tier accuracy --device auto --tag f2llm17b

# 2. fit the reranker on TRAIN, report on held-out DEV.
#    Expect it to REJECT the tuned weights and ship the identity: that is the
#    published result, not a failure of the run.
python scripts/tune_reranker.py --model f2llm17b --depth 50

# 3. stage-by-stage ablation with paired bootstrap
python scripts/dev_eval.py --model f2llm17b                               # dense only
python scripts/dev_eval.py --model f2llm17b --execution                   # + execution
python scripts/explore_first_pass.py --models f2llm17b f2llm06b           # fusion + PRF

# 4. the submission artifact
python scripts/evaluate_mteb.py --device cpu --predictions build/predictions
python scripts/validate_submission.py appsretrieval_results.json
```

**On hardware.** Step 1 is the only expensive part: 17,530 texts through a
1.7B encoder. On one T4 it is 24 minutes. On a 6-core laptop CPU it is roughly
four hours, because a single document encode costs ~1.6 s there. Everything
after step 1 runs on the cached vectors in seconds.
### Run it on a hosted GPU instead

Three generators package the pipeline for a hosted runner. All three embed the
source as a tarball built from the working tree by one shared packer
(`app/packaging.py`), so none of them can run code that has drifted from this
repository, and none of them is a hand-maintained copy:

| generator | target | purpose |
|---|---|---|
| `scripts/build_eval_kernel.py` | Kaggle script kernel | headless; this produced the submitted run |
| `scripts/make_kaggle_notebook.py` | Kaggle notebook | the reproduction walkthrough, written to be read |
| `scripts/make_notebook.py` | Colab notebook | same pipeline, Colab conventions |

```bash
python scripts/make_kaggle_notebook.py --out build/notebook --hubness artifacts/apps/hubness_f2llm17b.npz
```

```bash
kaggle kernels push -p build/notebook
```

The Kaggle notebook reproduces the submitted 0.96558 end to end — it prints the
frozen recipe off the live encoder object rather than quoting it, runs
`mteb.evaluate`, re-validates the emitted JSON, and finishes with the per-query
decomposition from `scripts/audit_execution_rerank.py`. Budget about 25 minutes
to encode and three hours in the execution sandbox at depth 20.

---

## 7. Repository-side capabilities (P1 and the bonus goal)

The same core backs a live developer service over a real repository:

- **AST chunking with 100% span fidelity** — re-reading raw source bytes at
  stored offsets round-trips exactly (489 chunks across 171 files in
  `test_repos/commander`).
- **Incremental re-indexing.** The embedding cache is keyed on
  `H(encoder_rev, tokenizer_rev, template_rev, encoder_input_bytes)` — path and
  line numbers are excluded, so renaming or moving a file triggers **zero**
  re-embeddings and editing one function re-indexes in ~0.01s.
- **Commit-scoped retrieval — this is P1, not the bonus.** A pinned-commit
  query restricts candidates to that commit's tree snapshot; 50 randomized
  property trials yield zero cross-commit leakage. An earlier version of this
  README called it the bonus goal, which was wrong: the brief defines the
  bonus as retrieval *across* versions, which is a different problem.
- **Cross-version retrieval — the actual bonus goal.**
  `VersionedIndex.retrieve_across_versions` searches every indexed version at
  once. The brief names the difficulty rather than leaving it implied: across
  versions the snippets "would still be very similar, which would make them
  hard to rank properly". Searching all snapshots naively makes that worse — a
  function that survived ten commits unchanged returns ten near-identical rows
  and crowds everything else out of the top-k.

  Results are therefore grouped by a canonical identity (file path + symbol
  name, which is what survives a version change), each group collapses to its
  best-scoring variant, and the hit carries the versions it spans plus a
  `variants` count of how many *distinct* contents share that identity. On
  `test_repos/commander` across two versions, a five-result budget returns
  **five distinct snippets collapsed against three ungrouped**. Pass
  `collapse_identical=False` to see the ungrouped ranking, which is what makes
  the difference measurable rather than asserted.
- **The dense pass reaches this path too.** `add_snapshot(..., embeddings=...)`
  turns it on per version and `uses_dense()` reports whether it is active —
  because a versioned index running lexical-only is a materially weaker system
  than the one §3 measures, and that should be visible rather than inferred
  from a score.
- **Labelled structural edges.** Direct AST matches are `VERIFIED`; dynamic
  dispatch and computed lookups are `CANDIDATE` with cited spans. Unknown is
  reported as unknown.

```bash
python scripts/verify_span_fidelity.py test_repos/commander
```
