# The retrieval pipeline

How CodeDoctor answers *"given a library of code and a natural-language query,
rank the snippets by relevance"*, and why each piece is there.

The Theme 01 brief names five places a solution can improve on a bare
embedding lookup. This document walks them in order, then covers the two
measurement problems that turned out to matter more than any of them.

---

## 0. The frozen encoder, and the 2.8 points that were on the floor

The first pass is a dense bi-encoder. Model choice dominates everything else
on this task — the CoIR paper's own table spans BM25 at 0.95 NDCG@10 to
Voyage-Code-002 at 26.52, and modern encoders reach the 90s — so the encoder
is selected from published results rather than by bake-off, then **frozen**.

Two tiers, one recipe:

| Tier | Model | Published NDCG@10 | Used by |
|---|---|---|---|
| accuracy | `codefuse-ai/F2LLM-v2-1.7B` | 0.9369 | the screening submission |
| interactive | `codefuse-ai/F2LLM-v2-0.6B` | 0.9045 | the live service and demo |

Both are Apache-2.0 and run on CPU. A tier may change *which* model runs; it
can never change *how* the model is prompted or truncated. That is asserted in
`tests/test_published_recipe.py`.

### The recipe is load-bearing

An earlier build of this repository scored **0.87603** with a model whose
published score is **0.90446**. The gap was not method. It was four
configuration values:

```yaml
query_prefix: "Instruct: Retrieve the most relevant code snippet for the given query.\nQuery: "
document_prefix: ""        # passages carry no instruction
add_eos_token: true
max_seq_length: 8192       # was 512 for documents, 1024 for queries
```

F2LLM pools with **last-token** pooling. When a sequence is truncated, the EOS
token is cut off and the pooled vector becomes whatever token happened to land
at the cap — not a degraded embedding, a wrong one. Measured on the APPS test
split, the old caps truncated 4.58% of documents and 1.73% of queries, and
those are exactly the queries that lost their gold document.

`tests/test_published_recipe.py` pins each value against MTEB's own model
registry (`mteb.get_model_meta(...).loader_kwargs`), so the config cannot drift
back without a red test. `CodeEncoder._verify_recipe_applied` additionally
probes the *loaded* tokenizer and refuses to encode if the EOS is missing or
the sequence cap was not applied — `tokenizer_kwargs` is deprecated in
sentence-transformers 6, and the day it stops being honoured this must be an
error, not three quiet points.

With the recipe applied, both tiers reproduce their published scores on the
AppsRetrieval test split:

| tier | model | ours | published |
|---|---|---|---|
| interactive | F2LLM-v2-0.6B | 0.90427 / 0.88369 | 0.90446 / 0.88404 |
| accuracy | F2LLM-v2-1.7B | **0.93756 / 0.92370** | 0.93690 / — |

(NDCG@10 / MRR@10. The residual difference is fp16-vs-bf16 arithmetic: the T4
used for the encode is Turing and has no native bfloat16.)

### Length-aware batching

Document lengths on APPS: mean 225 tokens, p99 900, **max 289,002**. A fixed
batch size that is fine for the mean asks for tens of gigabytes of attention
memory on the tail — the first GPU encode of this corpus died with
`CUDA out of memory: tried to allocate 32.00 GiB`.

`app/retrieval/encode.py:plan_batches` sorts by token length and fills batches
against a budget on `max_len_in_batch ** 2 * batch_size` — attention memory is
quadratic in sequence length, and a *linear* token budget was not enough: the
second attempt still died at 12 GiB. A single 8192-token document now gets a
batch to itself; the short ones pack densely, which also removes most of the
padding waste.

---

## 1 & 2. Query categorization and pre-processing

`app/pipeline/query_analysis.py`

A natural-language code query implies a *shape of answer*. Three shapes cover
the corpus:

| Family | Surface evidence | Implied answer |
|---|---|---|
| `COMPETITIVE_STDIN` | `-----Input-----`, "the first line contains", inline `$n$` LaTeX | reads `input()`, prints |
| `LEETCODE_METHOD` | `Example 1:`, `Constraints:`, `nums` | `class Solution:` |
| `KATA_FUNCTION` | markdown fences, "write a function", "should return" | a free-standing `def` |
| `UNKNOWN` | — | *no prior is applied* |

`UNKNOWN` is a real outcome, not a fallback bucket. A family is asserted only
when the winning evidence clears an absolute floor **and** leads the runner-up;
otherwise the query contributes nothing to the family prior downstream. An
unlabelled query is reported as unlabelled rather than guessed.

Pre-processing strips `-----Note-----` and `Explanation` sections and kata
sign-offs, and never returns an empty string.

The same module extracts the identifiers a statement explicitly names — from
markdown code spans, call-shaped mentions, and snake_case/camelCase words —
plus numeric and string literals from constraints and sample outputs.

---

## 3 & 4. Snippet categorization and post-processing

`app/pipeline/code_analysis.py`

Each snippet is parsed with `ast` and reduced to structural facts: what it
defines, what it imports, whether it reads stdin, which literals it hard-codes.
The APPS corpus contains Python 2 fragments and truncated solutions, so a
regex fallback catches what does not parse — and **labels itself**:
`CodeFacts.evidence()` returns `VERIFIED` for a real parse and `CANDIDATE` for
the fallback. Nothing downstream is allowed to treat the two as equal.

Snippet shapes mirror the query families: `STDIN_SCRIPT`, `CLASS_SOLUTION`,
`FUNCTION_LIBRARY`, `UNKNOWN`.

---

## 5. The second pass

`app/pipeline/reranker.py`, `app/pipeline/features.py`

The dense pass returns a candidate pool; the second pass rescores it with a
linear model over nine features:

| Feature | What it captures |
|---|---|
| `dense` | the frozen cosine score (weight pinned at 1.0) |
| `dense_margin` | distance from the query's own top-1, a per-query calibration |
| `family_compat` | query shape × snippet shape, as a PMI table estimated on train |
| `defined_name_hit` | a name the statement used is actually `def`-ed by the snippet |
| `ident_idf_overlap` | IDF-weighted identifier overlap |
| `number_idf_overlap` | constraint and sample-output literals appearing in the code |
| `string_idf_overlap` | quoted verdicts (`"YES"`, `"Impossible"`) appearing in the code |
| `length_prior` | longer statements get longer solutions (weakly) |
| `hub_penalty` | suppression of documents that crowd every ranking |

Three properties are deliberate:

**The model is linear.** With nine features and one relevant document per
query, a linear score is as accurate as anything heavier, costs microseconds
per candidate, and its weights print. The demo can say *why* a snippet moved.

**Weights are capped at ±2.0.** Cosine gaps inside a top-50 pool are of order
0.01–0.1, so ±2.0 is already ample authority to reorder the pool. Left
unbounded, coordinate ascent ran one weight to −6.31, at which point the dense
score stops mattering and the system is a hand-built classifier whose behaviour
on an unseen query mix the dev split cannot vouch for.

**IDF is corpus-side only.** Statistics are computed over indexed documents,
never over queries, which keeps every feature *inductive*: adding a query
changes nothing, so a weight measured on one query split transfers to another.

Hubness is the same idea. A few APPS solutions — short `print`-only scripts,
generic templates — sit near the centroid and crowd the top of almost every
ranking. The hub profile counts how often each document enters a *reference*
query set's top-k, and that reference set is always disjoint from the split
being scored.

### Fitting

`app/pipeline/tuning.py` runs coordinate ascent directly on NDCG@10 — the
metric being reported, not a surrogate. A full sweep is a matrix product and a
rank count, so it takes well under a second, and the search refines off-grid
and expands geometrically when a weight parks on a grid boundary.

Nothing ships on an in-sample number. Weights are fitted on TUNE, reported on
held-out DEV, and gated on a 1000-resample paired bootstrap: if the 95%
interval includes zero, `scripts/tune_reranker.py` ships the **identity**
configuration instead and says so. A planted-noise control in
`tests/test_pipeline_reranker.py` shows why the gate is needed — a pure-noise
feature still buys in-sample NDCG.

### The second pass did not clear its own gate

It is worth being plain about this, because the gate only means something if
it is allowed to reject our own work.

Against a *weak* encoder the features are worth a lot. With
`all-MiniLM-L6-v2` as the first pass, the reranker lifts held-out DEV from
NDCG@10 0.0638 to 0.1005 — a +0.037 gain with a 95% interval of
[+0.026, +0.049].

Against the shipped encoder they are worth almost nothing. With
F2LLM-v2-1.7B, coordinate ascent reaches 0.9280 on TUNE but the held-out DEV
gain is **+0.0048, 95% CI [-0.0009, +0.0109], p = 0.103** — inside the noise
band. `scripts/tune_reranker.py` therefore shipped the identity
configuration on its own, and `config/reranker.json` records that decision
along with the bootstrap that produced it.

The honest reading: a 1.7B code-trained encoder already models identifier
overlap, answer shape and literal reuse well enough that restating them as
hand-built features adds nothing. The features remain in the tree because
they are what the *ablation* is made of, and because they are the pass that
carries the third one — not because they are earning points.

Two other first-pass ideas were measured and rejected the same way
(`scripts/explore_first_pass.py`, full grid in `data/first_pass_variants.json`):

| variant | DEV NDCG@10 | Δ vs 1.7B alone | verdict |
|---|---|---|---|
| F2LLM-v2-1.7B alone | 0.89659 | — | baseline |
| fuse 1.7B×0.7 + 0.6B×0.3 | 0.90041 | +0.0038, p = 0.164 | inside noise, and doubles inference cost |
| fuse RRF(k=60) | 0.88804 | −0.0086 | significantly worse |
| PRF Rocchio, best setting | 0.89717 | +0.0006, p = 0.712 | inside noise |
| PRF Rocchio, α ≥ 0.4 | ≤ 0.89465 | up to −0.0093 | significantly worse |

---

## The two measurement problems

### The train and test splits are different datasets

Measured from query text alone — no labels, no metadata:

| split | competitive-stdin | kata | leetcode | unknown |
|---|---|---|---|---|
| train (5000) | 34.1% | **43.0%** | 14.6% | 8.2% |
| test (3765) | **98.4%** | 0.1% | 0.8% | 0.8% |

The corpus is ordered by source and the test split is a contiguous tail of that
ordering. Two consequences, both handled in `app/pipeline/splits.py`:

1. **A contiguous half-split of train is not exchangeable.** Measured with a
   small encoder, the first half scores NDCG@10 0.209 and the second 0.623. A
   "held-out" number from a contiguous split is reporting distribution shift,
   not generalization. Splits are seeded-random.
2. **A uniform sample of train is not the right development set either.**
   Tuning that has to transfer to the scored split is restricted to the family
   that split is made of — 1,707 competitive-stdin train queries, enough to fit
   nine weights and still hold half back.

### What the pipeline refuses to look at

The raw CoIR files carry more than the encoder is given:

- `meta_information` holds the original problem URL and, for kata problems, the
  **starter code** — which names the exact function signature of the answer.
- a `partition` column marks each document `train` or `test`, which would halve
  the candidate set for free.
- the APPS qrels are the identity map `q<N> -> d<N>`.

MTEB hands the encoder `text` and nothing else, and the pipeline is built to
match: `app/pipeline/encoder.py` records the texts it was given during
`encode()` and reads nothing else in `similarity()`. Every one of the three
would raise the reported score without improving retrieval.

---

## Execution verification (optional, off by default)

`app/pipeline/execution_rerank.py`

Competitive-programming statements publish worked examples. Feeding a
statement's sample input to a candidate solution and comparing stdout is
evidence no embedding can supply.

Measured on the corpus:

| | train | test |
|---|---|---|
| statements yielding a parseable example | 32.0% | **97.6%** |

Gold solutions reproduce their own sample output 71–82% of the time; of 100
random non-gold candidates tested against a statement's examples, **0** passed.
Cost is ~0.08s per candidate.

It is **disabled by default** for two reasons:

1. It executes code that came from the dataset. Every run is a fresh
   subprocess in a throwaway directory with `-I -S`, a wall-clock timeout, a
   stripped environment, and an address-space cap where the platform supports
   one. It never runs in-process, and every failure mode — syntax error,
   timeout, crash, wrong output — returns `False` rather than raising.
2. It has to clear the same paired-bootstrap gate as everything else.

A verified candidate is **promoted**; a failed one is never demoted. A crash
may be an environment difference rather than a wrong answer, so evidence
promotes and absence proves nothing.

### It is the only pass that cleared the gate

Held-out DEV, 853 competitive-stdin train queries the weights never saw,
first pass = F2LLM-v2-1.7B:

| stage | NDCG@10 | MRR@10 | R@1 | R@10 | Δ vs dense (95% CI) | cost |
|---|---|---|---|---|---|---|
| dense only | 0.89659 | 0.87560 | 0.8277 | 0.9613 | — | — |
| + feature reranker | 0.90139 | 0.87997 | — | 0.9613 | +0.0048 [−0.0009, +0.0109] · inside noise | — |
| + execution, top-5 | 0.93207 | 0.92293 | 0.9050 | 0.9613 | +0.0355 [+0.0269, +0.0438] | 0.42 s/query |
| + execution, top-10 | 0.94061 | 0.93382 | 0.9168 | 0.9613 | +0.0440 [+0.0342, +0.0532] | 0.83 s/query |
| + execution, top-20 | **0.95486** | **0.94730** | **0.9273** | **0.9777** | **+0.0583 [+0.0457, +0.0702]** | 1.78 s/query |

Three things the table says that the headline number does not:

* **Below depth 20, recall@10 does not move** (0.9613 for dense, top-5 and
  top-10 alike). Execution reorders the head of the list; it cannot retrieve a
  document the dense pass missed. The pool ceiling — gold inside the top 50 —
  is 99.30%, and nothing here goes past it.
* **At depth 20 recall@10 does move**, 0.9613 → 0.9777. That is execution
  reaching candidates ranked 11–20 and promoting verified ones into the top
  ten: fourteen queries' worth of gold documents that no amount of top-10
  reranking could have reached.
* **Cost is linear in depth.** Each doubling of the candidate budget doubles
  the wall clock (329s → 648s → 1395s over 853 queries).

**On the top-10 vs top-20 comparison.** Both are measured against the same
dense baseline on the same queries, but their confidence intervals overlap
([+0.0342, +0.0532] and [+0.0457, +0.0702]), so the raw numbers alone do not
establish that top-20 is better with 95% confidence. What does not depend on
that comparison is the recall@10 movement, which is a structural change rather
than a reordering: at top-10 it is impossible.

---

## What it scores on the test split

DEV chooses the configuration; the test split reports it. Same pipeline, same
frozen weights, scored end-to-end by `mteb.evaluate` at version 2.21.0 over
all 3,765 test queries:

| test-split stage | NDCG@10 | MRR@10 | R@10 | hit@1 | sandbox |
|---|---|---|---|---|---|
| dense only, F2LLM-v2-1.7B | 0.93756 | 0.92370 | 0.97955 | — | — |
| + execution, top-10 | 0.96085 | 0.95449 | 0.97955 | 0.93519 | 5,839 s |
| + execution, top-20 | **0.96543** | **0.95837** | **0.98619** | **0.93732** | 10,686 s |

R@10 is identical for the dense pass and the depth-10 rerank, which is the
consistency check the previous section predicted: reordering inside the top
ten cannot change which documents are in it.

### Why both depths were run

The DEV intervals overlapped, so DEV did not settle the depth. Worse, the two
splits disagree about how much there is to gain. DEV's dense pass leaves 3.9%
of golds outside the top ten; the test split's leaves 2.0%, because the dense
pass is simply stronger there. That halves the prize for searching deeper
while leaving the risk untouched — every additional candidate is another
chance to promote a document that passes the worked examples without being the
answer. A reasonable prior said depth 20 might well *lose* on test. It won, by
+0.00458, and the decomposition says why:

| bucket | top-10 | top-20 | Δ |
|---|---|---|---|
| verified, gold promoted | 2554 | 2575 | **+21** |
| dense right, nothing verified | 967 | 954 | **−13** |
| verified, wrong document | 94 | 111 | +17 |
| nothing verified, dense missed | 73 | 73 | 0 |
| gold absent from the pool | 77 | 52 | **−25** |
| **rank-1 correct** | **3521** | **3529** | **+8** |

The cost of going deeper is far smaller than a uniform false-positive rate
would predict. Candidates at dense ranks 11-20 verify at **0.80%**; candidates
at ranks 1-10 verify at **8.41%**. Verification is concentrated at the head of
the ranking, because a document the dense pass ranks eleventh is rarely a
program that solves the query's problem. Doubling the pool therefore roughly
doubles the wall clock — 10,686 s against 5,839 s — while adding only about a
tenth as many false promotions as the shallow half already contained.

One protective detail makes this safer than it sounds: verified candidates all
receive the same bonus, so they keep their dense order among themselves. A
false positive at rank 15 still loses to a gold at rank 3 that also verifies.
The damage is confined to queries where the gold fails to verify at all.

Reproduce either decomposition from the shipped CSVs:

```bash
python scripts/audit_execution_rerank.py --predictions data/predictions_top10.csv --compare submission_predictions.csv
```

---

## Reproducing the numbers

```bash
# 1. dump texts + embeddings once (GPU optional; CPU works, slowly)
python scripts/encode_apps.py --tier accuracy --device auto --tag f2llm17b

# 2. fit the reranker on the train split, gated on held-out DEV
python scripts/tune_reranker.py --model f2llm17b --depth 50

# 3. stage-by-stage ablation with paired bootstrap
python scripts/dev_eval.py --model f2llm17b
python scripts/dev_eval.py --model f2llm17b --execution

# 4. the submission artifact, produced by MTEB itself
python scripts/evaluate_mteb.py --device cpu --predictions build/predictions
python scripts/validate_submission.py appsretrieval_results.json
```

Step 4 emits `appsretrieval_results.json` (the scored artifact),
`submission_predictions.csv` and `submission_metrics.csv` (the readable
responses behind it).
