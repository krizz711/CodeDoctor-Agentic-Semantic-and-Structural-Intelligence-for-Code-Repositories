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

## Execution verification (shipped on; `--no-execution` disables it)

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

**Two different defaults, deliberately.** The `ExecutionConfig` dataclass
defaults to `enabled=False`, and
[`tests/test_execution_rerank.py`](../tests/test_execution_rerank.py) pins that:
a library that runs untrusted code the moment you construct it is a bad
library, so a caller who does not ask for this pass never gets it. The shipped
submission config, `config/frozen.yaml`, sets `execution_rerank: true`,
because a bare `python scripts/evaluate_mteb.py` has to reproduce the number
we submitted. `--no-execution` overrides it back off.

The off-by-default dataclass exists for two reasons:

1. It executes code that came from the dataset. Every run is a fresh
   subprocess in a throwaway directory with `-I -S`, a wall-clock timeout, a
   stripped environment, and an address-space cap where the platform supports
   one. It never runs in-process, and every failure mode — syntax error,
   timeout, crash, wrong output — returns `False` rather than raising.
2. It had to clear the same paired-bootstrap gate as everything else before it
   was allowed into the shipped configuration. It did, by +0.0583 at depth 20;
   the nine-feature reranker did not, and ships as the identity.

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

### Two cost knobs, measured: one works, one does not

Execution is the expensive pass — 10,686 s of sandbox time on the submitted
run. Two ways to spend less of it were implemented behind default-off flags
and measured on held-out DEV at depth 20, each paired against the same control
by `scripts/compare_arms.py`:

| arm | NDCG@10 | Δ vs control (95% CI) | p | candidates run | wall clock |
|---|---|---|---|---|---|
| control, depth 20 | 0.95486 | — | — | 15,660 | 984 s |
| + early exit | 0.95270 | −0.00216 [−0.00447, −0.00040] · **rejected** | 0.041 | 4,662 (29.8%) | 301 s |
| + dense gate 0.20 | **0.95603** | +0.00117 [−0.00129, +0.00336] · neutral | 0.309 | 6,123 (39.1%) | 355 s |
| + dense gate 0.10 | 0.93942 | −0.01544 [−0.02387, −0.00757] · **rejected** | 0.000 | 2,282 (14.6%) | 127 s |
| + both (0.20) | 0.95371 | −0.00115 [−0.00427, +0.00186] · neutral | 0.465 | 2,509 (16.0%) | 153 s |

**Early exit was our prediction, and it was wrong.** The argument for it is
sound as far as it goes: every verified candidate in a row receives the same
bonus, so verified candidates keep their dense order and the first one to
verify is the one that ends up ranked first. Stopping there cannot change the
winner — and it does not. R@1 is bit-identical to the control, 0.9273 either
way, which is exactly what the argument predicts.

What the argument missed is what happens *below* rank 1. When the gold itself
would have verified at a later position, early exit never reaches it, so the
gold stays in the unverified block instead of being lifted into the verified
one. R@10 falls from 0.9777 to 0.9754 and NDCG follows. The saving is real —
70% fewer executions — but it is not free, and the interval excludes zero, so
the same gate that rejected the feature reranker rejects this.

**The dense gate works, and it is a cost lever rather than a quality one.**
Refusing to execute candidates more than 0.20 cosine below the best candidate
in their row runs 39% of the candidate budget for a quality difference that
does not clear the gate in either direction. Its point estimate is the best of
any arm and its R@1 is the highest measured, 0.9308 against the control's
0.9273 — three queries where a distant lookalike no longer gets the chance to
verify — but +0.00117 with a CI spanning zero is not a result, and is not
claimed as one.

The threshold was sized before it was tested, by `scripts/measure_dense_gap.py`:
on DEV the gold is the top candidate for over half of queries, and the gap
from the top to the gold is within 0.20 for 836 of 837 in-pool golds. The
0.10 arm is in the table because that measurement predicted it would start
discarding golds — it keeps only 96.65% of them — and it did, by −0.0154. A
sizing measurement that predicts a failure and is then confirmed is worth more
than one that only confirms the choice already made.

**Nothing here changed the submitted configuration.** Both flags default off,
`config/frozen.yaml` is untouched, and 0.96558 remains the control. The gate
is the candidate for the interactive service path, where 2.8× less sandbox
time at indistinguishable quality is the trade that matters; the submission
has no latency budget to protect.

### A promotion gate against false verification: rejected, and the reason is structural

False verification looked like the biggest remaining lever. Execution promotes
152 wrong documents on DEV, and the test-split decomposition put "verified,
wrong document" at 111 of 236 losses. So: still execute every candidate, still
record every verification, but refuse to *promote* one scoring more than Δ
below the best candidate in its row. `ExecutionConfig.max_promotion_gap`,
default `None`.

Because a promotion gate never changes which candidates run, one instrumented
pass (`scripts/instrument_execution.py`) supports an exact replay of every
threshold. `scripts/analyze_promotion_gate.py` refuses to report a threshold
until replaying the ungated configuration reproduces the control's recorded
NDCG; it does, to 0.00e+00.

**The thresholds came from the data, and the data rejected the obvious ones.**

| group | n | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| verified & correct | 593 | 0.0000 | 0.0434 | 0.1783 | 0.2205 |
| verified & **wrong** | 152 | 0.1653 | 0.2953 | 0.3553 | 0.4408 |

There is real separation — correct verifications sit at the top of the pool,
wrong ones far below — but correct ones extend to 0.2205, so any Δ under about
0.22 cuts into them. Δ = {0.01, 0.02, 0.05} were discarded on this evidence
rather than run as guesses.

| arm | NDCG@10 | MRR@10 | R@1 | R@10 | Δ vs ungated (95% CI) | p |
|---|---|---|---|---|---|---|
| ungated | 0.95486 | 0.94730 | 0.9273 | 0.9777 | — | — |
| gate 0.05 | 0.92172 | 0.90925 | 0.8851 | 0.9613 | −0.03314 [−0.04395, −0.02286] | 0.0000 |
| gate 0.10 | 0.93942 | 0.93097 | 0.9132 | 0.9660 | −0.01544 [−0.02387, −0.00757] | 0.0000 |
| gate 0.15 | 0.94951 | 0.94262 | 0.9261 | 0.9707 | −0.00535 [−0.01164, −0.00009] | 0.0660 |
| gate 0.20 | 0.95603 | 0.94890 | 0.9308 | 0.9777 | +0.00117 [−0.00129, +0.00336] | 0.3090 |
| gate 0.25 | 0.95588 | 0.94867 | 0.9297 | 0.9777 | +0.00102 [+0.00000, +0.00278] | 0.2560 |
| gate 0.30 | 0.95501 | 0.94750 | 0.9273 | 0.9777 | +0.00015 [+0.00000, +0.00046] | 0.6450 |

**No interval excludes zero on the positive side, so no improvement is
claimed.** (At 0.15 the percentile CI and the p-value disagree marginally —
upper bound −0.00009 against p = 0.066. That is the boundary behaviour of two
different summaries of the same resamples, not a result; the arm is at best
neutral and more likely slightly negative.)

Promotion gate 0.20 reproduces the *execution* gate 0.20 arm exactly —
0.95603, R@1 0.9308, R@10 0.9777 — which is what theory demands, since a
candidate that is never run and a candidate that is run but never promoted
produce the same scores. Two independently written code paths and a replay
agreeing to five decimals is the cross-check that makes the rest believable.

#### Why it cannot work, which is the useful part

The gate does exactly what it was asked to. At Δ = 0.20 it removes 58 of 152
false promotions and costs a single correct one. The score barely moves. The
counters say why:

| | queries |
|---|---|
| false verifications | 152, across 98 queries |
| ...gold also verified and still wins on dense order — **harmless** | 75 |
| ...gold never verified, so any wrong one wins — *addressable* | 10 |
| ...gold verified but a **closer** wrong one wins — unreachable | 13 |

Promoted candidates keep their dense order, and the gold is the top candidate
for over half of these queries. So a false positive that also verifies usually
lands *below* the gold and changes nothing. Only 23 of 98 affected queries are
harmed at all.

Of those 23, thirteen are structurally beyond this gate's reach: the wrong
document beat the gold precisely because it sits *closer* to the top, and ten
of the thirteen sit at gap exactly 0.0000 — they are the top dense candidate.
A gate that removes far candidates cannot touch a near one. The populations
are close to disjoint, and the gate is aimed at the wrong end of the
distribution.

That leaves ten addressable queries out of 853, of which a Δ = 0.20 gate
catches roughly half — consistent with the observed R@1 move of 0.9273 →
0.9308, about three queries.

**And it points at the real lever.** Of 767 golds sitting in the executed
top-20, only 593 verified. **174 golds failed to reproduce their own worked
examples** — a population 7.6× larger than the 23 queries false promotion
actually costs. Execution's problem on this corpus is not that it promotes the
wrong document; it is that it fails to recognise the right one. Parsing more
example formats, relaxing output comparison, or raising the per-candidate
timeout all attack a target seven times the size of the one this experiment
was built for.

`config/frozen.yaml` is untouched and the flag defaults to `None`.

### The comparison was the problem: +0.00468, and the interval excludes zero

The promotion-gate result pointed at gold verification recall: of 767 golds in
the executed window, only 593 verified. `scripts/diagnose_verification_misses.py`
re-runs those 174 failures and records why each one failed.

| cause | n | share |
|---|---|---|
| output mismatch | 105 | 60.3% |
| crashed | 64 | 36.8% |
| passed one example, failed another | 5 | 2.9% |
| **timeout** | **0** | **0.0%** |

No timeouts at all, which removes a plausible fix before any work went into
it: the two-second budget is not what loses golds.

The 64 crashes are 26 `ValueError`, 19 `SyntaxError`, 9 `IndexError`, 5
`ModuleNotFoundError`, 4 `EOFError`, 1 `NameError`. Two of those deserve
naming. The `ModuleNotFoundError`s are the `-I -S` sandbox doing its job, at
the documented rate of about one verification in 120. The `SyntaxError`s are
*not* Python-2 code — that was tested, and 0 of 19 match Python-2 patterns.
Twelve are `'return' outside function` and seven are unbound `nonlocal`: the
corpus stored function **bodies** without their enclosing `def`. A dataset
extraction artifact, and nothing the pipeline can repair. (`ast.parse` accepts
all 19, because those errors are raised during symbol-table construction in
`compile()`; an initial check with the wrong tool said they were fine.)

Sampling the output mismatches showed the parser is not at fault either. Most
are multiple-valid-answer problems — permutations and colourings, where the
reference prints one correct answer and the gold prints another — which no
exact comparison can ever verify. But one class is a pure defect in the
comparison: a solution printing `1.0000000000000002` against a sample showing
`1.000000000`. Numerically identical, textually different, counted as wrong.

Offline sizing said 29 of the 105 mismatching golds were in that class.
`ExecutionConfig.numeric_tolerance` compares tokens as numbers when both parse
as numbers, falling back to equality otherwise, and requires the same token
count — so it can only ever turn a mismatch into a match.

| arm | NDCG@10 | MRR@10 | R@1 | R@10 | Δ vs control (95% CI) | p |
|---|---|---|---|---|---|---|
| control | 0.95486 | 0.94730 | 0.9273 | 0.9777 | — | — |
| **+ numeric tolerance 1e-6** | **0.95954** | **0.95312** | **0.9355** | **0.9789** | **+0.00468 [+0.00117, +0.00891]** | **0.017** |

The first arm in this series whose interval excludes zero, on 2000 resamples.
Seven more queries win at rank 1.

Because a relaxation admits false verifications as readily as it recovers real
ones, net R@1 cannot settle whether this is a genuine gain. A second
instrumented run measured the split directly:

| | gold | false | total |
|---|---|---|---|
| exact comparison | 593 | 152 | 745 |
| numeric tolerance | 622 | 154 | 776 |
| change | **+29** | **+2** | +31 |

Twenty-nine golds recovered against two new false verifications — and +29 is
exactly what the offline sizing predicted. Precision *rises*, 0.7960 to
0.8015, while gold verification recall goes 0.7731 to 0.8110. The relaxation
is lopsided in the right direction because a wrong program rarely produces
output that is numerically equal to the right answer; it produces output that
is simply different.

Promotion gates stacked on top add nothing: 0.20 gives +0.00073 (p = 0.461)
and 0.25 gives +0.00102 (p = 0.245), both intervals spanning zero. That is the
same structural reason as before, and it holds regardless of the comparison
rule.

**Six arms were rejected or neutral before this one.** Early exit, dense gates
at 0.10 and 0.20, promotion gates at 0.05, 0.10, 0.15, 0.20. None of the
clever gating worked. What worked was noticing the comparison was wrong.

`config/frozen.yaml` is untouched and `numeric_tolerance` defaults to `None`.
Adopting it on the submission means re-running the test split — a deliberate
decision, not a side effect of a DEV result. If anything the effect there
should be larger: the test split is 98.4% competitive-stdin with 97.6% example
coverage, against DEV's 91.8%.

---

## What it scores on the test split

DEV chooses the configuration; the test split reports it. Same pipeline, same
frozen weights, scored end-to-end by `mteb.evaluate` at version 2.21.0 over
all 3,765 test queries:

| test-split stage | NDCG@10 | MRR@10 | R@10 | hit@1 | sandbox |
|---|---|---|---|---|---|
| dense only, F2LLM-v2-1.7B | 0.93756 | 0.92370 | 0.97955 | — | — |
| + execution, top-10 | 0.96085 | 0.95449 | 0.97955 | 0.93519 | 5,839 s |
| + execution, top-20, exact comparison | 0.96543 | 0.95837 | 0.98619 | 0.93732 | 10,686 s |
| + numeric comparison (**shipped**) | **0.96558** | **0.95857** | **0.98619** | **0.93758** | 11,920 s |

**The last row is the shipped artifact, and it is not an improvement we can
claim.** Against the exact-comparison run, paired over the same 3,765 queries
with 2000 resamples: **+0.00015, 95% CI [−0.00046, +0.00081], p = 0.630** —
inside the noise band. Four queries improved, three worsened, 3,758 unchanged.
It is shipped because it is the higher measured value and because comparing
numbers as numbers is the more correct rule, not because the score moved.

**The DEV result did not transfer, and that is worth stating plainly.** On
held-out DEV the same change was +0.00468 with an interval excluding zero. On
test it is noise. The reason is headroom: DEV's dense pass leaves R@1 at
0.8277, test's at 0.8882, and after execution test is already at 0.93732. The
84 extra verifications the numeric rule buys on test are mostly golds that
were *already* winning their query, so they change nothing. A relaxation can
only help where something was being lost, and on the test split very little
was.

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
