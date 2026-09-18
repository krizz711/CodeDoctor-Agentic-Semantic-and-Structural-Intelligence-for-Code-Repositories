# Submission deck — slide-by-slide content

Source material for the PPT. Every number here is reproduced by a command in
[`README.md` §6](../README.md); nothing is quoted that a reviewer cannot re-run.

> Numbers marked `[TBD]` are filled from the final evaluation run before the
> deck is built. Do not ship the deck with a `[TBD]` in it.

---

## Slide 1 — Problem

Given a library of code and a natural-language query, rank the snippets by
relevance. Thousands of snippets, snippets of arbitrary length, CPU only,
faster than generation because retrieval is the first step of the pipeline.

Not in scope: generating an answer, explaining the results.

---

## Slide 2 — Architecture in one picture

```
  query ──► categorize ──┐                  ┌── categorize ◄── corpus
            (family,     │                  │   (AST facts,
             identifiers)│                  │    shape)
                         ▼                  ▼
                  frozen dense encoder (F2LLM-v2, CPU)
                              │  top-50 pool
                              ▼
                   9-feature linear reranker
                              │
                              ▼      (optional, off by default)
                   execution verification
                              │
                              ▼
                       ranked snippets
```

One frozen encoder. Two passes. Every addition measured against the pass below it.

---

## Slide 3 — The 2.8 points that were on the floor

Our own earlier build scored **0.87603** with a model whose published score is
**0.90446**. Same model. Same data. The gap was four config values:

| | earlier build | published recipe |
|---|---|---|
| query instruction | generic | `Retrieve the most relevant code snippet for the given query.` |
| document prefix | — | none (correct) |
| `add_eos_token` | not set | `true` |
| `max_seq_length` | 512 doc / 1024 query | **8192** |

**Why it matters:** F2LLM pools with **last-token** pooling. Truncation cuts the
EOS, so the pooled vector becomes an arbitrary interior token — a *wrong*
vector, not a degraded one. 4.58% of documents and 1.73% of queries were being
silently corrupted.

`tests/test_published_recipe.py` now pins all four against MTEB's own model
registry, so this cannot regress.

*Takeaway: on this benchmark, reproducing the recipe exactly is worth more than
most algorithmic ideas.*

---

## Slide 4 — Pre/post processing (brief axes 1–4)

**Query → implied answer shape**

| family | evidence | expects |
|---|---|---|
| competitive-stdin | `-----Input-----`, `$n$`, "the first line contains" | `input()` / `print` |
| leetcode-method | `Example 1:`, `Constraints:`, `nums` | `class Solution:` |
| kata-function | ``` fences, "write a function" | free-standing `def` |
| **unknown** | — | **no prior applied** |

`UNKNOWN` is a real outcome. A family is asserted only on decisive evidence;
otherwise the query contributes nothing downstream rather than a guess.

**Snippet → actual shape.** `ast.parse` per snippet, regex fallback for the
Python-2 and truncated fragments in the corpus — and the fallback *labels*
itself `CANDIDATE` rather than `VERIFIED`. Unknown is reported as unknown.

---

## Slide 5 — The second pass (brief axis 5)

Nine features, linear, weights printable:

`dense` · `dense_margin` · `family_compat` · `defined_name_hit` ·
`ident_idf_overlap` · `number_idf_overlap` · `string_idf_overlap` ·
`length_prior` · `hub_penalty`

Three deliberate constraints:

1. **Linear.** One relevant document per query and nine features — a linear
   score is as accurate as anything heavier, runs in microseconds, and the demo
   can print *why* a snippet moved.
2. **Weights capped at ±2.0.** Cosine gaps inside a top-50 pool are 0.01–0.1, so
   ±2.0 is ample. Unbounded, coordinate ascent ran one weight to **−6.31**, at
   which point the encoder stops mattering and the system is a hand-built
   classifier the dev split cannot vouch for.
3. **IDF is corpus-side only.** Adding a query changes nothing, so a weight
   measured on one split transfers to another.

Fitted by coordinate ascent **directly on NDCG@10** — the reported metric, not
a surrogate.

---

## Slide 6 — The measurement problem nobody else will mention

Query-family mix, from query text alone:

| split | competitive-stdin | kata | leetcode | unknown |
|---|---|---|---|---|
| train (5000) | 34.1% | **43.0%** | 14.6% | 8.2% |
| test (3765) | **98.4%** | 0.1% | 0.8% | 0.8% |

The corpus is ordered by source; the test split is a contiguous tail of it.

**Consequence 1:** a contiguous half-split of train is not exchangeable — the
two halves score 0.209 and 0.623. A "held-out" number taken that way is
reporting distribution shift, not generalization.

**Consequence 2:** a *uniform* sample of train is the wrong dev set too. We tune
on the 1,707 competitive-stdin train queries, split at random.

*Takeaway: everyone on this leaderboard is validating against a distribution
that isn't the one being scored.*

---

## Slide 7 — What we refused to use

The raw CoIR files carry more than the encoder is given:

- `meta_information` → the original problem URL **and the starter code**, which
  names the exact function signature of the answer.
- `partition` → marks each document train/test; would halve the candidate set.
- qrels are the identity map `q<N> → d<N>`.

MTEB hands the encoder `text` and nothing else, and the pipeline is built to
match. Each of these raises the score without improving retrieval.

*Takeaway: we know the shortcuts exist, and the code is structured so it cannot
take them.*

---

## Slide 8 — Execution verification (optional, off by default)

Feed a statement's sample input to a candidate; compare stdout.

| | train | test |
|---|---|---|
| statements with a parseable worked example | 32.0% | **97.6%** |

- gold solutions reproduce their own sample output: **71–82%**
- random non-gold candidates that passed: **0 / 100**
- cost: ~0.08 s per candidate

**It executes dataset-authored code** — fresh `python -I -S` subprocess,
throwaway cwd, stripped environment, 2s timeout, address-space cap where the
platform allows. Never in-process.

It is **shipped on**, and that is a decision we defend rather than hide: it is
the only pass that cleared the paired-bootstrap gate, and leaving it on is what
makes a bare reproduce command match our submitted number. `--no-execution`
turns it off and reports the dense-only score.

A verified candidate is *promoted*; a failed one is never demoted — a crash may
be an environment difference, so evidence promotes and absence proves nothing.

---

## Slide 9 — Results

**First, the recipe reproduces.** Test split, both tiers:

| tier | model | ours | published |
|---|---|---|---|
| interactive | F2LLM-v2-0.6B | 0.90427 / 0.88369 | 0.90446 / 0.88404 |
| accuracy | F2LLM-v2-1.7B | **0.93756 / 0.92370** | 0.93690 / — |

**Then, what each pass is worth.** Held-out DEV, 853 queries the weights never
saw, every delta from a 1000-resample paired bootstrap:

| stage | NDCG@10 | MRR@10 | R@1 | R@10 | Δ vs dense (95% CI) |
|---|---|---|---|---|---|
| dense only | 0.89659 | 0.87560 | 0.8277 | 0.9613 | — |
| + feature reranker | 0.90139 | 0.87997 | — | 0.9613 | +0.0048 [−0.0009, +0.0109] · **rejected** |
| + execution, top-10 | 0.94061 | 0.93382 | 0.9168 | 0.9613 | +0.0440 [+0.0342, +0.0532] |
| + execution, top-20 | **0.95486** | **0.94730** | **0.9273** | **0.9777** | **+0.0583 [+0.0457, +0.0702]** |

Final test-split submission: `[TBD]`

**Say the negative result out loud.** Our own nine-feature reranker did not
clear its own gate. Against a weak encoder it is worth +0.037; against
F2LLM-v2-1.7B it is worth +0.0048 with a CI spanning zero, so the tuner shipped
the identity config and recorded why. Dual-encoder fusion (+0.0038, p = 0.164)
and pseudo-relevance feedback (neutral to negative) were rejected the same way.

Recall@10 holds at 0.9613 through top-10 — execution reorders the head of the
list, it cannot retrieve what the dense pass missed. At top-20 it moves to
0.9777: execution reaches ranks 11–20 and pulls verified golds into the top
ten, which no top-10 reranking could do. Pool ceiling 99.30%.

### A live query, not a leaderboard row

`python scripts/demo_query.py --query "find the shortest path between two nodes in a weighted graph"`

```
pass 1  query family : unknown  (no decisive evidence -> no prior applied)

#1  d3306  cosine=0.5588  shape=function_library
    import heapq
    from collections import defaultdict
    def shortest(N, edge):
        ver, inf = defaultdict(list), 10**10
        ...
#2  d1770  cosine=0.5265  shape=function_library
    def shortestPath(gra, srs, des): ...
#3  d4626  cosine=0.5240  shape=function_library
    def processes(start, end, processes):
        '''Dijkstra's shortest path algorithm'''

LATENCY (CPU)  classify=1.2ms  encode=924.3ms  search=21.3ms   total=946.9ms
```

Three different shortest-path implementations, on a phrasing that appears
nowhere in the corpus. Note the query family came back `unknown` — free text
carries none of the competitive-programming markers, so no prior was applied
rather than a wrong one being guessed.

*Also show one dataset query where the third pass moves the answer: the
statement, the dense top-5, and the snippet execution promoted to #1.*

---

## Slide 10 — Speed

Measured end to end on a 6-core i5-11260H laptop with 7.7 GB RAM — a
deliberately unflattering machine, and the numbers are reported as measured:

| operation | cost | note |
|---|---|---|
| query classification | 1.2 ms | pure regex |
| query encode (0.6B, CPU) | 924 ms | dominates; this box is RAM-starved |
| dense search over 8,765 snippets | 21 ms | one numpy matmul |
| feature rerank of top 50 | 0 ms | identity config — it did not ship |
| **free-text query, total** | **947 ms** | no worked examples, so no execution |
| execution verification (top 10) | +361 ms | only fires on statements with examples |
| **dataset query, total** | **2,077 ms** | |
| corpus encode, 17,530 texts | 12 min (0.6B) / 24 min (1.7B) | one-time, on one T4 |

The index is built once; only the query is encoded at query time. That is why
a query answers in under a second against 8,765 snippets on a laptop — and it
is the same code path the service runs.

Two tiers, one recipe: 1.7B for the screening submission, 0.6B for interactive
use. A tier changes *which* model runs, never *how* it is prompted.

---

## Slide 11 — Versions and evolution (P1 + bonus)

- Embedding cache keyed on `H(encoder_rev, tokenizer_rev, template_rev,
  encoder_input_bytes)` — **path and line numbers excluded**, so renaming or
  moving a file triggers **zero** re-embeddings.
- Editing one function re-indexes in ~0.01 s.
- Commit-pinned queries restrict candidates to that commit's tree snapshot:
  **0 leaks across 50 randomized property trials**.
- 100.0% span fidelity — 489 chunks across 171 files round-trip exactly.

---

## Slide 12 — Engineering notes worth one line each

- **Attention memory is quadratic.** Documents run 225 tokens on average to
  289,002 at the tail. A linear token budget still OOMed at 12 GiB; batches are
  budgeted on `max_len² × batch`.
- **T4 has no native bfloat16.** Torch emulates it and the encode slows by
  roughly 10×. `select_dtype` picks fp16 below compute capability 8.0.
- **The recipe is verified at load, not assumed.** `tokenizer_kwargs` is
  deprecated in sentence-transformers 6; the day it stops being honoured, the
  EOS quietly vanishes and every embedding is wrong while the run still
  succeeds. The encoder probes the loaded tokenizer and refuses to encode.
- **The hub profile is content-keyed.** It is built against one corpus ordering
  and applied against MTEB's — index alignment would have silently shifted every
  hub score onto the wrong document.
- **A planted-noise negative control** in the test suite shows why the held-out
  gate exists: a pure-noise feature still buys in-sample NDCG.
