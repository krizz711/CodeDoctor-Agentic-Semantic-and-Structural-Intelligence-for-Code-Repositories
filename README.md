<p align="center">
  <img src="docs/assets/logo.png" alt="CodeDoctor" width="170" />
</p>

<h1 align="center">CodeDoctor</h1>

<p align="center">
  <b>Agentic semantic and structural intelligence for code repositories</b><br/>
  Samsung PRISM GenAI Hackathon 2026 · Theme 01
</p>

<p align="center">
  <img alt="NDCG@10 0.96558" src="https://img.shields.io/badge/CoIR%20AppsRetrieval-NDCG%4010%200.96558-1f6feb?style=flat-square" />
  <img alt="tests" src="https://img.shields.io/badge/tests-309%20passed%20%C2%B7%205%20skipped-2ea043?style=flat-square" />
  <img alt="cpu" src="https://img.shields.io/badge/runs%20on-CPU-4dd8ff?style=flat-square" />
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-555?style=flat-square" />
</p>

---

Ask a repository a question in plain English. Get back **ranked code** — with
the evidence for why each result ranked where it did, which retrieval passes
ran, which were deliberately skipped, and what each one contributed to the
score.

It is not a chatbot. The centre of the screen says *Relevant code · ranked by
relevance*, never *AI answer*, because a ranked span of real source with a file
and line number is checkable and a generated paragraph is not.

| | |
|---|---|
| **CoIR AppsRetrieval · NDCG@10** | **0.96558** |
| Dense pass alone, same frozen recipe | 0.93756 |
| Published score for the same encoder | 0.93692 |
| Test suite | **309 passed, 5 skipped** |
| Cold index of a 7,132-line codebase | **1.0 s** (lexical, no model) |
| Identifier lookup | **0.4 ms** |
| First question on an unindexed repo | **5.6 s**, including building the index |

---

## 1 · The problem has two halves that want opposite things

```mermaid
flowchart LR
  subgraph S["Screening half"]
    direction TB
    S1["CoIR AppsRetrieval<br/>8,765 programs · 3,765 queries"]
    S2["one correct answer per query"]
    S3["accuracy decides the screen<br/>judges re-run it; time and<br/>resources are scored too"]
    S1 --> S2 --> S3
  end
  subgraph D["Demo half"]
    direction TB
    D1["a real repository<br/>and somebody typing"]
    D2["no labels, no gold"]
    D3["latency is everything<br/>15 s is worse than 2 ms"]
    D1 --> D2 --> D3
  end
  S -.->|same three passes| D
```

A single pipeline serves both. What changes between them is **which passes
run**, and that decision is made per query by a router rather than fixed in
advance.

## 2 · Architecture

```mermaid
flowchart TD
  Q["natural-language query"] --> R{"query router<br/><i>reads cheap signals</i>"}

  R -->|always| EX["<b>exact symbol</b><br/>inverted index · ~0.1 ms"]
  R -->|always| BM["<b>BM25</b><br/>postings list · ~2 ms"]
  R -->|"call/import intent"| ST["<b>structural</b><br/>call graph · ~1 ms"]
  R -->|"no identifier to match"| DE["<b>dense</b><br/>F2LLM-v2 encoder · seconds on CPU"]

  EX --> F["reciprocal rank fusion<br/>k = 60"]
  BM --> F
  ST --> F
  DE --> F

  F --> E{"evidence strong?"}
  E -->|"rank 1 IS the named symbol"| OUT
  E -->|"ambiguous → escalate"| DE

  F --> V["<b>execution verification</b><br/>run the candidate against the<br/>query's own worked examples"]
  V --> OUT["ranked code + evidence"]

  style R fill:#0b1728,stroke:#4dd8ff,color:#edf4ff
  style V fill:#0b1728,stroke:#ffd65a,color:#edf4ff
  style OUT fill:#0b1728,stroke:#42d392,color:#edf4ff
```

Indexing is **tree-sitter** over Python, JavaScript, TypeScript and TSX,
producing structure-aware chunks with exact source spans plus a call/import
edge table.

## 3 · The router — the agentic layer

Running every pass on every query makes the cheap case cost as much as the
expensive one. A question naming `authenticate_user` is answered exactly by an
index lookup, and then, in a naive pipeline, waits on an embedding that cannot
change the answer.

```mermaid
sequenceDiagram
  participant U as User
  participant R as Router
  participant L as Lexical passes
  participant D as Dense encoder

  U->>R: "createOption"
  R->>R: plan — identifier present, defer dense
  R->>L: exact + bm25
  L-->>R: rank 1 = createOption
  R->>R: escalate? rank 1 IS the named symbol → no
  R-->>U: 0.4 ms

  U->>R: "how does it decide an option is required"
  R->>R: plan — 7 words, no identifier
  R->>L: exact + bm25
  R->>D: dense (planned up front)
  D-->>R: vectors
  R-->>U: 26,126 ms (26,097 ms encoding)
```

Both measured on the same corpus in the same session. Every decision carries a
reason string, and the UI renders them — a planner whose choices you cannot
inspect is an if-statement with better branding.

**The sharpest gate is execution verification**, because it needs worked
examples in the query text, and that is a property of the *dataset*, not the
wording:

| corpus | queries carrying worked examples | execution pass |
|---|---|---|
| CoIR AppsRetrieval | **3,676 / 3,765 — 97.6%** | fires |
| CodeTransOceanDL | **0 / 180 — 0.0%** | cannot fire |
| CodeTransOceanContest | **0 / 221 — 0.0%** | cannot fire |

`parse_worked_examples()` answers that in microseconds, so the pass is cheap to
skip and expensive to run blind.

## 4 · What each pass is actually worth

Held-out DEV, paired bootstrap, 2,000 resamples — measured, not asserted:

| stage | NDCG@10 | verdict |
|---|---|---|
| Dense only | 0.89659 | baseline |
| + execution verification, top-5 | 0.93207 | |
| + execution verification, top-10 | 0.94061 | |
| + execution verification, top-20 | 0.95486 | ✅ **+0.0583, 95% CI [+0.0457, +0.0702]** |
| + nine-feature reranker | +0.0048 | ❌ **CI spans zero — rejected** |
| + numeric output comparison | +0.00468 | ✅ CI [+0.00117, +0.00891], p = 0.017 |

**The reranker made the number go up and was rejected anyway.** It ships as the
identity configuration — the code is present, the weights are not applied, and
`config/reranker.json` records the decision. That is the single most important
thing in this repository: a gate that is only honest when it costs you
something.

Depth 20 over depth 10 is measured too: it recovers 25 golds sitting at dense
ranks 11–20, costs 13 to false verification, and nets **+8 rank-1 hits**. It is
affordable precisely because the deeper half verifies at 0.80% against 8.41%
for ranks 1–10 — a document the dense pass ranked 11th is far less likely to be
a program that passes the same examples.

## 5 · Proof of test results

```
$ python -m pytest -q
........................................................................ [ 45%]
............................................sssss....................... [ 68%]
........................................................................ [ 91%]
..........................                                               [100%]
309 passed, 5 skipped in 64.45s (0:01:04)
```

Verbatim in [`docs/evidence/test-run.txt`](docs/evidence/test-run.txt).
**30 test files, 314 tests.** The largest groups:

| tests | file | what it pins |
|---|---|---|
| 45 | `test_execution_rerank.py` | sandbox, timeouts, worked-example parsing |
| 34 | `test_pipeline_reranker.py` | the rejected reranker stays the identity |
| 33 | `test_workbench.py` | fusion arithmetic, version isolation, promotion |
| 21 | `test_router.py` | every routing decision, and the escalation loop |
| 21 | `test_pipeline_analysis.py` | query-family classification |
| 16 | `test_pipeline_encoder.py` | the frozen recipe is applied, not described |

Four of these exist purely to stop a number being reported that was never
computed: `test_result_cache_disabled`, `test_config_freeze`,
`test_published_recipe`, `test_firewall_audit`.

The five skips are the MTEB-registry checks, which need the optional `mteb`
dependency. They are **not** quietly ignored — CI runs them in their own job
and fails if they collect anything other than five tests.

> **How we found out 30 tests were not running.** `test_published_recipe.py`
> called `pytest.importorskip("mteb")` inside a class decorator. That argument
> is evaluated at *module import*, and `importorskip` raises, so the whole file
> was skipped and reported as a single innocuous `1 skipped`. Thirty tests —
> every frozen-recipe guard in the project — had not run in the default
> environment. Scoping the skip to the class surfaced them, and one had drifted:
> it still asserted `execution_candidates == 10` after the config moved to 20.
> A guard nobody runs is not a guard.

> ### The trap they guard against
>
> `mteb.evaluate` defaults to `cache=ResultCache(~/.cache/mteb)` with
> `overwrite_strategy="only-missing"`. Our encoder advertises the **upstream**
> model meta, so on any machine that has pulled MTEB's public results
> repository the evaluator matches a *published leaderboard result* and returns
> it **without running a single forward pass** — in about 11 seconds, in a
> well-formed result JSON, wearing the base encoder's score.
>
> We hit exactly this and nearly published it. Both evaluation paths now pass
> `cache=None`, and `tests/test_result_cache_disabled.py` fails if that is
> removed. **If an evaluation finishes in seconds, the number is not yours.**

## 6 · Does it generalise, or is it tuned to the benchmark?

The fair question. The repository answers it by running the **unchanged**
pipeline against tasks it was never tuned on — `scripts/blind_eval.py` imports
`build_mteb_encoder`, `get_pipeline_config` and `get_tier_model` from the
submission path rather than reimplementing them, so the only thing that differs
is the dataset handed to `mteb.evaluate`.

| task | NDCG@10 | wall clock | execution eligible |
|---|---|---|---|
| CodeTransOceanDL | 0.35442 | 47 min (CPU) | 0 / 180 |

**This is not presented as a win.** On that task the reranker is the identity
and no query carries worked examples, so the pipeline reduces to its dense pass
— there is nothing of this project's contribution left in the number. The
honest claim is narrower, and it is still worth making: the frozen recipe
transfers to a task shaped nothing like APPS without collapsing, and where a
pass cannot contribute it is **reported as skipped rather than quietly
credited**.

## 7 · Repository intelligence (P1 and the Bonus goal)

```mermaid
flowchart LR
  subgraph V["Version isolation · P1"]
    C1["commit A"] --> S1["snapshot A"]
    C2["commit B"] --> S2["snapshot B"]
    QQ["query pinned to A"] --> S1
    S1 --> RR["results"]
    S2 -. "never reachable" .-> RR
  end
```

**P1 — version-aware retrieval.** A query pinned to commit K sees that commit's
chunks and nothing else. An unknown commit **raises** rather than silently
falling back to HEAD, and every result set is checked for cross-commit leakage
before it is returned.

**Bonus — retrieval across all versions.** Searching every snapshot naively
makes things worse: a function that survived ten commits returns ten
near-identical rows and crowds everything else out. Results are grouped by
**canonical identity** — file path plus symbol name, which is what survives a
version change — and each group collapses to its best-scoring variant, carrying
the list of versions it appears in.

Measured on a real repository: **5 distinct snippets in the top 5, against 3
when ungrouped.** Evidence in
[`data/blind/tier3-commander.txt`](data/blind/tier3-commander.txt).

**Structural retrieval.** "What calls `verify_candidate`" is a graph query
against the extracted call/import edges, not a text match.

## 8 · Evaluation integrity

- **Leakage firewall.** `DevelopmentHarness.get_test_split()` raises during
  development. Access needs an explicit authorisation reason and lands in
  `data/test_split_audit_log.json`.
- **No improvement claimed without a paired bootstrap CI excluding zero.**
- **Frozen configuration.** `config/frozen.yaml` pins the model *and its
  revision SHA*, both prefixes, `add_eos_token` and `max_seq_length`. Two tests
  assert the file against the published recipe.
- **Columns that would have leaked, refused.** `meta_information` is identical
  for 500/500 query–gold pairs and `partition` removes 57% of the candidate
  pool for free. Both documented, both unused.

### Disclosures added after the Samsung FAQ

The jury's clarifications (GPU only offline, CPU at retrieval, time and
resources scored, contamination documented, no overfitting to APPS) exposed
claims in this repository that were wrong or incomplete. Stated plainly:

- **The scored run used a GPU for queries, not only for the corpus.** The
  0.96558 run encoded corpus *and* queries on a Kaggle T4. Documents can now be
  served from a content-addressed cache built once offline
  (`--corpus-cache`), so an evaluation encodes only queries, on CPU. The
  per-query CPU cost of the 1.7B tier has **not** been measured yet and is not
  claimed.
- **"Latency is irrelevant for screening" was wrong.** The judges re-run this
  evaluation and score time and resources. Parallel execution is wired
  (`execution_workers`, one worker per physical core). A dense gate at 0.20
  was wired too, then switched off: it kept the correct solution out of
  execution about 20× more often on CodeContests than on APPS
  ([`experiments/STRESS_TEST.md`](experiments/STRESS_TEST.md), F4).
- **Contamination.** F2LLM-v2's training table ([arXiv 2603.19223](https://arxiv.org/abs/2603.19223),
  Table 8) does not name APPS or CodeContests. It does include
  OpenCodeReasoning-2 (16,632 samples), whose problems come from TACO, APPS
  and CodeContests (train and valid splits), and xCodeEval NL2Code (Codeforces
  problems). Exposure to APPS and CodeContests problems is therefore plausible;
  per-problem exposure is unverified. An earlier version of this paragraph
  attributed APPS training to F2LLM via the
  [C2LLM report](https://arxiv.org/html/2512.21332v1), which lists APPS for
  C2LLM, a different model. The train/test parity check in
  `docs/RETRIEVAL-PIPELINE.md` argues against memorisation of the scored split,
  not against domain familiarity. Our CodeContests replication set is only
  partly protected: 47 of its 108 dev statements come from the valid split.
- **Two configuration choices were made on the test split.** Execution depth
  (10 vs 20) and the numeric comparison rule were both run on test before being
  adopted; the audit log records it. "DEV chooses, test reports" did not hold
  for those two.
- **Numeric comparison is not an established gain.** Its DEV p = 0.017 came
  from one of about seven arms, which does not survive a multiple-comparison
  correction, and on test it was +0.00015 with an interval spanning zero.
- **Brief axes 1–4 contribute nothing to the scored number.** Query and
  snippet categorisation and the nine-feature reranker were measured and
  rejected (identity weights); the scored gain over the encoder is execution
  verification alone.
- **Sandbox defects, fixed.** On macOS the `preexec_fn` resource limits raised
  an uncaught `SubprocessError` that aborted the evaluation; on Windows stdin
  was encoded with the locale codec, so any example with a non-ASCII character
  failed. On Linux the rewritten sandbox gives identical verdicts
  (0 differences in 1,844 train-split runs).
- **One parser fix touches one test statement.** A bare `Note` heading leaked
  into the last expected output of CodeContests-style statements; the fix was
  found on train statements (40 of 41 changed extractions) and also changes
  test statement `q7726`.

### v2 pre/post-processing (branch `v2/prepost-architecture`)

Measured on the APPS **train** split (DEV/TUNE halves, all families) and on a
CodeContests set that is separate from APPS but only partly protected from the
encoder's training data; the test split was not used. Re-measured with the
shipped 1.7B encoder, then stress-tested. Full numbers, nulls included:
[`experiments/RESULTS.md`](experiments/RESULTS.md),
[`experiments/STRESS_TEST.md`](experiments/STRESS_TEST.md).

| component | status | evidence |
|---|---|---|
| Call-style verification (LeetCode `Solution` methods, Codewars functions, doctests) | **ships in `config/candidate_v2.yaml`** | 1.7B: DEV +0.0041, TUNE +0.0032 NDCG@10 (0.6B: +0.0071, +0.0053); only LeetCode/Codewars move, so ~0 expected on the ~99%-stdin test split |
| Statement parser v2: judging rules, call examples, `Note` / `Example` heading fixes | parser fixes in code; rules off | the `Example` fix recovered 40/217 CodeContests statements. Against external labels the parser's example precision is 0.913 (CodeContests) and ≥ 0.938 (APPS), **below the pre-registered 0.95 bar**; most of the gap is the labels (multi-answer problems store a different valid output), and 0.97–0.98 once those are counted (post hoc). Interactive statements are skipped and brace arrays are read as lists |
| Sandbox runner (no `preexec_fn`, UTF-8 bytes, output cap, crash classes, CPU-time mode) | in code | 0 verdict changes on Linux; fixes macOS crash and Windows non-ASCII failures |
| Adaptive deepening (ranks 21–50 when nothing verifies) | in `config/candidate_v2.yaml`; final say with the encoder choice (R2) | 0.6B +0.0047, UniXcoder +0.0120; neutral at 1.7B (not a claimed APPS gain) |
| Dense gate 0.20 (skip candidates far below the best score) | **off** | excluded the correct solution ~20× more often on CodeContests than APPS (Fisher p ≤ 6e-4) |
| Parallel execution, content-addressed corpus cache (`--corpus-cache`) | in code | parity tests; judges' re-run needs no GPU for documents |
| Cross-version lineage ranking (Bonus) | in code (`app/indexing/lineage.py`) | correct version first 0.167 → 0.953 on single-edit mutants (upper bound) |
| Hubness correction of the dense pass | wired, **off** | dense +0.005 (clean bank); with execution DEV +0.0052 but TUNE +0.0020, interval spans zero |
| Judge-aware comparator, learned fusion, cascade, metamorphic tests | **rejected** (below the +0.003 bar or worse) | see RESULTS |

## 9 · Quick start

```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --port 8077
```

Open <http://127.0.0.1:8077/>, choose **Open a folder…**, paste any local path.
Then just type a question — if the target is not indexed yet it builds itself
and answers. Measured from a cold server: **5.6 s to ten ranked hits**.

## 10 · Reproducing every number

```bash
python -m pytest -q                       # 309 passed, 5 skipped
python scripts/run_all.py                 # acceptance pipeline
python scripts/run_ablation.py            # per-pass value, paired bootstrap
python scripts/judge_endtoend.py --repo test_repos/commander
python scripts/evaluate_mteb.py --device cpu --output appsretrieval_results.json
python scripts/blind_eval.py --task CosQA --tier interactive
python scripts/make_blind_notebook.py     # GPU notebook for the blind suite
```

CI runs the tests, the lint, the C++ core, the evaluation harness, artifact
validation and an end-to-end retrieval on an unseen repository on every push.
The weight-loading and dataset-dependent tests are deselected there on purpose
— a job whose result depends on a third-party download is a coin toss, not a
check — and they run locally and in the Kaggle notebook instead.

## 11 · Layout

```
app/
  agent/router.py              which passes a query needs, and why
  retrieval/hybrid.py          exact + BM25 + dense, RRF fusion
  retrieval/encode.py          the frozen F2LLM recipe
  pipeline/execution_rerank.py sandboxed verification
  indexing/versioned.py        snapshot isolation, cross-version collapse
  parser/chunker.py            tree-sitter chunking with exact spans
  workbench.py                 the service behind the demo
config/frozen.yaml             the submission recipe, pinned by revision SHA
scripts/blind_eval.py          same pipeline, unseen datasets
web/dist/index.html            the workbench UI — no build step, no CDN
docs/evidence/                 raw output behind the numbers above
```

## 12 · Known limits

- **CPU encoding is slow** — 0.35 texts/s for the 0.6B tier on six threads. The
  router exists partly because of this. A GPU removes it: MTEB's own published
  `evaluation_time` for this encoder over the entire APPS corpus is 74.9 s.
- **The execution pass is APPS-shaped by construction.** It is the largest
  single contributor to the screening score and contributes nothing to a
  general repository question. §3 gives the number rather than the excuse.
- **Running the full suite with the demo server up can fail** on an 8 GB
  machine — two 2.4 GB encoders do not fit. Stop the server first.
- **The 1.7B accuracy tier needs a GPU or ~8 GB free.** The interactive tier is
  the 0.6B model and is what the live service uses.
