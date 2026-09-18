# CodeDoctor — Agent Work Order

**Target system:** Agentic Code Intelligence (Samsung PRISM GenAI Hackathon 2026, Theme 01)
**Authority:** `THEME1-BUILD-PLAN-v3.pdf` (in repo root) is the governing plan. This document is the implementation order derived from it.
**Repo under change:** `CodeDoctor-Agentic-Semantic-and-Structural-Intelligence-for-Code-Repositories`

---

## 0. How to read this document

This is not a greenfield brief. The repository already exists, has 65 files, a working C++ CLI, a passing test suite, and a CI pipeline. **Preserve what works. Do not rewrite for taste.**

Section 1 is a verified inventory of what is actually in the repo — it was produced by reading the source, not the README. **Treat Section 1 as ground truth and do not re-litigate it, but do re-verify anything you are about to change.** The README overclaims in several places; those places are named.

Sections 3–9 are the work, in priority order. Each phase has an acceptance test. **A phase is not done until its test passes and you have pasted the actual output.**

Section 10 is the reporting format. Section 11 is the prohibition list — read it before writing any code.

---

## 1. Verified current state

Read from source on 18 September 2026. "Claimed" means the README or a docstring says it; "actual" means what the code does.

### 1.1 What genuinely works and must be preserved

| Component | File | State |
|---|---|---|
| C++ CLI product | `src/main.cpp` | Real. 6 commands: `index`, `search`, `symbol`, `ask`, `callers`, `stats`. Works. |
| Repo scanner | `src/repository.cpp` | Real. Walks dirs, correct extension filter (`.js/.jsx/.mjs/.cjs/.ts/.tsx`), correct ignore list (`.git`, `node_modules`, `dist`, `build`, `coverage`, `.cache`, `.idea`, `.vscode`). Keep as-is. |
| Identifier normalisation | `src/indexer.cpp::normalizeIdentifier`, `app/indexing/sqlite_index.py::normalize_identifier` | Real and correct. `getUserData` / `get_user_data` / `GET_USER_DATA` all collapse. Tested. Keep. |
| Exact symbol lookup | `SymbolIndexer::searchSymbols` | Real. Normalised exact match. Keep. |
| Span arithmetic | both chunkers | Byte offsets and line numbers are computed correctly and round-trip against source. Tested in both languages. **This is the most valuable thing in the repo. Do not break it.** |
| RRF fusion | `app/retrieval/hybrid.py::_rrf` | Real, standard `1/(60+rank)`. Keep the function, change what feeds it. |
| Snapshot isolation | `app/indexing/versioned.py` | Partially real. Retrieval *is* restricted to a named snapshot and unknown snapshots raise `KeyError`. The no-silent-fallback property is genuinely there. Keep the property, replace the backing store. |
| Metrics | `app/evaluation.py` | Real NDCG@k and MRR, correctly implemented. Keep. |
| CI | `.github/workflows/ci.yml` | Real and green. Extend it, do not replace it. |

### 1.2 Claimed but not implemented

These are the gaps. Each one is a phase below.

- **No frontend of any kind.** No `package.json`, no `web/`, no `ui/`, no HTML. Sections 9 and 10 of the original brief describe "redesigning the existing frontend" — there is nothing to redesign. This is net-new work and it is **P2**, not P0.
- **No dense/semantic retrieval anywhere.** `hybrid.py` has three legs: exact, BM25, and TF-IDF cosine. The README calls TF-IDF cosine a similarity leg, which is accurate, but nothing in the repo produces an embedding. No encoder, no vector store, no FAISS usage. The plan's Workstream A depends entirely on a dense encoder that does not exist yet.
- **No tree-sitter.** Both chunkers are regex + manual brace counting. Consequences, verified: a brace inside a string literal, template literal, comment, or regex literal will desynchronise `findMatchingBrace`. Arrow functions (`const f = () => {}`), object-literal methods, generators, and class fields are **not chunked at all** by either implementation. `FILE_SUMMARY` only fires when a file yields *zero* chunks, so a file with one detected function and ten arrow functions silently indexes 10% of its content.
- **No incremental indexing.** `SQLiteIndex.index_file` deletes every chunk for a file and re-inserts. `SymbolIndexer::indexFile` *appends* with no replacement, so re-indexing a file inside one process duplicates every chunk. There is no chunk-level content hash used for reuse, no embedding cache, no re-embed counter. **This is the plan's P1 goal and it is absent.**
- **No git integration.** `VersionedIndex` snapshots are arbitrary user-supplied strings held in an in-memory dict, persisted to one JSON blob. No commit SHAs, no parent relationships, no changed-file computation, no diff hunks, no commit messages. `index --commit <sha>` does not exist.
- **No imports/requires extraction.** The plan's B4 needs imports, requires, and call sites. Only a same-file call heuristic exists.
- **No Dockerfile.** Required by the submission guidelines (C1). Absent.
- **No development harness and no leakage firewall.** `mteb` evaluates only the `test` split. There is no loader for the 5,000-row `train` split and nothing prevents tuning against test qrels. Plan tasks A2 and T2.
- **`validate_submission.py` validates the wrong shape.** It checks for `metrics.mrr` / `metrics.ndcg_at_10` at the top level. `mteb.TaskResult.to_dict()` does not produce that shape. The validator will pass a file that is not a valid MTEB result and fail a file that is.

### 1.3 Actively wrong — fix or delete

1. **`src/repository.cpp::extractEdges` is not a call graph.** It emits `source = file_path`, `target = definition_name`, so `CodeGraph::callersOf()` returns *file paths, not calling functions*. It also matches the definition site itself as a call to itself, and it only collects definitions from the same file — so cross-file calls, which is the entire question the theme asks ("which files call tool XYZ"), are never detected. Confidence is hardcoded `0.55` for every edge with no verified/candidate distinction.

2. **`app/indexing/sqlite_index.py` FTS5 is broken three ways.** (a) The table is declared external-content over `chunks` with `content_rowid='id'`, but `chunks.id` is TEXT (`path:start:end`), not an integer rowid. (b) Rows are inserted with `rowid = hash(chunk_key) & 0x7fffffff` — Python's `hash()` on `str` is **salted per process**, so this is non-deterministic across runs and breaks reproducibility (plan test T1). (c) The delete path is `int(chunk_id) if chunk_id.isdigit() else 0`, and chunk ids are never all-digits, so it always deletes rowid 0 — FTS rows leak on every re-index. (d) Nothing in the repo ever *queries* `chunks_fts`. FTS5/BM25 is claimed in the README and is unreachable code.

3. **`src/retriever.cpp` contains a hardcoded query hack.** `if (term == "authentication" || term == "authenticate")` → also search `"auth"`. This is demo-specific fakery. Delete it.

4. **`app/retrieval/hybrid.py::QUERY_EXPANSIONS` is a fitted synonym table.** It maps `preprocessed → (preprocess, normalize, trim, sanitize)`. Note that `normalize` is the *name of the document* that `tests/test_hybrid_retrieval.py` asserts must rank first. **The test passes because the synonym table was written to make it pass.** Both the table and that assertion are a benchmark hack. Delete the table; rewrite the test against real behaviour.

5. **`tests/test_model_catalog.py` asserts invented version strings** (`"DeepSeek V4"`, `"Kimi K2.6"`, `"GPT-5.5"`, `"Gemini 3.5"`) and `len(catalog) >= 30`. These are unverifiable and lock in an out-of-scope subsystem. See Section 11.3.

6. **`SymbolIndexer::load` field-splitting is fragile.** In the escape tracking, `escaped` is assigned *after* the character is appended and toggles on any backslash. Round-trip on content containing `\\` adjacent to an encoded tab is not obviously correct. Verify with a property test before trusting the index format on a real repository, or replace the format (see Phase 3).

7. **Duplicated, divergent chunkers.** `src/chunker.cpp` and `app/parser/chunker.py` implement the same feature with different semantics — the C++ method pass requires the match to be inside a `CLASS` chunk, the Python one does not. The plan mandates **one** shared core. Two implementations of the same thing, disagreeing, is the exact failure the plan's "one shared core" rule exists to prevent.

---

## 2. Non-negotiable facts from the v3 plan

Do not re-derive these; they were verified against a local install and primary sources. Do not substitute values you remember.

**The screening task**
- MTEB task name `AppsRetrieval`, main score `ndcg_at_10`, English query → Python code.
- `eval_splits = ['test']` **only**. MTEB will not score train for you.
- 8,765 documents, 3,765 queries. **Exactly one relevant document per query** (min 1, max 1, mean 1.0).
- Because there is one gold document, NDCG@10 is a function of that document's rank and collapses to zero if it leaves the top 10. **Recall at candidate-pool size is a hard ceiling on the final score.**
- Average query 1,669.8 chars; average doc 573.5 chars; longest query 5,742; longest doc 289,048 (outlier). Truncation is a tail problem, not an average one.
- Hugging Face `CoIR-Retrieval/apps` additionally exposes a **5,000-row train split** that MTEB does not evaluate but we can load ourselves.

**The library, as installed: `mteb 2.21.0`**
- `mteb.get_task("AppsRetrieval")`, `mteb.evaluate(model, tasks, encode_kwargs=...)`, `mteb.TaskResult.to_dict()` all work.
- `mteb.models.abs_encoder.AbsEncoder` exists.
- **`.encode()` takes a `DataLoader[BatchedInput]`, not a `list[str]`.** The guidelines' snippet implies otherwise. Any pre/post-processing must be written against the DataLoader interface.
- `.similarity(embeddings1, embeddings2)` is the real rerank hook.
- `mteb.types.TopRankedDocumentsType = Mapping[str, list[str]]` — native two-stage rerank.
- `mteb.models.search_encoder_index.FaissSearchIndex` — built-in FAISS backend. **Do not write your own vector store.**
- `mteb.load_results(tasks=["AppsRetrieval"])` pulls published scores locally.

**Model selection is a lookup, not a bake-off**
- Start with **`codefuse-ai/F2LLM-v2-0.6B`** — published NDCG@10 **90.45** on this task, Apache 2.0, built on Qwen3-0.6B-Base, 1024 dims.
- Fallbacks down the same family if CPU budget fails: `F2LLM-v2-330M` (83.86), `Qwen3-Embedding-0.6B` (75.34), `jina-embeddings-v5-text-small` (73.28), `F2LLM-v2-160M` (58.95), `gte-modernbert-base` (56.41), `F2LLM-v2-80M` (52.20).
- **Sanity floor is 88.** A pipeline scoring in the 20s is *broken, not untuned* — 26.52 was the 2024 state of the art and is two years stale.
- **Licence trap:** `Salesforce/SFR-Embedding-Code` is **CC-BY-NC-4.0**, non-commercial. This is a Samsung-sponsored hackathon whose prize is a PRISM worklet. Measure it as a ceiling reference if you like; **it does not ship.**
- The current default in `scripts/evaluate_mteb.py` is `all-MiniLM-L6-v2`, which will score far below the floor. It is fine as a T0 conformance smoke model and must not be the submitted configuration.

---

## 3. Phase 1 (P0) — One core, one chunker, real AST

**Goal:** collapse the two divergent chunkers into one tree-sitter-backed implementation, and make span fidelity provable on a real repository.

Do this first. Every other phase inherits it, and every citation in the demo inherits its correctness.

### Tasks

1. **Pick the ownership boundary and write it down.** The plan wants the C++ runtime responsible for indexing/search where practical, with embeddings allowed in Python. Decide explicitly: either (a) tree-sitter C API linked into the C++ core with Python calling the CLI/bindings, or (b) tree-sitter Python as the single chunker with C++ retained for index and search. **Either is acceptable. Two chunkers is not.** Record the decision and the reason in `docs/ARCHITECTURE.md`.

2. **Introduce `tree-sitter-javascript` and `tree-sitter-python`.** JavaScript/TypeScript is first-class — the provided codebase is JavaScript. Python is needed because the screening corpus is Python.

3. **Chunk these node types**, preserving exact byte spans, line ranges, parent scope, and signature:
   - function declarations, arrow functions assigned to bindings, object-literal methods, class declarations, class methods, class fields with function values, generators, async variants, exported forms of all of the above, module-level top-level code, and one short file-summary chunk **per file, always** (not only when nothing else matched).
   - Oversized functions split at statement boundaries, with the signature repeated into each piece and the parent identifier retained.

4. **Delete the regex chunker that loses.** Do not leave it as a fallback. A silent fallback to a chunker with different semantics is the bug class the plan calls most damaging.

5. **Replace the brittle count assertions.** `tests/cpp_core_test.cpp` asserts `chunks.size() == 3`. Assert properties, not counts.

### Acceptance test — plan T7, span fidelity

Run over a **real mid-sized open-source JavaScript repository**, not `demo_repo/` (which is two files). For every chunk in the index, re-read the source file at the stored byte offsets and compare to the stored text.

```
PASS: 100.0% exact match  (N chunks across M files, repo=<name>@<sha>)
```

**One hundred percent. Not 99.9.** Also report chunk coverage: what fraction of non-blank, non-comment source bytes land in at least one chunk. Report the number; do not tune it yet.

---

## 4. Phase 2 (P0/P1) — Real dense retrieval, real fusion

**Goal:** the semantic leg becomes an actual embedding model, and the three legs fuse through one code path used by both products.

### Tasks

1. **Add an encoder wrapper** (`core/encode.py` or equivalent) with **separate query-mode and document-mode instruction prefixes**, both configurable, both pinned in a config file rather than loose in code. These models are prefix-sensitive and this is the cheapest large gain available.

2. **Confirm the similarity function matches what the model was trained with.** Cosine against dot product is a silent and large error. Assert it, don't assume it.

3. **Use `mteb.models.search_encoder_index.FaissSearchIndex`.** Do not hand-roll a vector store.

4. **Keep the existing RRF** in `hybrid.py::_rrf`. Replace the TF-IDF leg with the dense leg. Retain exact-symbol and BM25 as the other two legs. Graph-only candidates receive graph features — **never invent a dense or lexical rank for a method that did not retrieve the item.**

5. **Make FTS5 actually work and actually get queried.** Fix the external-content/rowid mismatch, replace `hash()` with a deterministic content hash (SHA-256 truncated, or an integer surrogate key stored in `chunks`), fix the delete path so re-indexing does not leak FTS rows, and wire a real FTS5/BM25 query path into retrieval.

6. **Delete `QUERY_EXPANSIONS`** and the `"authentication" → "auth"` branch in `src/retriever.cpp`.

7. **One fusion implementation, called by both wrappers.** This is the plan's first integration checkpoint.

### Acceptance tests

- **T0 conformance:** the guidelines' own snippet, unmodified, against pinned `mteb`, with a trivial sentence-transformer, produces a valid `appsretrieval_results.json` containing `ndcg_at_10` and `mrr_at_10`. Expect the `encode` signature to differ from the brief — **record the difference in the README, do not silently patch around it.**
- **T1 reproducibility:** same command twice on the same machine plus once from a fresh clone → NDCG@10 identical to **four decimal places**. If it is not, stop feature work; the usual causes are an unseeded shuffle, an unpinned model revision, or the salted `hash()` above.
- **Vector parity:** a test asserting that the harness wrapper and the service wrapper, given the same input string and prompt mode, produce **byte-identical vectors**.

---

## 5. Phase 3 (P1) — Genuinely incremental indexing

**This is the plan's explicitly-scored P1 goal.** It is currently absent, and it is the single most demoable thing in the system.

### The cache key

```
K_emb = H( encoder_revision, tokenizer_revision, template_revision, encoder_input_bytes )
```

**Path and line number are deliberately outside the key.** That is the whole point: a function that moves keeps its vector; a function whose body changes does not.

### Required behaviour

```
Move code within a file  → reuse embedding
Move code between files  → reuse embedding
Rename a file            → reuse embedding, zero re-embeds
Change a function body   → re-embed that chunk only
Delete code              → remove affected index entries
```

### Tasks

1. Chunk-level content hashing, stored in the index.
2. On change: diff against the indexed tree, reparse **only changed files**, reuse unchanged chunk facts, re-embed **only** chunks whose encoder input actually changed, refresh edges for files whose imports or exports moved.
3. Fix `SymbolIndexer::indexFile` to replace rather than append — re-indexing a file must not duplicate its chunks.
4. **Instrumentation is part of the deliverable, not an extra.** Expose real counters the API and tests can read: chunks total, embeddings total, changed chunks, re-embedded, reused, elapsed. The frontend will display these later and they must come from here, never from a literal.

### Acceptance test — plan T8, incremental equals rebuild

Run a scripted sequence of edits, deletes, renames, moves, and import changes. Then compare the incrementally-maintained index against a **clean rebuild**, on deterministic facts only: chunk hashes, symbol table, edge set.

**Never compare approximate-nearest-neighbour ordering.**

```
PASS  incremental == rebuild  (chunk hashes, symbols, edges)
PASS  re-embed counter: changed=2 re-embedded=2 reused=12479
PASS  file rename → re-embeddings = 0
PASS  single-function edit → re-index wall clock = 0.43s
```

If a rename triggers re-embeddings, **path or line number leaked into the cache key** — that is the specific mistake this design exists to avoid.

---

## 6. Phase 4 (P1) — Structural queries that are honest

### Tasks

1. **Rewrite edge extraction with tree-sitter queries.** Extract into an edge table carrying an **extractor name** and a **confidence**:
   - `imports`, `requires`, `exports`, `calls`, `references`, `file/module relationships`.
2. **Fix the caller semantics.** `callersOf(symbol)` must return the **calling function** (with its file and span), not the containing file. Cross-file calls must resolve — that is the actual question being asked.
3. **Bounded expansion:** from the top ~20 seeds, at most 2 hops, capped node count. No unbounded graph walks.
4. **Verified vs candidate is a first-class field.** Dynamic dispatch and computed property access will not resolve, and that is acceptable. **Presenting an unresolved guess as a fact is not.** Every structural answer cites the span that proves it, or is labelled `candidate`.

The theme slide names two queries explicitly — *which files call one tool before another*, and *where a settings deeplink is used*. Both must work on camera.

### Acceptance tests

- Ten fixtures return the exact expected file set, each with a citing span.
- Everything unresolved is labelled `candidate`, never `verified`.
- **T10, no silent fallback:** query an unindexed snapshot, a deleted file, a malformed query, and a syntactically broken source file. Each must return an **explicit named failure or an explicitly partial result**. An unindexed snapshot returns `snapshot not indexed` and **never** HEAD content.

---

## 7. Phase 5 (P1) — Real git, real version scoping

Replace the in-memory snapshot dict with actual git history.

### Tasks

1. `index --commit <sha>` as a real command. Store commit metadata, parent relationships, changed files, changed chunks.
2. **Membership decides eligibility.** A query pinned at commit K sees that tree's chunks and nothing else. **Similarity never selects a version** — two versions of one file are ~99% identical and land on nearly identical vectors, which is precisely why similarity is the wrong tool here.
3. For "when did X change" queries: enumerate the commit range, run BM25 over **commit messages** as the natural-language bridge, and rank the **diff hunks** themselves rather than the file versions.
4. Keep the existing explicit-failure property from `versioned.py` — it is the one part of this subsystem that is already right.

### Acceptance test — plan T9, zero cross-commit leakage

Randomised property test: pick a commit and a query, retrieve pinned to that commit, assert every returned occurrence belongs to that commit's tree. **At least 50 randomised trials, including branch switches and a merge commit.**

```
PASS  50/50 trials, 0 leaks
PASS  5/5 "when did X change" fixtures → correct commit in top 3
PASS  two near-identical file versions ordered correctly by change evidence
```

This is a **correctness property, not a quality metric** — the threshold is exact. Zero. If it is not green, **do not present evolutionary retrieval at all.** A version-aware retriever that silently mixes versions is worse than not claiming the bonus goal, because it is invisible and wrong.

---

## 8. Phase 6 (P1) — Evaluation integrity

This phase is what separates a score from a claim.

### Tasks

1. **Development harness on the train split (plan A2).** Load the 5,000-row train split from Hugging Face directly, compute NDCG@10 and MRR@10 in our own code. Calibrate our metric implementation against MTEB's by scoring one model both ways on test **once**, then never again until the final run.

2. **Leakage firewall (plan T2).** Implement as a loader flag: in development mode the test-split accessor **raises** rather than returns.
   ```
   PASS  test split accessed exactly 2 times this week (T4 calibration, final run)
   PASS  all N tuning runs logged with firewall=ON
   ```
   Any tuning decision made while the firewall was off is **void** — re-derive on train or drop the component.

3. **Candidate-pool recall ceiling (plan T3, A5).** **Before writing any reranker**, measure recall@10, @50, @100, @200 on the development set. With one gold document per query, recall at pool size is a hard upper bound on anything a reranker can achieve. Write the headroom down as a number.
   - If recall@50 is close to NDCG@10, there is nothing to rerank — fix candidate generation instead.
   - If recall@50 is low in absolute terms, the encoder is wrong and no downstream stage saves it.
   - **The expected outcome is that this cancels the reranker and query expansion.** That is the plan working, not failing.

4. **Noise band (plan T5).** Paired bootstrap over per-query NDCG, ≥1,000 resamples, producing a confidence interval for every A-vs-B comparison. **No component ships on a difference inside the band.**

5. **Ablation matrix (plan T6).** One row per component — base encoder, prefixes, reranker, fusion, expansion — each measured with the component removed, everything else fixed. Every shipped component must have a positive interval excluding zero.

6. **Fix `scripts/validate_submission.py`** to validate the actual `mteb.TaskResult.to_dict()` shape, not the bespoke `metrics.mrr` shape it currently checks.

7. **Sanity floor (plan T4).** Reproduce `F2LLM-v2-0.6B` unmodified against its published 90.45 first — that single number establishes both the floor and that the harness is wired correctly. Our score must then be **≥ 88**. If the reference model does not reproduce, **the bug is ours and it is in the harness.** Fix it there; every later measurement depends on it.

8. **Generalisation probe (plan T12).** Hold out a second code-retrieval task from a different distribution, plus ~30 hand-labelled JavaScript queries against the demo repository. Re-run the full configuration on both after every component is added. **A component that improves APPS while regressing either held-out set does not ship**, even if it raises the screening score — the hands-on round runs our code on their queries, so a benchmark-fitted mechanism fails publicly rather than privately.

---

## 9. Phase 7 (P2) — Frontend, built from scratch

**Do not start this until Phases 1–5 have green acceptance tests.** The plan is explicit: do not spend effort on cosmetic frontend work while core retrieval is incomplete.

There is no existing frontend. Build one. It must look like a serious developer tool — **VS Code + GitHub + intelligent code search** — and explicitly **not** like a chatbot. No "AI CODE AGENT" banner, no "Ask our AI anything…", no gratuitous animation.

Primary flow: **Search → Inspect → Understand → Verify → Explore.**

### Panels

- **Repository explorer** — repo, folders/files, branch, selected commit, indexing state.
- **Natural-language search** — prominent, handling queries like *"Where is authentication handled?"*, *"Which files call createSession()?"*, *"What changed in authentication?"*
- **Results** — file, line range, symbol, snippet, relevance, retrieval method/signals, related files.
- **Code viewer** — Monaco, with the relevant lines highlighted, opened at the exact stored span.
- **Structural view** — only the relations relevant to the selected result. **Not a giant unreadable graph.**
- **Verified vs candidate** — `✓ Verified` / `△ Candidate`, driven by the confidence field from Phase 4.
- **Commit selector** — historical retrieval must look visibly different from current-repo search.
- **Index status panel** — chunks, embeddings, changed/re-embedded/reused, elapsed. **Every value from the Phase 3 counters.**
- **Evidence panel** — organised into entry point / implementation / caller / test / configuration, **omitting categories that have no evidence** rather than showing empty ones.

### Acceptance test

Every number and relationship visible in the UI must be traceable to a backend call. Demonstrate by stopping the backend: the UI must show explicit error states, not stale or invented data.

---

## 10. Required deliverables

Report in exactly this structure.

**A. Architecture assessment** — what the original did, what changed, what you deliberately preserved and why.

**B. Completed improvements** — features now genuinely implemented, each with the passing test output that proves it.

**C. Remaining gaps** — everything still incomplete, stated plainly.

**D. Files changed** — the significant ones, with the reason for each.

**E. How to run** — exact commands for: build, start backend, start frontend, index a repository, index at a commit, search, run tests, regenerate the MTEB results file.

**F. Demo flow** — the sequence demonstrating: natural-language search → exact code evidence → structural relationship query → incremental re-index after an edit → historical commit retrieval.

**G. Test results table** — every acceptance test in this document, with `PASS` / `FAIL` / `NOT RUN` and the actual output.

**H. Docker + cold start** — a `Dockerfile` must exist, and the plan's T11 must pass: someone who did not write the code clones the tagged commit onto a clean machine, follows only the README, and gets both artifacts (results file regenerates, service answers a query) **with no questions asked of the author.** If the README fails, **the README is wrong, not the reader** — rewrite it from the observed failure.

---

## 11. Prohibitions

### 11.1 Do not fake anything

No hardcoded search results. No fake latency. No fake index statistics. No invented graph relationships. No fabricated commit history. No simulated semantic results. No placeholder numbers in the UI.

If the frontend needs a capability the backend lacks, **implement the minimum real backend support** rather than simulating it.

### 11.2 Do not claim success without evidence

If something cannot be tested or verified, **say so explicitly**. A named gap costs nothing. An unlabelled number that does not reproduce costs everything.

Do not report a latency or index-time figure measured on a substitute repository as though it came from the real one (plan T13). If the provided codebase arrives too late to re-measure, **name the substitute repository explicitly** in the README, the deck, and the video.

### 11.3 Out of scope — do not extend

- **No new embedding model.** No training, no fine-tuning.
- **No new parser.** Use tree-sitter.
- **No distributed index, message broker, or graph database.**
- **No LLM agent loop in the retrieval path.** `app/models/` (30 BYOK chat aliases, `config/models.json`) stays where it is as an optional, clearly-separated explanation layer. **Do not expand it, do not wire it into retrieval, and do not let it become the product's identity.** Also loosen `tests/test_model_catalog.py` — asserting specific invented version strings and a minimum catalog size locks in scope the plan explicitly excludes.
- **No crash-recovery machinery, compaction scheduler, tombstone collection, or durability layer.** If the index breaks on a demo repository we rebuild it in seconds.
- **No latency allocation table.** Measure end to end and report what you get.

### 11.4 Do not delete failing tests

Investigate the cause. The two fitted tests named in §1.3 are the exception: **rewrite** them against real behaviour rather than deleting them or preserving the fit.

### 11.5 Do not rewrite for preference

Do not replace a working technology because you would have chosen differently. The C++ core, the CLI surface, the RRF implementation, the normalisation logic, and the span arithmetic all work. Improve them in place.

---

## 12. Order of work — summary

| Phase | What | Priority | Gate |
|---|---|---|---|
| 1 | One tree-sitter chunker, delete the duplicate | **P0** | T7 span fidelity, 100% |
| 2 | Dense encoder, real FTS5, one fusion path | **P0/P1** | T0, T1, vector parity |
| 3 | Content-addressed incremental indexing | **P1** | T8 all four conditions |
| 4 | Honest structural queries, verified/candidate | **P1** | 10 fixtures + T10 |
| 5 | Real git, membership-based version scoping | **P1** | T9 zero leaks |
| 6 | Dev harness, firewall, recall ceiling, ablation | **P1** | T2, T3, T4≥88, T5, T6, T12 |
| 7 | Frontend workspace | **P2** | All values backend-traced |

**Start with Phase 1. Do not code ahead of the inventory — verify anything in Section 1 that you are about to change, then change it.**