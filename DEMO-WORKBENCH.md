# Retrieval workbench — local demo

> **Not part of the GitHub submission.** This file and everything it describes
> live in the working tree only. Nothing here has been committed or pushed.
> See *What is in the working tree* at the bottom for exactly which files are
> demo and which are engine, so the two can be separated.

The screening number is one half of Theme 01. The other half is someone
sitting in front of the system typing questions at it. This is that half: a
retrieval workbench, not a chat window. The centre of the screen says
**Relevant code**, never *AI answer* — generation after retrieval is out of
scope for the theme, and the UI is built to make that obvious at a glance.

## Run it

```bash
python -m uvicorn app.server:app --port 8077
```

Open <http://localhost:8077>. No build step, no CDN, no `npm install` — the
page is one self-contained HTML file, so it renders with the network
unplugged.

## The demo flow

```
pick a corpus → pick a version → ask a question → read the top 10
   → click #1 → inspect the evidence → switch version → same question
   → watch the ranking move → read the latency
```

That single path covers P0, P1, the Bonus and speed.

### Three corpora, and why each is there

| corpus | language | what it proves |
|---|---|---|
| **CodeDoctor · app/** | Python | Four *real* commits from this repository, so ranks genuinely move between versions. P1 and the Bonus. |
| **commander** | JavaScript | A third-party CLI library. Different language, different repository, no relationship to the screening corpus. Generalisation. |
| **CoIR AppsRetrieval · train** | Python | The screening corpus. The only one where the execution pass has worked examples to run — so it is the only place the third pass fires live. |

The AppsRetrieval corpus needs `artifacts/apps/`, which is gitignored
(hundreds of MB of float32). When it is absent the corpus is listed as
unavailable **with the command that regenerates it** rather than hidden:

```bash
python scripts/encode_apps.py
```

> **Train, never test.** The workbench runs the *train* queries. The test
> split has a two-access budget recorded in `data/test_split_audit_log.json`,
> and a live UI would spend it on a page load.

## What the screen shows, and why

**The pass strip** reports all four passes every time, including the ones that
did not run, with the reason the pipeline itself produced:

- `Execution — not run — pass executes Python; this corpus is JavaScript`
- `Execution — not run — statement has no machine-readable worked examples`
- `Dense — not run — no vectors for this snapshot, lexical only`

A demo that only shows the passes that succeeded is one a judge cannot check
by looking. This one can be checked by looking.

**The evidence panel** shows the arithmetic, not a claim about it. Reciprocal
rank fusion is a sum, so each result carries its per-method rank, raw score
and contribution, and those terms add up to the reported score exactly. When
the execution pass promotes something, the bonus appears as its own line and
the result is badged `was #4`.

**The timings** separate query encoding from scoring. On a repository a query
answers in single-digit milliseconds; on a 2,600-character AppsRetrieval
statement the same query takes seconds, and essentially all of the difference
is one transformer forward pass on a CPU. Lumping them together would make
retrieval look slow when what is slow is the encoder.

**A version comparison refuses to mislead.** If one side ran the dense pass
and the other did not, the movement in the table is partly the retrieval
changing rather than the code, so the page says the two versions are not
comparable instead of presenting the diff as version drift.

**The isolation chip** counts leaks against the snapshot's own document ids —
it is checked, not asserted, and it covers every candidate retrieved, not just
the ten that are displayed. An unknown version raises rather than falling back
to HEAD.

## Warming the dense index

The interactive encoder is F2LLM-v2-0.6B on CPU, roughly 2.5 seconds per
chunk. That is fine once and unacceptable per page load, so encode ahead of
time into the cache the server reads:

```bash
python scripts/warm_workbench.py --target self --versions 4
```

Re-running is cheap — anything already cached is skipped, so this is also how
to top the cache up after a commit. A chunk that survived a commit unchanged
is encoded once for the whole history, which is what makes four versions cost
far less than four times one. Measured on `app/`:

| | |
|---|---|
| cold warm-up, 4 versions | **15.3 min** — 1,250 chunks, of which only **269** were encoded and 878 came from the cache |
| server rebuild against that cache | **48 s**, 2 encodes, 1,145 reused (most of it is loading the weights) |
| query, dense + lexical | **339 ms**, of which 338 ms is the query encode and **1.4 ms** is scoring |
| query, lexical only (`commander`, 489 chunks) | **1.6–2.9 ms** |

Without a warm cache the workbench still answers, lexically, and the pass
strip says so. It never pretends a dense pass ran.

> **Stop the server first.** The encoder is ~1.4 GB resident, and the warm-up
> script loads its own copy. On a 8 GB machine running both at once puts the
> box into paging and the encode slows to a crawl — measured here as 1,000
> seconds of kernel time for zero chunks encoded. Warm first, then serve.

### Oversized chunks carry no vector

The chunker emits a whole-file `FILE_SUMMARY` chunk and a whole-class `CLASS`
chunk *in addition to* every function inside them, so those chunks can reach
tens of thousands of characters. Past the model's 8,192-token cap the input is
truncated, and since pooling is last-token the vector is then taken from an
arbitrary interior token — a **wrong** vector, not a degraded one. That is the
exact failure this project already paid about three NDCG points for once.

So a chunk whose encoder input exceeds `DENSE_MAX_INPUT_CHARS` (6,000) keeps
its row in the matrix, filled with zeros. Cosine against a zero row is exactly
0, so the dense pass returns no rank for it and fusion never invents one —
while BM25 and exact matching still find it. The pass strip says how many.

On `app/` that is 28 of 372 chunks, and a coverage check says 27 of those 28
are already in the index as their own functions. The one that is not
(`_extract_javascript_edges_part2`) is a split part the chunker could not split
further; it stays findable lexically.

There is a second reason for the cap, which is simply time: CPU attention is
quadratic, and a single 8k-token forward pass on this machine runs for minutes.
Before the cap existed the warm-up burned a thousand seconds of CPU without
finishing its first batch of eight.

## Known rough edges

- **`commander` has one version.** It was vendored in a single commit and
  never changed, so there is nothing to compare. The Versions tab says this
  rather than showing an empty diff. Use `app/` for P1.
- **A server restart drops the indexes.** They are in memory. Rebuilding a
  repository target takes seconds against a warm embedding cache.
- **The newest `app/` snapshot is the working tree**, not the HEAD commit, and
  is labelled `working tree` when the two differ. Older snapshots are read
  from the git object store, so an uncommitted edit cannot leak backwards.
- The previous VS Code-style explorer is still served at `/explorer.html`.

## What is in the working tree

Demo only — safe to drop entirely:

| path | |
|---|---|
| `web/dist/index.html` | the workbench (replaces the old explorer page) |
| `web/dist/explorer.html` | the old page, moved here so nothing is lost |
| `app/workbench.py` | the service behind it |
| `app/server.py` | `/api/workbench/*` routes, appended below the existing ones |
| `scripts/warm_workbench.py` | the offline encode |
| `tests/test_workbench.py` | 33 tests |
| `tests/test_server_api.py` | frontend assertions updated to the new page |
| `.claude/launch.json`, this file | local tooling |

Engine changes, independent of the UI:

| path | change |
|---|---|
| `app/retrieval/hybrid.py` | BM25 moved to an inverted index — **~440 ms → ~2 ms** per query on a 500-chunk repository, with bit-identical scores (asserted in `tests/test_workbench.py`). Symbol names normalised once instead of per query. Fusion now reports per-method rank and contribution. **Bug fix:** the dense pass raised `AttributeError` when document vectors were supplied without an encoder — the exact configuration the live service uses. |
| `app/indexing/versioned.py` | one retriever per snapshot instead of one per query, dropped when a commit is re-indexed; `warm()` and `last_timings()`. |
| `app/indexing/cache.py` | WAL journal and a busy timeout, so the warm-up script and the server can share `build/cache.db` instead of failing each other's locks. |

To revert the UI and keep the engine work:

```bash
git checkout web/dist/index.html && rm web/dist/explorer.html
```
