# Retrieval workbench demo

The workbench returns ranked source spans with retrieval and execution evidence.
It is part of the current repository implementation. Its symbol/BM25/structural
router is a separate path from the submitted MTEB evaluator and does not produce
its P0 score. Generation after retrieval is outside Theme 01's scope.

## Run and select the model

Install the platform's PyTorch 2.8.0 wheel and `requirements-submission.txt`, then:

```sh
python -m uvicorn app.server:app --host 127.0.0.1 --port 8077
```

Open `http://127.0.0.1:8077`. The default accuracy tier is the pinned
**F2LLM-v2-1.7B** CPU encoder. Set `CODEDOCTOR_TIER=interactive` before starting
for the separate 0.6B tier. Label the tier in any recording; its latency cannot be
paired with a different tier's accuracy. Model downloads require connectivity;
the static page itself has no build step or CDN dependency.

## Select a target

| Target | What it supports | Limit |
| --- | --- | --- |
| Open a folder | Source parsing, ranked retrieval, rebuilding with recipe-keyed vector reuse | One `working` snapshot, even when `.git` exists; external history is not imported |
| CodeDoctor app | Built-in snapshots from this checkout's Git history, version comparison and cross-version ranking | Requires a normal Git checkout; source changes and available history determine snapshots |
| commander | Python/JavaScript grammar and retrieval smoke coverage on a third-party JavaScript library | Often only one distinct snapshot; this is not a measured generalization benchmark |
| CoIR AppsRetrieval train | Python prompt/example retrieval from prepared local artifacts | Requires `artifacts/apps/`; use train queries, preserve the test-access boundary |

Supported source extensions are `.py`, `.js`, `.jsx`, `.mjs`, `.cjs`, `.ts` and
`.tsx`; the arbitrary-folder guard is 4,000 source files. Unsupported languages
are outside the current grammar coverage. Source suffix support alone is not an
accuracy guarantee. The train-query policy is distinct from loader I/O isolation;
inspect loaders before preparing benchmark data, as `AGENTS.md` requires.

## Recording sequence

1. Open a small Python source folder, build its index and enter an actual question.
   Show ranked responses, file/line spans, pass reasons and query timings.
2. Inspect a result's evidence: per-method ranks and fusion contributions, and
   execution verdicts when the statement supplies runnable worked examples.
3. Edit source and rebuild. Show newly encoded and reused chunks rather than
   claiming an unmeasured rebuild speedup.
4. Select CodeDoctor app for a specific Git version, compare the same query on
   two versions, then demonstrate a ranking across indexed versions. Opening an
   external folder alone cannot demonstrate its Git history.
5. State the tier, device, fixture and limitations. Scores shown in the recording
   must identify whether they are historical or from validated current reporting.

A version comparison must disclose when dense coverage differs; otherwise changed
retrieval methods can be mistaken for code evolution. Unknown versions fail rather
than silently falling back to HEAD. Cross-version results retain provenance and
group related variants after verification, where executable examples are available.

## Dense preprocessing and cache reuse

```sh
python scripts/warm_workbench.py --target self --versions 4
```

The script uses the same default accuracy tier as the workbench. Stop other model
processes before warming on a memory-limited machine. The persistent vector cache
keys the encoder recipe and text; unchanged functions can reuse vectors across
edits/renames. In-memory indexes are rebuilt after a server restart. A corpus cache
is distinct from the disabled MTEB result cache. The FAQ permits GPU preprocessing
and updates; the query recipe uses CPU.

Oversized chunks above the current 6,000-character dense-input guard stay
lexically searchable but receive no dense rank. Whole-class and whole-file chunks
may therefore be excluded while their smaller function chunks remain eligible.
The page reports dense coverage and passes that did not run. Do not claim that
lexical-only latency measures dense semantic retrieval.

## Recorded fixture evidence

Retained hosted measurements used source `823aa87`, the 1.7B accuracy tier,
Python 3.12.13 and PyTorch 2.8.0+cpu on a 2.20 GHz Xeon (4 logical/2 physical CPUs).
They predate the present fixes. P1 used 40 Python functions in 10 modules, yielding
50 chunks, and three repetitions:

| Operation | Median seconds | Encoded / reused |
| --- | ---: | ---: |
| Full build | 102.777 | 50 / 0 |
| Unchanged | 1.512 | 0 / 50 |
| One function edit | 6.452 | 2 / 48 |
| Ten function edits | 29.507 | 13 / 37 |
| Add function | 4.345 | 2 / 50 |
| Delete function | 1.443 | 0 / 50 |
| Rename file | 1.423 | 0 / 50 |

All three repetitions matched fresh-build rankings on six probe queries, with
minimum vector cosine 1.0. See `results/hands-on/hands-on/p1.json` and its
`environment.json`; timing and memory are machine/fixture-specific.

The synthetic Bonus fixture has 12 lineages across three versions with incorrect
variants deliberately failing supplied examples. Correct variant first: 12/12
with execution verification and 1/12 without; median query 5.818 s, maximum
6.151 s (`results/hands-on/hands-on/bonus.json`). This is a functional check,
not retrieval accuracy on an unseen repository or the jury's internal dataset.
No new validated current P0 result is claimed by this guide.

## Limits to explain during the demo

Example parsing can fail; fragments may not run alone; passing visible examples
does not prove correctness. Queries without executable examples retain retrieval
ordering. Execution verifies Python code, while the parser supports the listed
JavaScript/TypeScript grammars. The subprocess runner limits resources and does
not isolate network or filesystem access. The jury's hands-on dataset is unseen
Python; a JavaScript demo is optional under the FAQ, not the required evaluation.
