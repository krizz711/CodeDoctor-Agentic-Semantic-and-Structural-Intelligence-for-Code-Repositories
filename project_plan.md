SAMSUNG PRISM · GENERATIVE AI HACKATHON · 3RD EDITION 2026

# Theme 01: Agentic Code Intelligence
## Build plan, version four

Revised after adversarial review and a full fact-check against the installed MTEB library, the published CoIR results, and MTEB's own cached result set for this exact task. Five claims are corrected here, and the fifth — found last — changes the shape of the week more than the other four together. The plan carries a fourteen-test suite whose job is to tell us we are off track while there is still time to turn.

*(Note: Version 4 includes critical amendments prescribed by the LLM Council regarding noise-band testing overhead, evolutionary retrieval semantics, and CPU agentic loop latency.)*

Every number in this document was verified on 16 September 2026 against a local install of mteb 2.21.0, the CoIR paper (arXiv 2407.02883), and the Hugging Face dataset card. Numbers that could not be verified are marked as unverified rather than quietly rounded.

Supersedes versions one, two, and three.

---

## ONE
## What the fact-check overturned

Versions one and two were written before the library and the dataset had been read. Five of their claims did not survive contact with either. Three were load-bearing, and the first one below is the reason this document exists.

### Wrong, and this reshapes the week: the achievable score is not 26, it is 90
Version two set expectations from the CoIR paper's 2024 results table, where the best model on APPS was Voyage-Code-002 at 26.52. MTEB ships a cached result set for this exact task covering **235 model runs**, and reading it directly overturns that. The median run scores 9.31, the ninetieth percentile scores 90.01, and the best scores 98.62.

The obvious worry is that the task changed between 2024 and now. It did not, and the check is decisive: 226 of the 235 runs share a single dataset revision, and `intfloat/e5-base-v2` scores **11.52** in MTEB's records, which is the CoIR paper's figure for E5-base to the decimal. `bge-base-en-v1.5` scores 6.45, `gte-base` 5.97, `bge-m3` 14.74, `e5-mistral-7b-instruct` 23.46. The old models still score exactly what the paper says. The task is the same; the models got better.

The mechanism almost certainly includes in-domain training. APPS ships a public 5,000-pair train split, and the CoIR authors themselves acknowledge that models have begun to overfit the leaderboard — a contamination-limited successor benchmark now exists on the MTEB leaderboard precisely because of it. That is not a reason to avoid these models. It is a reason to be the team that says so out loud.

**Three consequences.** The T4 sanity floor was set at “clearly above 11.52”, which would have passed a pipeline scoring twelve while an Apache-2.0 model any teammate can download scores ninety; it is now 88. Model selection stops being a bake-off and becomes a lookup plus one verification run. And the reranker and query expansion are now likely dead on arrival, because there is very little headroom above ninety for them to recover.

### The shortlist, read off MTEB's cache rather than guessed
Permissively licensed, small enough for CPU, with published NDCG@10 on this task: **codefuse-ai/F2LLM-v2-0.6B at 90.45** (Apache 2.0, built on Qwen3-0.6B-Base, 1024 dimensions), F2LLM-v2-330M at 83.86, Qwen3-Embedding-0.6B at 75.34, jina-embeddings-v5-text-small at 73.28, F2LLM-v2-160M at 58.95, gte-modernbert-base at 56.41, F2LLM-v2-80M at 52.20. The same family reaches 97.19 at 14B if we ever wanted it, and the 80M variant is small enough to keep the whole JavaScript index resident for Workstream B. **Start at F2LLM-v2-0.6B.**

### Wrong: the queries are enormous and truncation is a first-order problem
The CoIR paper's dataset table lists APPS average query length as 1,400 and average corpus length as 575, and I read those as words. MTEB's own computed statistics for the task give an average query length of **1,669.8 characters** and an average document length of **573.5 characters**, with a total of 11,313,530 characters across the split. The paper's figures are characters too. A 1,670-character query is roughly 250 to 400 tokens, comfortably inside a 512-token encoder. **Truncation is a tail problem, not an average one** — the longest query is 5,742 characters and the longest document is an outlier at 289,048. Handle the tail, do not redesign around it.

### Wrong: the final encode is an overnight job we must schedule around
The test split is **8,765 documents and 3,765 queries**. That is roughly 12,500 short texts totalling eleven megabytes of characters. On eight CPU cores even a half-billion-parameter encoder finishes this in well under an hour, and a 100M-class model in minutes. Version one built a whole risk mitigation around an eleven-hour run that does not exist. The sizing task stays, because measuring is cheap and assuming is what produced this error, but it drops from half a day to an hour and stops driving the calendar.

### Wrong: tune on the train split, because MTEB will evaluate it
The Hugging Face dataset does carry a train split of 5,000 query-document pairs alongside the 3,770 test pairs, so clean tuning data exists. But the MTEB task object declares `eval_splits = ['test']` and nothing else. MTEB will not score the train split for us. **We have to build the development harness ourselves**, loading the train qrels directly from Hugging Face and computing NDCG@10 and MRR@10 in our own code. That is half a day nobody had budgeted, and it is now task A2.

### Right, and more decisive than version one realised
BM25 scores **0.95** NDCG@10 on APPS in the CoIR paper's own results table. For scale, in the same table UniXcoder scores 1.36, GTE-base 3.24, BGE-base 4.05, Contriever 5.14, BGE-M3 7.37, OpenAI-Ada-002 8.70, E5-base 11.52, E5-Mistral-7B 21.33, and Voyage-Code-002 26.52. Lexical retrieval is not weak here, it is absent. And the spread across dense models is a factor of eight, which means **model selection dominates every other decision in Workstream A combined.**

### New, and it changes the retrieval design
MTEB's statistics report **exactly one relevant document per query** — minimum one, maximum one, average 1.0, across all 3,765 queries. This is a single-answer task. NDCG@10 therefore reduces to a function of the rank of one document, and collapses to zero the moment that document falls out of the top ten. Two consequences follow, and the revised plan is built around them: reranking the head of the list is worth a great deal, and it is worth *nothing at all* if the single gold document is not already inside the candidate pool. Recall at the pool size is a hard ceiling on the final score, and it must be measured before any reranker is written.

---

## TWO
## Verified ground truth

Everything below was read off the installed library or a primary source, not recalled. Anything the team later disputes should be re-derived the same way rather than argued about.

### The task
– MTEB task name `AppsRetrieval`, type Retrieval, main score `ndcg_at_10`, languages English query against Python code.
– Evaluation splits: `['test']` only.
– 8,765 documents, 3,765 queries, 3,765 relevant pairs, exactly one gold document per query.
– Average document 573.5 characters, average query 1,669.8 characters; longest query 5,742 characters, longest document 289,048 characters.
– Hugging Face `CoIR-Retrieval/apps` additionally exposes a 5,000-row train split, which MTEB does not evaluate but we can load ourselves.

### The library, as installed
`mteb 2.21.0`. The guidelines' code snippet is close but not current, and the difference will cost a day if it is discovered late. Verified surface:
```python
mteb.get_task("AppsRetrieval") # works
mteb.evaluate(model, tasks, encode_kwargs=...) # works
mteb.TaskResult.to_dict() # works, so the JSON path in the brief is fine
mteb.models.abs_encoder.AbsEncoder # exists
    .encode(inputs: DataLoader[BatchedInput], *, # <-- NOT a list[str]
        task_metadata, hf_split, hf_subset, prompt_type, **kwargs)
    .similarity(embeddings1, embeddings2) # <-- the rerank hook, it is real
mteb.types.TopRankedDocumentsType = Mapping[str, list[str]] # <-- native two-stage rerank
mteb.models.search_encoder_index.FaissSearchIndex # built-in faiss backend
mteb.load_results(tasks=["AppsRetrieval"]) # pulls published scores locally
```
Two things follow. First, **the encoder receives a DataLoader of batched inputs, not a list of strings**, so any pre and post-processing pipeline has to be written against that interface; the brief's snippet implies otherwise. Second, **we do not need a blind bake-off**. MTEB ships a results cache covering this exact task, and its registry already knows the plausible candidates — codesage-v2 in three sizes, the codefuse F2LLM v2 family from 80M to 14B, and the cross-encoder ettin-reranker family from 17M to 1B. Query the published scores first, shortlist three, and measure only those.

### One licence trap, verified
Salesforce SFR-Embedding-Code, the strongest code retriever most write-ups will point us at, is released under **CC-BY-NC-4.0**, explicitly for research purposes in support of an academic paper. Its published CoIR averages are 61.9 for the 400M and 67.4 for the 2B. Those numbers are attractive and the licence is non-commercial. This is a Samsung-sponsored hackathon whose stated prize is a PRISM worklet with a Samsung mentor. Shipping a non-commercial model as the core of a prototype a company may want to carry forward is a real problem, and it is the kind of thing a jury of R&D engineers asks about. Measure it as a ceiling reference, ship something permissive.

---

## THREE
## The deliberation, in brief

Three reviewers went through version one. One defended it, one attacked it, one refereed and wrote this plan. What follows is only what changed — agreement is not recorded.

### The motion: split the work into two products
**For.** The screening gate is binary and eliminative. The guidelines say the top submissions are selected for hands-on evaluation. A repository engine that never clears screening is never seen. The dataset facts reinforce this: the gate is a flat single-answer retrieval task with no repository structure in it whatsoever, so it cannot be won as a side effect of building the demo.
**Against.** The split is right in kind and badly wrong in proportion. Version one staffs two of four people on the score for the entire build week. But the corpus is 8,765 documents, MTEB ships the harness, the faiss backend, the model registry and the published scores. That is not five days of work, it is closer to two. Meanwhile the scoring rubric the jury actually applies is thirty percent working prototype, twenty-five technical depth, twenty innovation, fifteen relevance to theme and ten presentation. NDCG appears nowhere in it.
**Conceded.** The proportion was wrong. Workstream A is rebalanced to two people for two days and one person thereafter, and the freed engineer moves to the prototype on day three.

### The attack that landed hardest
**Against.** Version one claims “A is B's evaluation harness — tune where the signal is, deploy where the demo is.” That claim is false. APPS is competitive-programming problem statements matched to whole standalone Python solutions, one gold answer each. The demo is natural-language questions about a JavaScript repository, where the answer is a set of interdependent snippets. Different language, different granularity, different cardinality, different notion of relevance. Tuning a prefix on APPS tells you almost nothing about repository search, and believing otherwise is how a team ships a prototype tuned on the wrong distribution.
**For.** Partly right. The transfer is real but narrow: it covers encoder family, prompt-prefix mechanics, fusion plumbing and reranker wiring. It does not cover relevance. The claim should be narrowed, not deleted.
**Ruling.** Narrowed, and a test added. The plan no longer asserts transfer; it measures it. T12 holds out a second, different retrieval task and the hand-labelled JavaScript set, and any component tuned on APPS must not regress either of them. A component that helps APPS and hurts the JavaScript set is a benchmark hack and gets cut on the evidence rather than on taste.

### The technical point that changed the build order
**Against.** Version one calls the reranker the highest-leverage task after model choice. With exactly one gold document per query that is only true conditionally. If the bi-encoder does not place the gold document inside the candidate pool, no reranker recovers it, and the effort is wasted. Recall at the pool size is the ceiling, and version one never measures it.
**For.** Accepted without argument. That is simply correct.
**Ruling.** Recall at the pool size becomes T3 and is measured *before* any reranker is written. If recall@50 is already high, the reranker is the right investment and the ceiling tells us exactly how much it can be worth. If it is low, candidate generation is the bug and reranking is a distraction.

### Unresolved, and therefore pre-registered
**Against.** Query expansion is unlikely to pay here. The queries already average sixteen hundred characters of detailed problem specification; there is little the model does not already know about intent.
**For.** Length is not the issue. The gap is modality — English prose against Python source. A pseudo-solution moves the query into the document's space, and that is the mechanism, independent of how verbose the query already is.
**Ruling.** Neither side produced evidence, so it is not decided by argument. The experiment is pre-registered with its success threshold and its kill date written down before it runs, and it is run on the development harness only.

### Where the dissent still stands
The reviewer against the plan did not withdraw one objection, and it is recorded rather than smoothed over. **We are building Workstream B on an asset we have not seen.** The theme states that a sample open-source JavaScript codebase is provided with the problem statement. If it arrives late, or is far larger or stranger than a typical library, several of B's pass conditions — particularly the sub-second re-index — may turn out to be calibrated against the wrong repository. Their position is that no latency claim should be rehearsed into the video until the real repository is indexed. That is now T13.

---

## FOUR
## The test suite

This is the part the plan was missing. A pass condition tells you a task is finished; these tell you the project is still pointed in the right direction. Each one names what happens on failure, because a test with no failure action is decoration. T0 to T6 guard the score, T7 to T11 guard the prototype, T12 to T14 guard against fooling ourselves.

**T0 Harness conformance**
Run the brief's own snippet, unmodified, against the pinned mteb version with a trivial sentence-transformer. This is the first thing anyone runs, before any retrieval code exists.
**Green when**: A valid `appsretrieval_results.json` is produced containing `ndcg_at_10` and `mrr_at_10`.
**On failure**: Do not patch around it. Record the exact API difference, pin the version that works, and write both the version and the working call into the README. The encode signature in 2.21.0 takes a DataLoader, so expect this.

**T1 Reproducibility**
The same command run twice on the same machine, and once by a different team member from a fresh clone.
**Green when**: NDCG@10 identical to four decimal places across all three runs.
**On failure**: Stop feature work. A non-reproducible pipeline makes every measurement below meaningless, and the usual causes are an unseeded shuffle or an unpinned model revision.

**T2 Leakage firewall**
Development runs must be physically unable to read the test qrels. Implement it as a loader flag: in development mode the test-split accessor raises rather than returns, and the development harness reads only the 5,000-row train split from Hugging Face.
**Green when**: Every tuning run in the log ran with the firewall on, and the test split was read exactly twice all week — once for T4 and once for the final submission run.
**On failure**: Any tuning decision made while the firewall was off is void. Re-derive it on train or drop the component. This is the difference between a score and a claim.

**T3 Candidate-pool recall ceiling**
Before writing a single line of reranker, measure recall@10, @50, @100 and @200 of the bi-encoder on the development set. With one gold document per query, recall at the pool size is a hard upper bound on what any reranker can achieve.
**Green when**: Recall@50 is high enough that the gap between it and NDCG@10 is large — that gap is exactly the headroom a reranker can convert, and it must be worth the day it costs.
**On failure**: If recall@50 is close to NDCG@10 there is nothing to rerank; fix candidate generation instead. If recall@50 is low in absolute terms, the encoder is wrong and no downstream stage will save it. Either way, this test redirects the week.

**T4 Sanity floor against the published distribution**
Compare our test-split score against MTEB's cached results for this exact task: median run 9.31, ninetieth percentile 90.01, best 98.62, and F2LLM-v2-0.6B specifically at 90.45.
**Green when**: We are at or above **88**. Anything in the twenties means the pipeline is broken rather than untuned — that was the 2024 state of the art and it is two years stale.
**On failure**: Do not tune, debug. The usual causes are a prefix applied to the wrong side, query and document embeddings swapped, a similarity function the model was not trained with, or silent truncation. Re-run the reference model unmodified until it reproduces its published 90.45, then diff our pipeline against that.

**T5 Noise band before any claim [AMENDED BY COUNCIL]**
Use standard significance testing libraries or simplified ablation on per-query NDCG to compute a confidence interval for every A-versus-B comparison. Compute it once, early, and reuse the machinery. (Do not burn hours building a custom 1,000-resample bootstrap).
**Green when**: Every improvement recorded in the log states its interval, and no component is adopted on a difference that falls inside it.
**On failure**: Discard the claim. Most small retrieval gains on a 3,765-query set are noise, and a team that chases them spends its last three days moving sideways.

**T6 Ablation matrix**
One row per component — base encoder, prefixes, reranker, fusion, expansion — each measured with the component removed and everything else held fixed.
**Green when**: Every component that ships has a positive interval that excludes zero, and the matrix is in the deck.
**On failure**: Cut the component. This doubles as jury preparation: the first question a Samsung R&D engineer asks about an architecture diagram is which box actually earns its place.

**T7 Span fidelity**
For every chunk in the JavaScript index, re-read the source file at the stored byte offsets and compare to the stored text.
**Green when**: One hundred percent exact match. Not 99.9.
**On failure**: Stop and fix the chunker before anything is built on it. Every citation in the demo, and every structural claim, inherits this error if it is wrong.

**T8 Incremental equals rebuild**
After a scripted sequence of edits, deletes, renames, moves and import changes, compare the incrementally maintained index against a clean rebuild on deterministic facts only — chunk hashes, symbol table, edge set. Never compare approximate-nearest-neighbour ordering.
**Green when**: The two agree exactly, and the re-embedding counter shows that only genuinely changed chunks were re-encoded. A file rename triggers zero re-embeddings.
**On failure**: The cache key is wrong. Almost always this means path or line number leaked into the key, which is the specific mistake the design exists to avoid.

**T9 Zero cross-commit leakage**
Randomised property test: pick a commit and a query, run retrieval pinned to that commit, assert every returned occurrence belongs to that commit's tree. At least fifty randomised trials, including branch switches and a merge commit.
**Green when**: Zero leaks. This is a correctness property, not a quality metric, so the threshold is exact.
**On failure**: Do not present evolutionary retrieval at all until it is green. A version-aware retriever that silently mixes versions is worse than not claiming the bonus goal, because it is invisible to the user and wrong.

**T10 No silent fallback**
Query an unindexed snapshot, a deleted file, a malformed query and a syntactically broken source file.
**Green when**: Each returns an explicit, named failure or an explicitly partial result. An unindexed snapshot returns *snapshot not indexed* and never HEAD content.
**On failure**: Fix immediately and add the case to the fixture set. Silent fallback is the single most damaging bug class in a retrieval system, because it produces confident wrong answers.

**T11 Cold-start reproduction**
A team member who did not write the code clones the tagged commit onto a machine that has never run the project, follows only the README, and runs the Docker build.
**Green when**: Both artifacts appear — the results file regenerates and the service answers a query — with no questions asked of the author.
**On failure**: The README is wrong, not the reader. Rewrite it from the observed failure and repeat with a different person.

**T12 Generalisation probe**
The one the dissenting reviewer insisted on. Hold out a second code-retrieval task from a different distribution, plus the thirty hand-labelled JavaScript queries. Re-run the full configuration on both after every component is added.
**Green when**: No component that ships improves APPS while regressing either held-out set.
**On failure**: Cut the component from the shipped configuration even if it helps the screening score. A mechanism that only works on one benchmark will fail in the hands-on round, where they run our code on their queries — and it will fail publicly rather than privately.

**T13 Real-repository calibration**
Every latency and index-time number must be measured on the codebase actually provided with the problem statement, not on the substitute repository used to start development.
**Green when**: The numbers in the video and the deck were produced on the real repository, on the machine used for recording, and the log that produced them is in the repository.
**On failure**: Re-measure and re-record. If the real repository arrives too late to re-record, state the substitute by name in the video and the deck. An honest caveat costs nothing with this jury; an unlabelled number that does not reproduce costs everything.

**T14 Agentic Turnaround Latency [NEW FROM COUNCIL]**
Every iteration of an agentic loop (plan -> search -> read -> refine) incurs inference overhead. Measure the end-to-end latency of a complete loop running entirely on CPU.
**Green when**: The full multi-hop retrieval loop completes within acceptable interactive limits for a user (e.g., < 3-5 seconds).
**On failure**: Do not compromise the "working prototype" score (30%) for marginal NDCG gains. Immediately fall back to smaller encoder models (F2LLM-v2-160M or 80M) to bring latency back within bounds.

---

## FIVE
## Workstream A, revised

Two people for two days, then one. Shorter than version one because the corpus is small and MTEB supplies more than we assumed, and reordered because T3 now decides whether the reranker is worth building at all.

**A0 Register the team**
R1 · today, 30 minutes · blocks everything
Registration closes **16 September, 23:59**. Team name in the form CollegeName_TeamName, up to four members, one theme, one submission. The final submission link goes only to registered teams.
**Passes when**: Confirmation received and the exact team-name string committed to the README.

**A1 Harness conformance**
R1 · 3 hours · needs A0, runs T0 and T1
Install mteb, pin the version, run the brief's snippet with a trivial encoder. Expect the encode signature to differ from the brief; record the difference rather than working around it silently.
**Passes when**: T0 and T1 green, and the pinned version plus the working call are in the README.

**A2 Development harness on the train split**
R2 · half a day · needs A1, enables T2
MTEB evaluates only the test split, so build our own: load the 5,000-row train split from Hugging Face, compute NDCG@10 and MRR@10 ourselves, and wire the leakage firewall so that in development mode the test accessor raises. Verify our metric implementation against MTEB's by scoring one model both ways on the test split once, on day one, and never again until the final run.
**Passes when**: Our NDCG@10 agrees with MTEB's to four decimal places on the one calibration run, and T2's firewall is active for every subsequent experiment.

**A3 Reproduce the reference model, do not hold a bake-off**
R2 · 3 hours · needs A2, runs T4
Model selection is already answered by MTEB's cache. Download codefuse-ai/F2LLM-v2-0.6B, run it unmodified, and confirm it reproduces roughly its published 90.45. That one number establishes both the floor and the fact that our harness is wired correctly.
Only if it will not run on our CPU within budget, walk down the family — 330M at 83.86, 160M at 58.95, 80M at 52.20 — and record the quality traded for speed. Record every candidate's licence; SFR-Embedding-Code is CC-BY-NC-4.0 and does not ship.
**Passes when**: The reference model reproduces its published score within a point or two, and T4 is green at 88 or above.
*If it does not reproduce, the bug is ours and it is in the harness. Fix it here, because every measurement for the rest of the week depends on this one being right.*

**A4 Prefixes and the similarity function**
R1 · half a day · needs A3, runs T4
Sweep query-side and document-side instruction prefixes independently. Confirm the similarity function matches what the model was trained with — cosine against dot product is a silent and large error. Then run the sanity floor.
**Passes when**: Best prefix pair beats no-prefix outside the T5 noise band, and T4 is green against E5-base's published 11.52.

**A5 Measure the ceiling, and expect it to cancel the next two tasks**
R2 · 2 hours · needs A4, runs T3
Recall@10, @50, @100 and @200 on the development set. With one gold document per query and a bi-encoder already near ninety, recall@50 is likely very high and the headroom very small. Measure it rather than assuming in either direction.
**Passes when**: T3 green, and the headroom written down as a number that explicitly authorises or cancels A6 and A7.
*The honest expectation is that this kills both. That is a good outcome, not a disappointment — it buys two engineer-days for the prototype, which is where the marks are.*

**A6 Reranker, only if A5 authorised it**
R2 · 1 day · conditional on T3
Two-stage retrieval is natively supported and the registry carries the ettin-reranker family from 17M to 1B. Do not open this task on enthusiasm; open it on the number A5 produced.
**Passes when**: Development NDCG@10 improves outside the noise band.
*If A5 showed little headroom, skip it and move to Workstream B the same morning.*

**A7 Query expansion, almost certainly not**
R2 · 0 to 1 day · conditional on T3
Version two pre-registered this as a genuine open question. The score distribution has largely settled it: a pipeline at ninety has perhaps eight points of headroom in total, and expansion is an expensive way to chase a fraction of that.
**Passes when**: Only opened if A5 showed headroom *and* A6 did not consume it.
*Default is to skip. Record the decision and the number behind it, so the deck can say we considered and rejected it on evidence rather than never having thought about it.*

**A8 Freeze, ablate, run, release**
R1 · 1 day · needs A3 to A7, runs T5, T6, T11, T12
Freeze the configuration to a single file. Run the ablation matrix. Then the test split, once. Create the release and attach the results file.
**Passes when**: T5, T6, T11 and T12 all green, the frozen file is what the run read, and the test split has been touched exactly twice all week.

---

## SIX
## Workstream B, revised

Unchanged in substance from version one, which survived review intact, but now gaining the freed engineer on day three and carrying explicit tests. The ordering reflects that B5 and B6 are the two goals the hands-on round actually names.

**B0 Obtain or substitute the repository**
E1 · 1 hour · today, gates all of B
Check whether the sample JavaScript codebase has been provided. If not, pick a mid-sized open-source JavaScript library today, record the choice and the commit, and proceed. Do not wait.
**Passes when**: A repository path and pinned commit are recorded, and if it is a substitute that fact is written where the video script will see it, per T13.

**B1 JavaScript chunker with exact spans**
E1 · 1 day · needs B0, runs T7
Tree-sitter JavaScript. Functions, methods, classes, module-level code and a file-summary chunk, with byte offsets and line ranges. Oversized functions split at statement boundaries with the signature repeated and the parent identifier retained.
**Passes when**: T7 green at one hundred percent.

**B2 Index and exact symbol lookup**
E1 · half a day · needs B1
SQLite: chunks, symbols, FTS5 over source and signature and doc comment. Raw identifiers for exact lookup, normalised form for search.
**Passes when**: Twenty known symbols return their definition at rank one; camelCase and snake_case variants both hit.

**B3 Hybrid retrieval and the labelled set**
E2 · 1 day · needs B2, feeds T12
Dense plus BM25 plus exact symbol, fused by reciprocal rank fusion. Hand-write and label thirty natural-language queries against the repository. Note that unlike APPS, several of these will have more than one correct answer — that difference is the point, and it is what T12 measures.
**Passes when**: Hybrid recall@10 beats both FTS5-only and dense-only on the labelled set.

**B4 Structural and usage queries**
E2 · 1 day · needs B2, runs T10
Imports, requires and call sites into an edge table with extractor and confidence. Bounded expansion, two hops, capped node count. The theme names two queries explicitly — which files call one tool before another, and where a settings deeplink is used — so both must work on camera.
**Passes when**: Ten fixtures return the exact expected file set with a citing span each, unresolved cases are labelled *candidate* rather than *verified*, and T10 is green.

**B5 Incremental indexing, the P1 goal**
E1 and E3 · 1.5 days · needs B1, B2, runs T8
Chunk-level content hashing with the embedding key over encoder revision, tokenizer revision, template revision and encoder input bytes. Path and line number stay outside the key.
**Passes when**: T8 green on all four conditions, and the sub-second re-index is measured on the real repository per T13.

**B6 Evolutionary retrieval, the bonus goal [AMENDED BY COUNCIL]**
E2 · 1.5 days · needs B5, runs T9
Membership decides eligibility; diff hunks and commit-message BM25 are insufficient. *Council mandate: Integrate semantic versioning awareness to properly rank and distinguish highly similar code snippets that exist across historical states. Pure diff-lexical search fails to capture semantic equivalence.*
**Passes when**: T9 green at zero leaks, and for five fixtures the correct commit is in the top three.

**B7 Facet-covering bundles**
E2 · 1 day · needs B3, B4
Greedy marginal gain per token over facets: entry point, implementation, caller, test, configuration.
**Passes when**: Covers more required facets than plain top-k at equal token budget on ten multi-file tasks, and reports facets it could not fill.
*Still the first thing cut if the week slips, and cut before B6 rather than after.*

**B8 Demo path and honest numbers**
E1 · half a day · needs B3 to B6, runs T13
Measure p50 and p95 warm, cold start and full index time on the real repository. Rehearse the recorded sequence end to end, twice.
**Passes when**: T13 green: every number spoken in the video was measured on the repository being shown.

---

## SEVEN
## Calendar, owners, and the standing rules

### 16 Sep Tue
**A** A0 register, closes 23:59. A1 harness conformance, T0 and T1.
**B** B0 obtain or substitute the repository. B1 begins.

### 17 Sep Wed
**A** A2 development harness and the leakage firewall.
**B** B1 closes on T7. B2 begins.

### 18 Sep Thu
**A** A3 reproduce F2LLM-v2-0.6B, T4 at 88. A4 prefixes. A5 ceiling, T3.
**B** B2 closes. B3 begins.

### 19 Sep Fri
**A** A6 and A7 *only* if T3 authorised them. Expect not. **R2 moves to B today.**
**B** B3 closes. B4 begins, now with two engineers.

### 20 Sep Sat
**A** C5 contamination note: run the held-out probe and write the honest paragraph.
**B** B4 closes on T10. B5 begins.

### 21 Sep Sun
**A** Idle by design. If A is still running, something went wrong on the 18th.
**B** B5 continues, two engineers.

### 22 Sep Mon
**A** Score freeze. A8 ablation matrix, T5 and T6.
**B** B5 closes on T8. B6 begins.

### 23 Sep Tue
**A** A8 test-split run and release. Second and last touch of the test split.
**B** B6 closes on T9. B7 if the week held. C1 begins.

### 24 Sep Wed
**A** T11 cold-start reproduction, run by someone who did not write it.
**B** B8 and T13 on the real repository. Video and deck.

### 25 Sep Thu
**A** Tag **PRISM_GENAI_HACKATHON_Y2026** and submit by noon.
**B** Same.

### C5 The contamination note, and why it is a differentiator
We will submit a score near ninety. So will every team that reads the MTEB cache. That number therefore differentiates nobody, and a jury of research engineers will know it. What does differentiate us is being the team that explains it.

The deck gets one honest slide: APPS ships a public training split, the leading models are trained in domain, the benchmark's own authors acknowledge overfitting, and a contamination-limited successor exists for that reason. Then our own evidence — the T12 held-out probe and the thirty hand-labelled JavaScript queries — showing what our pipeline does where it cannot have been trained. **This is the twenty-five percent technical-depth mark, and it costs an afternoon.** A team that reports ninety without understanding it looks lucky; a team that reports ninety and then shows the held-out number looks like it belongs in a research organisation.

### Owners
**R1** harness, freeze, release, tag: A0, A1, A4, A8, C4. 
**R2** development harness, reference model, ceiling: A2, A3, A5 — then moves to Workstream B on 19 September, two days earlier than version two planned, because model selection turned out to be a lookup. 
**E1** chunker, index, incremental indexing, demo path: B0, B1, B2, B5, B8. 
**E2** hybrid, structural, evolutionary, bundles: B3, B4, B6, B7.

**Three people:** R1 absorbs A2 and A3, and B7 is cut on day one rather than day eight. 
**Two people:** one takes all of A through A5 and stops there, the other takes B0 to B5; B6 and B7 are both cut. A half-built version-aware retriever that fails T9 is worse than not claiming the bonus goal.

### Standing rules
– **No component ships on a difference inside the T5 noise band.** This is the rule most likely to be broken in the last seventy-two hours.
– **The test split is touched exactly twice.** Once for T4 calibration on day one, once for the final run. Every other measurement is on train.
– **T12 outranks the screening score.** A component that lifts APPS and regresses the JavaScript set does not ship, because the hands-on round runs our code on their queries.
– **A6 and A7 are conditional on T3, not on enthusiasm.** The expected outcome is that both are cancelled and those days go to Workstream B. Cancelling them is the plan working, not the plan failing.
– **Nothing new starts after the 22 September freeze.**

### What done means, restated
Thirteen green tests, a tagged commit a stranger can build, a results file in a release, a video of the system working rather than of numbers, a deck whose every claim traces to a logged run, and a limitations slide that names the substitute repository if we had to use one. The tests are not overhead on top of that list — they are the only evidence we will have that any of it is true.
