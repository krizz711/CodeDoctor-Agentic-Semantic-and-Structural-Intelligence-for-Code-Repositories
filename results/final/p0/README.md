# Corrected candidate P0 — accepted CPU evaluation, 2026-10-01

Private Kaggle notebook `lifemeansfun/codedoctor-correction-p0`, version2,
completed the registered full official AppsRetrieval scoring run. Version1
failed at test collection before model/APPS loading; the single infrastructure
retry repaired packaging only. Both attempts are retained in
`results/local-corrections/kaggle-attempt-{1,2}/` and their full local downloads
under the ignored `build/local-correction/` directory. No final-score tuning or
repeat evaluation followed. GitHub main/tag/release remain unchanged.

## Measurements

| Metric | Historical GPU queries / frozen configuration | Corrected CPU queries / submission configuration | Difference |
| --- | ---: | ---: | ---: |
| NDCG@10 | 0.96558 | **0.96583** | +0.00025 |
| MRR@10 | 0.9585696367967284 | **0.9582509116971268** | −0.0003187250996016 |
| Recall@10 | 0.98619 | **0.98805** | +0.00186 |

These small descriptive differences do not establish statistical improvement
or regression. No significance analysis was conducted. Configuration and query
encoding precision/device differ, so this comparison does not isolate the
effect of individual experiments or their combination. Historical evidence is
the unchanged root `appsretrieval_results.json` dated2026-09-19. Prior APPS test
exposure and possible model pretraining exposure prevent a fresh unseen-accuracy
claim. EXT-lockbox was not accessed.

The P0 runner took **2,609.959222538 seconds**, with peak process-tree RSS
**9,405,206,528 bytes** (9.41 GB decimal). This is saved-vector scoring replay,
including30 live query fidelity checks; it is not full live retrieval latency.
The complete notebook, including setup/tests/cold validation, ended at about
2,848 seconds. Query encoding was performed earlier in four CPU jobs:

| Shard | Queries | Job elapsed seconds |
| --- | ---: | ---: |
| 0 / 4 | 971 | 11,100.779351872 |
| 1 / 4 | 882 | 10,147.841933629 |
| 2 / 4 | 975 | 12,613.804469893 |
| 3 / 4 | 937 | 13,987.302033533 |

These are separate job elapsed times, not total concurrent wall time, query
latency or a speedup comparison. Shard source was `823aa87dc0c6388f67c1637043c406cf743f1290`;
encoder recipe compatibility, exact task keys, disjoint membership and provenance
were verified before scoring. Per-shard environment/report checksums are in
`environment.json`; complete reports and logs are in the retained local downloads.

## Source, environment and gates

- Measured clean commit: `18c4b6cbd097437bfb8498fefe572cddb658bc42`.
- Configuration: `config/submission.yaml`, SHA-256
  `9039497b871733c5236eb6060b0cf074c3d98ebeb30664c2ce711e4882f0aaf5`.
- Model: `codefuse-ai/F2LLM-v2-1.7B`, revision
  `3766d46e7a68545ed6190c15330983f9b39ab718`, CPU FP32 query encoding,
  unchanged instruction/EOS/last-token pooling, normalized2048 dimensions.
- Dataset: `CoIR-Retrieval/apps`, revision
  `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`:3,765 queries,8,765 corpus
  records,8,754 distinct document vectors. Offline document encoding used T4
  FP16 GPU in1,301.727 seconds; that preprocessing is separate from CPU queries.
- Linux Xeon2.20 GHz,2 physical/4 logical CPUs; Python3.12.13,
  Torch2.8.0+cpu, MTEB2.21.0. Exact package freezes are retained.
- 273 focused Linux tests passed; one optional local corpus-text fixture skipped;
  one deliberate simulated-spawn warning. The real1.7B cold Python functional
  check passed on newly constructed uncached inputs and explicit versions.
- All 56 P0 and39 cold-check recorded source/config/dependency hashes match the
  candidate. Later evidence-only commits may differ in commit ID only if every
  recorded blob still matches.
- Strict pinned-task text/recipe-key coverage, vector finiteness/nonzero norms,
  dimensions, shard membership and completeness passed. There were3,765 query
  hits with zero fallback encodings and8,765 document hits with zero misses.
- 30 seeded live checks, seed20260930: minimum/median cosine1.0.
  No execution spawn failures.100,530 candidates executed and3,540 candidates
  passed examples; the latter is a candidate count, not correct-query accuracy.
- Official MTEB result caching disabled (`cache=None`). The notebook recomputed
  NDCG/MRR against pinned task qrels, checked raw prediction identities and
  exported37,650 CSV rows with exact scores and upstream tie order.

## Preserved evidence and inspection

`appsretrieval_results.json` is the authentic MTEB task result.
`AppsRetrieval_predictions.json.gz` is deterministic lossless compression of the
raw ranked predictions; decompressed SHA-256:
`e8e95f5dcb9963097f115bfbc02d9539f3f2f4734fd5df364eeb9efd10a171dc`.
The original uncompressed JSON remains in the ignored download, not replaced.
`submission_predictions.csv` contains top10/query and `submission_metrics.csv`
the reported metrics. `prediction-verification.json` records in-notebook strict
recomputation. `execution_audit.json`, `environment.json`, `cpuinfo.txt`, package
freezes, `run.log`, complete notebook log and `validation.log` retain provenance
and failures/warnings. `cold-python/report.json` records separate functional
validation and source hashes. Its fixture/snapshot source is preserved.

`test_split_audit_log.json` is this hosted run's two-entry audit.
`test-access-reconciliation.json` hashes it separately from the unchanged
historical seven-entry audit; neither replaces or conceals earlier exposure.
Local acceptance is recorded in
`results/local-corrections/kaggle-attempt-2/acceptance.json`. It verified the
recorded metric recomputation and source; no independent second qrels
recomputation or APPS rerun is claimed. Ensurepip setup failure recovered via
virtualenv; nonfatal sitecustomize warnings remain in authentic logs.

From the repository root, using the portable submission environment:

```sh
python scripts/validate_submission.py results/final/p0/appsretrieval_results.json --official-apps
python scripts/verify_measured_source.py --environment results/final/p0/environment.json --commit HEAD
```

The isolated notebook build/launch and direct full-live CPU evaluator commands
are documented in `scripts/README.md` and `docs/submission/PROTOCOL.md`.
The local acceptance helper and slide updater are retained under ignored
`build/local-correction/` for audit, not advertised as clean-clone commands.
No new evaluation is required to inspect these authentic results.

Older P1/Bonus fixture measurements remain under `results/hands-on/` with their
original source identity; the current cold check does not replace their timing
benchmark. Human disclosure/video review, unexecuted Docker/remote-CI checks
and Samsung's amendment permission remain outside this completed CPU run.
