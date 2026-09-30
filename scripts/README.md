# Submission commands

Run commands from the repository root. Install `requirements-submission.txt`
(and PyTorch 2.8.0 for your platform). The hosted builder creates an isolated
Linux environment and installs the correct CPU or CUDA wheel itself.

## Recorded Kaggle stages

```sh
python scripts/build_submission_kernel.py --owner lifemeansfun --stage documents --out build/kernels/documents
kaggle kernels push -p build/kernels/documents
python scripts/build_submission_kernel.py --owner lifemeansfun --stage hands-on --out build/kernels/hands-on
kaggle kernels push -p build/kernels/hands-on
```

After the document stage completes, build/push `--stage queries --shard I/4`
for I = 0, 1, 2, 3, using a separate output directory for each. Each CPU job
encodes a disjoint set of queries. After all four complete and their source,
configuration and cache manifests agree:

```sh
python scripts/build_submission_kernel.py --owner lifemeansfun --stage p0 --shards 4 --out build/kernels/p0
kaggle kernels push -p build/kernels/p0
kaggle kernels output lifemeansfun/codedoctor-submission-p0 -p build/downloads/p0
python scripts/verify_measured_source.py --environment build/downloads/p0/results/environment.json --commit HEAD
```

The builder packages committed source. Commit changes first. GPU is enabled only
for document preprocessing. CPU query stores are checkpoints, not MTEB result
caches. Final scoring uses `cache=None` and 30 seeded live query checks. Its replay
time excludes the original query encoding; report both stages separately.
Do not rerun completed test scoring to select a configuration.

## Direct reproduction

```sh
python scripts/submission_run.py --stage documents --out results/documents
python scripts/submission_run.py --stage hands-on --out results/hands-on --tier accuracy
python scripts/submission_run.py --stage p0 --out results/final --corpus-cache results/documents/documents
```

The document stage requires CUDA. On CPU-only hosts, use
`python scripts/prepare_document_cache.py --device cpu --out build/documents`,
then supply `--corpus-cache build/documents`. Without query-store inputs, P0
encodes all queries live and checkpoints them. This can take many hours.

`evaluate_mteb.py` runs the official task; `validate_submission.py` checks JSON
format; `export_submission_csv.py` exports rankings. `verify_measured_source.py`
checks recorded source bytes against a commit. Format validation does not prove
score reproduction. `make_notebook.py` generates the companion Colab notebook.

`evaluate.py scripts/sample_eval.jsonl` is a small harness check, not the official
benchmark. Historical experimental runners are under `archive/research/scripts`.
