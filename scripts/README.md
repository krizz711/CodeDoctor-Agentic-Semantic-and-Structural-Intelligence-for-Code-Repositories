# Submission and development commands

Run from the repository root with Python 3.11 or 3.12. For Linux CPU, install
`torch==2.8.0` from `https://download.pytorch.org/whl/cpu`, then
`pip install -r requirements-submission.txt`. On macOS use the platform wheel
from PyPI; CUDA preprocessing uses the appropriate PyTorch 2.8.0 CUDA wheel.
`requirements-eval.txt` and singular `requirement.txt` resolve the same recipe.
Base service dependencies keep minimum versions; each measured run records its
actual transitive versions. A Linux `torch==2.8.0+cpu` freeze is evidence and
must not become the automatically selected cross-platform requirements lock.

## Routine checks

```sh
python -m pytest -q -rs -m "not model and not data"
python -m ruff check app scripts tests --select F401,F811
python -m compileall -q app scripts tests
make -B
make -B test
python scripts/evaluate.py scripts/sample_eval.jsonl --output build/sample_results.json
python scripts/validate_submission.py build/sample_results.json
```

Install `ruff` separately for development. Registry tests need the evaluation
stack. Model/data tests need their corresponding artifacts; report actual skips
and deselections. Sample scores and JSON format validation are not P0 evidence.

## Direct measurement commands

These commands describe the reporting procedure. APPS test scoring requires the
explicit team decision in `CLAUDE.md`; a setup or smoke test is not permission
for that run. The frozen recipe reports once and does not select a configuration.

```sh
python scripts/prepare_document_cache.py --device cpu --out build/documents
python scripts/submission_run.py --stage hands-on --out results/hands-on-new --tier accuracy
python scripts/submission_run.py --stage queries --shard 0/4 --out results/query-0 --corpus-cache build/documents
python scripts/submission_run.py --stage queries --shard 1/4 --out results/query-1 --corpus-cache build/documents
python scripts/submission_run.py --stage queries --shard 2/4 --out results/query-2 --corpus-cache build/documents
python scripts/submission_run.py --stage queries --shard 3/4 --out results/query-3 --corpus-cache build/documents
python scripts/submission_run.py --stage p0 --out results/final --corpus-cache build/documents --query-store-in results/query-0 --query-store-in results/query-1 --query-store-in results/query-2 --query-store-in results/query-3
```

The first command builds document vectors on CPU. On a CUDA host, replace
`--device cpu` with `--device cuda` for permitted offline preprocessing.
`submission_run.py --stage documents` is the hosted CUDA document stage;
it is not suitable for a CPU-only machine.

The staged P0 wrapper requires all completed query shards and checks provenance
before scoring. Run timing stages without competing heavy work. CPU query stores
are resumable checkpoints. They must record the matching
recipe, source and corpus and cannot contain GPU-generated query vectors for
the CPU operating point. MTEB result caching remains disabled (`cache=None`).
Scoring replay wall time excludes the original encoding; report both separately.
Live verification of checkpoint samples checks parity, not unseen generalization.

A judge can run `evaluate_mteb.py` directly without query-store arguments to encode
all queries live. Supply `--device cpu --config config/submission.yaml`, the complete
`--corpus-cache build/documents/corpus.sqlite` and matching
`--corpus-manifest build/documents/manifest.json --require-complete-corpus-cache`.
That path can take many hours; preserve equivalent environment and timing evidence.

`evaluate_mteb.py` invokes the official `AppsRetrieval` task.
`validate_submission.py` checks result format.
`export_submission_csv.py` exports the ranked responses requested by the brief:

```sh
python scripts/export_submission_csv.py --predictions results/final/predictions --results results/final/appsretrieval_results.json --output-dir results/final
python scripts/validate_submission.py results/final/appsretrieval_results.json
```

Keep the JSON, response CSV, metrics CSV, execution audit, environment manifest,
measured dependency freeze and raw log from the same reporting run.
`verify_measured_source.py --environment PATH --commit COMMIT` compares measured
source hashes with a commit. Format validation alone cannot reproduce a score.
The exact source and gating procedure is in `docs/submission/PROTOCOL.md`.

## Hosted builder

`build_submission_kernel.py` packages committed Git blobs and writes local Kaggle
kernel files. Preparing files does not upload or run them. Choose the account and
output directory explicitly, for example:

```sh
python scripts/build_submission_kernel.py --owner ACCOUNT --stage documents --out build/kernels/documents
python scripts/build_submission_kernel.py --owner ACCOUNT --stage hands-on --out build/kernels/hands-on
python scripts/build_submission_kernel.py --owner ACCOUNT --stage queries --shard 0/4 --out build/kernels/queries-0
python scripts/build_submission_kernel.py --owner ACCOUNT --stage p0 --shards 4 --out build/kernels/p0
```

Query stages use four disjoint CPU shards when that staged procedure is selected.
The P0 builder is prepared after matching document/query stage outputs are available.
Commit source before packaging a reporting run; `--allow-dirty` is for local smoke
work and still packages committed source. Uploaded runs require separate authorized
execution. Never re-run completed test scoring to choose a configuration.
`make_notebook.py` generates a companion notebook; it is not another implementation.
Historical experimental runners remain under `archive/research/scripts`.

## AI disclosure form

The generator preserves the original official DOCX package and fields rather than
rebuilding it. Restore the unchanged source template locally, then generate:

```sh
mkdir -p build/disclosure
git show f89c969:LangAI3.0_AI_Disclosure.docx > build/disclosure/template.docx
python scripts/fill_ai_disclosure.py --template build/disclosure/template.docx --output LangAI3.0_AI_Disclosure.docx --overwrite
```

The generator uses Python's standard library and accepts portable paths. Historical
request descriptions are labeled summaries, not quoted prompts. Exact historical
model versions, feature attribution and human modifications need team review.
Ethical confirmations, representative details, submission date and signature are
left for the representative; generating a form does not attest or sign it.
Render the generated DOCX and inspect every page before submission.
