# Evaluation scripts

`evaluate.py` is the dependency-light local harness. It accepts JSONL rows containing `query`, `documents`, and `relevant_ids`, and writes MRR/NDCG metrics plus ranked predictions.

`evaluate_mteb.py` is the official AppsRetrieval entry point. It uses the MTEB library and a CPU SentenceTransformers model, then writes the task result in the JSON format expected for a GitHub Release artifact.

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_mteb.py --output appsretrieval_results.json
```

The official evaluator downloads the Apps dataset and model at runtime. It is kept separate from the default CPU-only development dependencies.
