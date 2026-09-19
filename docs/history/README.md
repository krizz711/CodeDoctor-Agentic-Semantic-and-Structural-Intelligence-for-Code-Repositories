# Superseded work orders

These are the internal work orders and implementation reports from earlier
build phases, kept for provenance. **They are not current**, and several of
their conclusions were overturned by measurement:

| claim in these documents | what was actually measured |
|---|---|
| `codefuse-ai/F2LLM-v2-0.6B` is the production model | it is the *interactive* tier; the submission runs F2LLM-v2-1.7B |
| the shipped score is 0.87603 | that run truncated documents at 512 tokens under last-token pooling; the correct recipe scores 0.90427 for the same model |
| "recall ceiling mathematically confirms reranker cancellation" | the pool ceiling is 99.30% at depth 50, and reranking by *execution* is worth +0.0440 NDCG@10 on held-out DEV |
| a hand-built feature reranker is the second pass | it failed its own paired-bootstrap gate (+0.0048, CI spans zero) and ships as the identity |

For what the system does now, read
[`../RETRIEVAL-PIPELINE.md`](../RETRIEVAL-PIPELINE.md) and the repository
[`README.md`](../../README.md).
