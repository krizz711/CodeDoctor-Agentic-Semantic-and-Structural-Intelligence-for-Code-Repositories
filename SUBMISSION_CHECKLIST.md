# Submission Checklist

Required deliverables per the Theme 1 guidelines: a results JSON (as a GitHub Release
artifact), a PPT, and a repo with working instructions.

## 1. Encoder is real — done in code

- [x] `PrePostPipelineEncoder` implemented in `app/retrieval/mteb_encoder.py`
- [x] `scripts/evaluate_mteb.py` instantiates it instead of a bare `SentenceTransformer`
- [x] `batch_size=64` (the brief specifies 64; the old script used 32)
- [x] Task is `mteb.get_task("AppsRetrieval")`
- [x] No random / constant / placeholder vector path anywhere
- [x] Tests pin determinism and ranking behaviour (`tests/test_mteb_encoder.py`)

**Do not** port the `manish-workspace` encoder. It returns `np.random.rand(...)`,
which yields a structurally valid JSON with meaningless scores.

## 2. Verify before the long run — you must do this

```bash
pip install -r requirements-eval.txt
python -m pytest tests/test_mteb_encoder.py -q      # 14 tests, all should pass
python scripts/verify_encoder.py                    # offline structural check
python scripts/verify_encoder.py --real-model       # downloads the dense backbone
```

`verify_encoder.py --real-model` is the meaningful check: it confirms the real
backbone loads on CPU and ranks relevant snippets above irrelevant ones.

## 3. Generate the results JSON — you must do this

Requires network access to HuggingFace (model weights + AppsRetrieval dataset).

```bash
python scripts/evaluate_mteb.py --output appsretrieval_results.json
```

The script prints NDCG@10 and MRR and **exits non-zero if scores are at
random-retrieval level**, so a broken encoder cannot silently produce a
submission file.

Optional dense-only ablation, useful for the PPT:

```bash
python scripts/evaluate_mteb.py --lexical-weight 0.0 --output ablation_dense_only.json
```

- [ ] `appsretrieval_results.json` generated
- [ ] NDCG@10 and MRR recorded and look plausible (not near zero)
- [ ] Ablation run recorded for the PPT

## 4. Commit and release — you must do this

- [ ] `appsretrieval_results.json` committed to `main`
- [ ] Release tag created on `main`
- [ ] JSON **attached as a release artifact** (not just committed — the guidelines
      require the artifact)
- [ ] README instructions verified by following them on a clean checkout

## 5. Other deliverables

- [ ] PPT covers: preprocessing steps, embedding model, the dense+lexical fusion
      design, and retrieval results for a hard query
- [ ] Demo video shows live query responses and timing, not just metrics

## Still open (not addressed by this fix)

These were flagged in the earlier review and remain outstanding:

- `EmbeddingRetriever` in `app/retrieval/embeddings.py` is still unwired from the
  CLI/API path (the MTEB encoder has its own dense backbone).
- `QUERY_EXPANSIONS` still has only 4 hardcoded entries copied from the problem
  statement example.
- No cross-encoder reranking stage.
- `_bm25` / `_tfidf` still rebuild counters per query; `bm25s` is in requirements
  but unused.
- P1 versioning still stores full snapshot copies; `content_hash` is computed but
  unused for deduplication.
- Bonus (cross-version retrieval) is still blocked by design in `VersionedIndex`.
