from app.evaluation import evaluate_queries, ndcg_at_k, reciprocal_rank


def test_retrieval_metrics():
    assert reciprocal_rank(["wrong", "right"], {"right"}) == 0.5
    assert ndcg_at_k(["right", "wrong"], {"right"}) == 1.0
    metrics = evaluate_queries([(["right", "wrong"], {"right"})])
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_10"] == 1.0
