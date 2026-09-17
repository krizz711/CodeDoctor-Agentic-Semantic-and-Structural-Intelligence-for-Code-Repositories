import pytest
from app.eval_harness import run_evaluation, index_corpus

def test_t0_harness_execution():
    """
    T0: The evaluation harness executes end-to-end without crashing.
    """
    try:
        results = run_evaluation("F2LLM-v2-0.6B")
        assert "ndcg@10" in results
    except Exception as e:
        pytest.fail(f"Harness execution crashed: {e}")

def test_t1_reproducibility():
    """
    T1: Running the harness twice yields exactly the same metric.
    Prevents unseeded shuffles.
    """
    res1 = run_evaluation("F2LLM-v2-0.6B", seed=42)
    res2 = run_evaluation("F2LLM-v2-0.6B", seed=42)
    
    assert res1["ndcg@10"] == res2["ndcg@10"], "T1 FAILED: Runs are non-deterministic"

def test_t2_leakage_firewall():
    """
    T2: Leakage Firewall. The system MUST raise an exception if it tries to index the test split.
    """
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        index_corpus(split="test")
        
    assert index_corpus(split="train") == True

def test_t4_sanity_floor():
    """
    T4: The reference baseline model (F2LLM-v2-0.6B) must score > 88 NDCG.
    """
    results = run_evaluation("F2LLM-v2-0.6B")
    actual_ndcg = results["ndcg@10"]
    
    assert actual_ndcg > 88.0, f"T4 FAILED: Baseline NDCG ({actual_ndcg}) is below sanity floor of 88.0"

if __name__ == "__main__":
    pytest.main(["-v", __file__])

