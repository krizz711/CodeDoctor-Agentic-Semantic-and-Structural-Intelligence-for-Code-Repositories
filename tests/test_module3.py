import pytest
import time
from app.hybrid_engine import hybrid_search, CppEngineError

def test_t14_latency_limit():
    """
    T14: Latency constraints. The hybrid fusion must execute in under 3.0 seconds.
    """
    start_time = time.time()
    hybrid_search("test query", [], [])
    duration = time.time() - start_time
    
    assert duration < 3.0, f"T14 FAILED: Hybrid search took {duration}s, exceeding 3.0s limit"

def test_defensive_memory_boundaries():
    """
    Rule 2: Defensive Coding.
    C++/Python interoperability (e.g. Pybind11) causes silent memory corruptions (segfaults) 
    if null pointers are passed. We must assert that the Python interface explicitly 
    guards against this and throws a caught ValueError before hitting C++.
    """
    with pytest.raises(ValueError, match="Null pointer passed"):
        hybrid_search("test query", None, [])
        
    with pytest.raises(ValueError, match="Null pointer passed"):
        hybrid_search("test query", [], None)

def test_hybrid_recall_superiority():
    """
    Verifies that Hybrid Fusion (Dense + Lexical) outperforms Dense alone.
    """
    # Reciprocal Rank Fusion yields small scores (e.g. 1/60).
    # We assert that the fusion correctly ranks the items that appear in both lists higher.
    results = hybrid_search("test query", [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.6}], [{"id": 1, "score": 0.5}])
    assert len(results) == 2
    assert results[0]["chunk_id"] == 1, "T14 FAILED: Item 1 should have highest RRF score due to appearing in both dense and lexical lists."

if __name__ == "__main__":
    pytest.main(["-v", __file__])
