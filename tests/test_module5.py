import os
os.environ["DB_PATH"] = "test_module5.db"
import pytest
from app.db import apply_migrations
from app.main import run_pipeline

apply_migrations()

def test_t12_generalisation():
    """
    T12: Generalisation Probe.
    The system MUST NOT be hardcoded to evaluate or index only one specific repository.
    We verify this by running the pipeline against two distinct simulated repositories
    and ensuring the inputs route dynamically.
    """
    res1 = run_pipeline("React-Core", "useEffect hook")
    res2 = run_pipeline("Vue-Core", "setup function")
    
    assert res1["repo"] == "React-Core", "T12 FAILED: Hardcoded repo name detected"
    assert res2["repo"] == "Vue-Core", "T12 FAILED: Hardcoded repo name detected"

def test_t13_end_to_end_stability():
    """
    T13: E2E Pipeline Stability.
    The final stitched pipeline must not crash under valid inputs and must 
    return a structured response.
    """
    from app.indexer import index_file
    index_file("TestRepo", "test.js", "main", "function search() { return query; }")
    response = run_pipeline("TestRepo", "search query")
    
    assert response["status"] == "success", "T13 FAILED: Pipeline crashed"
    assert isinstance(response["results"], list), "T13 FAILED: Pipeline returned invalid structure"
    if response["results"]:
        assert "content" in response["results"][0], "T13 FAILED: Missing content in response"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
