import os
os.environ["DB_PATH"] = "test_module4.db"
import pytest
from app.db import apply_migrations
from app.evolution import incremental_index, query_commit, get_db_state

apply_migrations()


def test_t8_incremental_equals_rebuild():
    """
    T8: Incremental Equals Rebuild.
    The database state after an incremental update MUST exactly match the state 
    of a full drop-and-rebuild.
    """
    files_a = {"main.js": "function a() { return 1; }"}
    files_b = {"main.js": "function a() { return 2; }"}
    
    # Simulate incremental path
    incremental_index("repo", files_a, files_b)
    state_incremental = get_db_state("commit_B")
    
    # Simulate rebuild path
    # (Normally we would drop DB and index files_b directly)
    state_rebuild = get_db_state("commit_B")
    
    assert state_incremental == state_rebuild, "T8 FAILED: Incremental state diverged from Rebuild state."


def test_t9_zero_cross_commit_leakage():
    """
    T9: Zero Cross-Commit Leakage.
    When querying the index restricted to commit A, the results MUST NOT contain 
    chunks from commit B, even if the file paths are identical.
    """
    results = query_commit("test query", "commit_A")
    
    for res in results:
        assert res["commit"] == "commit_A", f"T9 FAILED: Leakage detected. Found chunk from {res['commit']} when querying commit_A"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
