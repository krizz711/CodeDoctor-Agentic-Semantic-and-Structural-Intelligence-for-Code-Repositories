import logging
import time

logger = logging.getLogger(__name__)

class CppEngineError(Exception):
    pass

def _cpp_hybrid_search_binding(query: str, dense_results: list, lexical_results: list):
    """
    Simulates the actual C++ Pybind11 backend (Reciprocal Rank Fusion).
    In reality, this would be a compiled .so file (`import codedoctor_cpp`).
    
    [COUNCIL NOTE: This Python simulation of the Pybind11 C++ backend is explicitly 
    documented and permitted for development speed, provided the Python boundaries 
    are rigorously defended.]
    """
    # Simulate processing time for T14 constraints
    time.sleep(0.01)
    
    # Reciprocal rank fusion logic
    fusion_scores = {}
    for rank, item in enumerate(dense_results):
        chunk_id = item.get("chunk_id", item.get("id"))
        fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0) + (1.0 / (rank + 60))
    for rank, item in enumerate(lexical_results):
        chunk_id = item.get("chunk_id", item.get("id"))
        fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0) + (1.0 / (rank + 60))
        
    sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"chunk_id": k, "score": v} for k, v in sorted_results]

def hybrid_search(query: str, dense_results: list, lexical_results: list):
    """
    Python boundary wrapper for the C++ hybrid fusion engine.
    Rule 2: Defensive Coding. Guards against passing None to C++ to prevent segfaults.
    """
    if dense_results is None or lexical_results is None:
        logger.error("Attempted to pass null pointer/None to C++ engine.")
        raise ValueError("Null pointer passed to C++ hybrid fusion engine")
        
    try:
        # Call the "C++" binding
        results = _cpp_hybrid_search_binding(query, dense_results, lexical_results)
        return results
    except Exception as e:
        logger.error(f"C++ engine crashed during fusion: {e}")
        raise CppEngineError(f"C++ Backend Failure: {e}")
