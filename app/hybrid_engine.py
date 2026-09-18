import logging
import time

logger = logging.getLogger(__name__)

class CppEngineError(Exception):
    pass

import codedoctor_cpp

def _cpp_hybrid_search_binding(query: str, dense_results: list, lexical_results: list):
    """
    Calls the actual C++ Pybind11 backend (Reciprocal Rank Fusion).
    """
    # Call the compiled C++ extension
    return codedoctor_cpp.hybrid_search(query, dense_results, lexical_results)

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
