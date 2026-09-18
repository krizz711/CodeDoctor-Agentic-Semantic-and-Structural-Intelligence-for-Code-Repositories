"""CodeDoctor pre/post-processing retrieval pipeline.

Implements the five improvement axes named in the Theme 01 brief:

1. Query categorization          -> :mod:`app.pipeline.query_analysis`
2. Query pre-processing          -> :mod:`app.pipeline.query_analysis`
3. Code-snippet categorization   -> :mod:`app.pipeline.code_analysis`
4. Pre/post processing of code   -> :mod:`app.pipeline.code_analysis`
5. Multiple retrieval passes     -> :mod:`app.pipeline.reranker`

The dense encoder itself is frozen; every module in this package is a
deterministic, CPU-only transform around it.
"""

from app.pipeline.query_analysis import QueryFacts, QueryFamily, analyze_query
from app.pipeline.code_analysis import CodeFacts, CodeFamily, analyze_code

__all__ = [
    "QueryFacts",
    "QueryFamily",
    "analyze_query",
    "CodeFacts",
    "CodeFamily",
    "analyze_code",
]
