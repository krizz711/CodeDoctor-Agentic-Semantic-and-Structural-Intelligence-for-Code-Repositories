from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    repository_path: str = Field(..., description="Absolute path to the repository to index")
    commit_hash: str = Field(default="head", description="Git commit hash to associate with this index")
    language: str = Field(default="javascript", description="Primary language of the repository")


class IndexReport(BaseModel):
    files_indexed: int
    chunks_indexed: int
    symbols_indexed: int
    edges_extracted: int
    index_path: str


class ChunkResult(BaseModel):
    file_path: str
    name: str
    type: str  # FUNCTION, CLASS, METHOD, FILE_SUMMARY
    start_line: int
    end_line: int
    snippet: str = Field(default="", description="Source code snippet")
    signature: str = Field(default="", description="Function/class signature")


class SearchResult(BaseModel):
    file_path: str
    name: str
    type: str
    score: float
    snippet: str
    start_line: int
    end_line: int
    sources: list[str] = Field(default_factory=list, description="Which retrieval methods matched (exact, bm25, tfidf)")


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResult]


class SymbolResult(BaseModel):
    name: str
    type: str
    chunk_id: str
    file_path: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str = "CALLS"
    file_path: str = ""
    line: int = 0
    confidence: float = 1.0


class CallersResponse(BaseModel):
    symbol: str
    callers: list[str]
    edges: list[GraphEdge]


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the codebase")
    limit: int = Field(default=5, description="Number of evidence chunks to retrieve")


class AskResponse(BaseModel):
    question: str
    answer: str
    evidence: list[SearchResult]
    grounded: bool = Field(default=True, description="Whether the answer is grounded in retrieved evidence")


class StatsResponse(BaseModel):
    total_files: int
    total_chunks: int
    total_symbols: int
    total_edges: int
    languages: list[str]
    index_path: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.2.0"
    index_loaded: bool = False
