from pydantic import BaseModel, Field
from typing import List, Optional

class FileCreate(BaseModel):
    repo_name: str
    file_path: str
    commit_hash: str
    content: str

class ChunkCreate(BaseModel):
    file_id: int
    start_byte: int
    end_byte: int
    semantic_hash: str
    ast_type: str
    content: str

class SearchResult(BaseModel):
    chunk_id: int
    file_path: str
    ast_type: str
    content: str
    score: float = Field(..., description="The combined hybrid retrieval score (Dense + BM25)")

class SearchResponse(BaseModel):
    query: str
    latency_ms: float = Field(..., description="Query execution time in milliseconds")
    results: List[SearchResult]
