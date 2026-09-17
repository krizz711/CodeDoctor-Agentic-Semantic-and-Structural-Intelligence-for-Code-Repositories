from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    AskRequest,
    AskResponse,
    CallersResponse,
    GraphEdge,
    HealthResponse,
    IndexReport,
    IndexRequest,
    SearchResponse,
    SearchResult,
    StatsResponse,
    SymbolResult,
)
from app.indexing.sqlite_index import SQLiteIndex
from app.retrieval.hybrid import HybridRetriever

router = APIRouter()

# Module-level state
_index: SQLiteIndex | None = None
_retriever: HybridRetriever | None = None


def get_index() -> SQLiteIndex:
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return _index


def get_retriever() -> HybridRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    return _retriever


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", version="0.2.0", index_loaded=_index is not None)


@router.post("/api/v1/index", response_model=IndexReport)
def index_repo(request: IndexRequest) -> IndexReport:
    global _index, _retriever

    if not os.path.exists(request.repository_path):
        raise HTTPException(status_code=400, detail=f"Repository path does not exist: {request.repository_path}")

    db_path = os.path.join(request.repository_path, "codedoctor.db")

    try:
        _index = SQLiteIndex(db_path)
        _retriever = HybridRetriever(_index)

        # We would run parsing and indexing here

        return IndexReport(files_indexed=0, chunks_indexed=0, symbols_indexed=0, edges_extracted=0, index_path=db_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize index: {e!s}") from e


@router.get("/api/v1/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query"), limit: int = Query(10, description="Max results")
) -> SearchResponse:
    retriever = get_retriever()

    try:
        docs = retriever.search(q, limit=limit)
        results = []

        for doc in docs:
            results.append(
                SearchResult(
                    file_path=doc.metadata.get("file_path", ""),
                    name=doc.metadata.get("name", "Unknown"),
                    type=doc.metadata.get("type", "UNKNOWN"),
                    score=doc.score or 0.0,
                    snippet=doc.content,
                    start_line=doc.metadata.get("start_line", 0),
                    end_line=doc.metadata.get("end_line", 0),
                    sources=doc.metadata.get("sources", []),
                )
            )

        return SearchResponse(query=q, total_results=len(results), results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e!s}") from e


@router.get("/api/v1/symbol/{name}", response_model=list[SymbolResult])
def get_symbol(name: str) -> list[SymbolResult]:
    idx = get_index()

    try:
        if not hasattr(idx, "get_symbols"):
            raise HTTPException(status_code=501, detail="Method not implemented in SQLiteIndex")

        symbols = idx.get_symbols(name)  # type: ignore
        if not symbols:
            raise HTTPException(status_code=404, detail=f"Symbol '{name}' not found")

        results = []
        for sym in symbols:
            results.append(
                SymbolResult(
                    name=sym["name"], type=sym["type"], chunk_id=sym["chunk_id"], file_path=sym.get("file_path", "")
                )
            )
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Error retrieving symbol: {e}") from e


@router.get("/api/v1/callers/{symbol}", response_model=CallersResponse)
def get_callers(symbol: str) -> CallersResponse:
    idx = get_index()

    try:
        if not hasattr(idx, "get_callers"):
            raise HTTPException(status_code=501, detail="Method not implemented in SQLiteIndex")

        edges_data = idx.get_callers(symbol)  # type: ignore

        callers = []
        edges = []

        for e in edges_data:
            callers.append(e["source"])
            edges.append(
                GraphEdge(
                    source=e["source"],
                    target=e["target"],
                    relation=e.get("relation", "CALLS"),
                    file_path=e.get("file_path", ""),
                    line=e.get("line", 0),
                    confidence=e.get("confidence", 1.0),
                )
            )

        return CallersResponse(symbol=symbol, callers=list(set(callers)), edges=edges)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Error retrieving callers: {e}") from e


@router.get("/api/v1/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    idx = get_index()

    try:
        if not hasattr(idx, "get_stats"):
            raise HTTPException(status_code=501, detail="Method not implemented in SQLiteIndex")

        s = idx.get_stats()  # type: ignore
        return StatsResponse(
            total_files=s.get("total_files", 0),
            total_chunks=s.get("total_chunks", 0),
            total_symbols=s.get("total_symbols", 0),
            total_edges=s.get("total_edges", 0),
            languages=s.get("languages", []),
            index_path=idx.db_path,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Error retrieving stats: {e}") from e


@router.post("/api/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    retriever = get_retriever()

    try:
        docs = retriever.search(request.question, limit=request.limit)
        evidence = []

        for doc in docs:
            evidence.append(
                SearchResult(
                    file_path=doc.metadata.get("file_path", ""),
                    name=doc.metadata.get("name", "Unknown"),
                    type=doc.metadata.get("type", "UNKNOWN"),
                    score=doc.score or 0.0,
                    snippet=doc.content,
                    start_line=doc.metadata.get("start_line", 0),
                    end_line=doc.metadata.get("end_line", 0),
                    sources=doc.metadata.get("sources", []),
                )
            )

        return AskResponse(
            question=request.question,
            answer="Evidence retrieved. LLM integration pending.",
            evidence=evidence,
            grounded=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {e!s}") from e


@router.get("/api/v1/graph/{symbol}")
def get_graph(symbol: str, depth: int = Query(2, description="Traversal depth")) -> dict[str, Any]:
    idx = get_index()

    try:
        if not hasattr(idx, "get_subgraph"):
            raise HTTPException(status_code=501, detail="Method not implemented in SQLiteIndex")

        subgraph = idx.get_subgraph(symbol, depth)  # type: ignore
        return {"symbol": symbol, "depth": depth, "graph": subgraph}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Error retrieving graph: {e}") from e
