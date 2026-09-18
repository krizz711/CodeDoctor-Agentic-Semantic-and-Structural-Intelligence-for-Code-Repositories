from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.indexing.cache import EmbeddingCache
from app.indexing.incremental import IncrementalIndexer, IncrementalStats
from app.indexing.sqlite_index import SQLiteIndex
from app.indexing.versioned import GitVersionManager, VersionedIndex
from app.retrieval.hybrid import Document, HybridRetriever
from app.retrieval.structural import StructuralEngine


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "web" / "dist"

app = FastAPI(title="CodeDoctor API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
INDEX_PATH = REPO_ROOT / "build" / "codedoctor.db"
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

sqlite_index = SQLiteIndex(str(INDEX_PATH))
cache = EmbeddingCache(str(REPO_ROOT / "build" / "cache.db"))
incremental_indexer = IncrementalIndexer(sqlite_index, cache=cache)
structural_engine = StructuralEngine(sqlite_index)
versioned_index = VersionedIndex()

# Target repository to explore (defaults to test_repos/commander or demo_repo)
DEFAULT_TARGET_REPO = REPO_ROOT / "test_repos" / "commander"
if not DEFAULT_TARGET_REPO.exists():
    DEFAULT_TARGET_REPO = REPO_ROOT / "demo_repo"


def get_active_repo() -> Path:
    return DEFAULT_TARGET_REPO if DEFAULT_TARGET_REPO.exists() else REPO_ROOT


class SearchRequest(BaseModel):
    query: str
    commit: Optional[str] = None
    limit: int = 10
    weights: Optional[dict[str, float]] = None


@app.get("/api/repo/info")
def get_repo_info() -> dict[str, Any]:
    repo = get_active_repo()
    head_sha = "uncommitted"
    branch = "main"
    try:
        head_sha = GitVersionManager.resolve_sha(repo, "HEAD")
        branch = GitVersionManager.run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass

    # Build file tree
    files = []
    for p in repo.rglob("*"):
        if p.is_file() and not any(part in {".git", "node_modules", "dist", "build"} for part in p.parts):
            if p.suffix.lower() in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".py"}:
                files.append(str(p.relative_to(repo)).replace("\\", "/"))

    files.sort()
    stats = incremental_indexer.last_stats.to_dict()

    return {
        "name": repo.name,
        "path": str(repo),
        "branch": branch,
        "head_commit": head_sha[:8] if head_sha != "uncommitted" else head_sha,
        "files_count": len(files),
        "file_tree": files,
        "stats": stats,
    }


@app.get("/api/repo/file")
def get_file_content(path: str = Query(...)) -> dict[str, Any]:
    repo = get_active_repo()
    target = (repo / path).resolve()
    if not target.exists() or not str(target).startswith(str(repo)):
        raise HTTPException(status_code=404, detail=f"File {path} not found")

    raw = target.read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    # Get chunks in file from index
    chunks = [
        dict(r)
        for r in sqlite_index.conn.execute(
            """
            SELECT c.* FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE f.path = ?
            ORDER BY c.start_byte
            """,
            (path.replace("\\", "/"),),
        ).fetchall()
    ]

    return {
        "file_path": path,
        "content": content,
        "lines": content.count("\n") + 1,
        "bytes": len(raw),
        "chunks": chunks,
    }


@app.post("/api/search")
def search_code(req: SearchRequest) -> dict[str, Any]:
    query = req.query.strip()
    if not query:
        return {"query": "", "results": [], "total": 0}

    # If pinned to a commit, use VersionedIndex
    if req.commit:
        try:
            results = versioned_index.retrieve(query, commit=req.commit, limit=req.limit)
            return {
                "query": query,
                "commit": req.commit,
                "total": len(results),
                "results": [
                    {
                        "file_path": r.document.file_path,
                        "name": r.document.name,
                        "type": r.document.type,
                        "start_line": r.document.start_line,
                        "end_line": r.document.end_line,
                        "start_byte": r.document.start_byte,
                        "end_byte": r.document.end_byte,
                        "score": round(r.score, 5),
                        "snippet": r.document.content[:400],
                        "sources": r.sources,
                        "signals": r.signals or {},
                    }
                    for r in results
                ],
            }
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # General hybrid search over current index
    raw_chunks = sqlite_index.all_chunks()
    if not raw_chunks:
        # If not indexed yet, index active repo first
        incremental_indexer.index_repository(get_active_repo(), compute_embeddings=False)
        raw_chunks = sqlite_index.all_chunks()

    docs = [
        Document(
            document_id=c["id"],
            file_path=c["file_path"],
            content=c["content"],
            name=c["name"],
            type=c["type"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            start_byte=c["start_byte"],
            end_byte=c["end_byte"],
            signature=c["signature"] or "",
        )
        for c in raw_chunks
    ]

    retriever = HybridRetriever(docs, use_dense=False)
    results = retriever.search(query, limit=req.limit, weights=req.weights)

    # Classify into evidence facets (entry point / implementation / caller / test / configuration)
    evidence_bundles: dict[str, list[dict]] = {
        "entry_point": [],
        "implementation": [],
        "caller": [],
        "test": [],
        "configuration": [],
    }

    formatted_results = []
    for r in results:
        fpath = r.document.file_path.lower()
        res_entry = {
            "document_id": r.document.document_id,
            "file_path": r.document.file_path,
            "name": r.document.name,
            "type": r.document.type,
            "start_line": r.document.start_line,
            "end_line": r.document.end_line,
            "start_byte": r.document.start_byte,
            "end_byte": r.document.end_byte,
            "score": round(r.score, 5),
            "snippet": r.document.content[:400],
            "signature": r.document.signature,
            "sources": r.sources,
            "signals": r.signals or {},
        }
        formatted_results.append(res_entry)

        # Facet categorization
        if "test" in fpath or "spec" in fpath:
            evidence_bundles["test"].append(res_entry)
        elif "config" in fpath or fpath.endswith(".json") or fpath.endswith(".yaml"):
            evidence_bundles["configuration"].append(res_entry)
        elif "index" in fpath or "main" in fpath or "cli" in fpath:
            evidence_bundles["entry_point"].append(res_entry)
        elif r.document.type in ("FUNCTION", "METHOD", "CLASS"):
            evidence_bundles["implementation"].append(res_entry)
        else:
            evidence_bundles["caller"].append(res_entry)

    # Omit empty facet categories
    active_evidence = {k: v for k, v in evidence_bundles.items() if v}

    return {
        "query": query,
        "commit": "HEAD",
        "total": len(formatted_results),
        "results": formatted_results,
        "evidence_bundles": active_evidence,
    }


@app.get("/api/structural/callers")
def get_callers(symbol: str = Query(...)) -> dict[str, Any]:
    callers = structural_engine.callers_of(symbol)
    return {"symbol": symbol, "total": len(callers), "callers": callers}


@app.get("/api/structural/sequence")
def get_tool_sequence(first: str = Query(...), second: str = Query(...)) -> dict[str, Any]:
    matches = structural_engine.sequence_query(first, second)
    return {"first_tool": first, "second_tool": second, "matches": matches}


@app.get("/api/structural/deeplink")
def get_deeplinks(keyword: str = Query("bluetooth-settings")) -> dict[str, Any]:
    matches = structural_engine.deeplink_query(keyword)
    return {"keyword": keyword, "matches": matches}


@app.get("/api/git/history")
def get_git_history(limit: int = 15) -> dict[str, Any]:
    repo = get_active_repo()
    commits = []
    try:
        fmt = "%H%x00%an%x00%at%x00%s"
        out = GitVersionManager.run_git(repo, ["log", f"-n{limit}", f"--format={fmt}"])
        for line in out.splitlines():
            if not line:
                continue
            parts = line.split("\x00")
            if len(parts) >= 4:
                commits.append(
                    {
                        "sha": parts[0],
                        "short_sha": parts[0][:8],
                        "author": parts[1],
                        "timestamp": int(parts[2]) if parts[2].isdigit() else 0,
                        "message": parts[3],
                    }
                )
    except Exception:
        pass
    return {"commits": commits}


@app.get("/api/git/evolution")
def get_evolution(query: str = Query(...), file: Optional[str] = None) -> dict[str, Any]:
    repo = get_active_repo()
    results = versioned_index.find_when_changed(query, repo_path=repo, file_path=file, limit=5)
    return {"query": query, "file": file, "results": results}


@app.get("/api/index/status")
def get_index_status() -> dict[str, Any]:
    stats = incremental_indexer.last_stats.to_dict()
    stats["total_indexed_chunks"] = sqlite_index.count_chunks()
    stats["total_indexed_files"] = sqlite_index.count_files()
    stats["cached_embeddings"] = cache.count()
    return stats


@app.post("/api/index/reindex")
def trigger_reindex(force_rebuild: bool = False) -> dict[str, Any]:
    repo = get_active_repo()
    stats = incremental_indexer.index_repository(
        repo, compute_embeddings=False, force_rebuild=force_rebuild
    )
    # Also extract structural edges
    for f_row in sqlite_index.conn.execute("SELECT path FROM files").fetchall():
        path = f_row["path"]
        abs_p = (repo / path).resolve()
        if abs_p.exists():
            try:
                structural_engine.index_edges_for_file(path, abs_p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {"status": "success", "stats": stats.to_dict()}


# Serve static web files if present
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
