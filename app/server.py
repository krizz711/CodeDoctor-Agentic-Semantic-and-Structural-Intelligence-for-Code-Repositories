from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

from app.indexing.cache import EmbeddingCache, recipe_cache_key, stack_cached
from app.indexing.incremental import IncrementalIndexer
from app.indexing.sqlite_index import SQLiteIndex
from app.indexing.versioned import GitVersionManager, VersionedIndex
from app.pipeline.query_analysis import analyze_query
from app.retrieval.hybrid import Document, HybridRetriever
from app.retrieval.structural import StructuralEngine
from app.workbench import get_workbench


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


def _cached_document_embeddings(documents: list[Document]) -> Optional[np.ndarray]:
    """Vectors for every chunk, or ``None`` if the index is not fully embedded.

    The key is the encoder *input*, matching how IncrementalIndexer stored it,
    so a renamed or moved file still hits.  Returning ``None`` on a partial
    index is deliberate: a dense pass over some of the corpus would silently
    rank the embedded chunks above the rest.
    """
    vectors: list[np.ndarray] = []
    encoder = incremental_indexer._get_encoder()  # the recipe the vectors were stored under
    for document in documents:
        encoder_input = (
            f"{document.signature}\n{document.content}"
            if document.signature
            else document.content
        )
        vector = cache.get(recipe_cache_key(encoder, encoder_input), real_only=True)
        if vector is None:
            return None
        vectors.append(vector)
    if not vectors:
        return None
    return stack_cached(vectors)


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

    # Dense scoring uses the vectors the indexer already cached, keyed on the
    # encoder input. Re-encoding the corpus per query would cost minutes; a
    # cache lookup costs microseconds. If any chunk is missing a vector the
    # dense pass is skipped entirely rather than run on a partial index.
    document_embeddings = _cached_document_embeddings(docs)
    retriever = HybridRetriever(docs, document_embeddings=document_embeddings)
    results = retriever.search(query, limit=req.limit, weights=req.weights)
    query_facts = analyze_query(query)

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
        "query_analysis": {
            "family": query_facts.family.value,
            "confidence": round(query_facts.confidence, 3),
            "verified": query_facts.is_verified(),
            "identifiers": sorted(query_facts.identifiers)[:12],
        },
        "retrieval": {
            "dense": document_embeddings is not None,
            "lexical": True,
            "indexed_chunks": len(docs),
        },
    }


@app.get("/api/structural/callers")
def get_callers(symbol: str = Query(...)) -> dict[str, Any]:
    callers = structural_engine.callers_of(symbol)
    return {"symbol": symbol, "total": len(callers), "callers": callers}


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


@app.get("/api/index/status")
def get_index_status() -> dict[str, Any]:
    stats = incremental_indexer.last_stats.to_dict()
    stats["total_indexed_chunks"] = sqlite_index.count_chunks()
    stats["total_indexed_files"] = sqlite_index.count_files()
    stats["cached_embeddings"] = cache.count()
    return stats


@app.post("/api/index/reindex")
def trigger_reindex(force_rebuild: bool = False, compute_embeddings: bool = False) -> dict[str, Any]:
    """Re-index the explorer's repository; only changed files are re-read.

    With ``compute_embeddings=false`` (the default: no model load) edited chunks
    get no vector, and the explorer's search then answers lexically for the
    whole corpus rather than rank a partly embedded one -- the response says so.
    ``compute_embeddings=true`` encodes exactly the chunks not in the cache.
    The hands-on demo surface is the workbench (``/api/workbench/build`` with
    ``rebuild=true``), which always encodes what changed.
    """
    repo = get_active_repo()
    stats = incremental_indexer.index_repository(
        repo, compute_embeddings=compute_embeddings, force_rebuild=force_rebuild
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
    return {
        "status": "success",
        "stats": stats.to_dict(),
        "embeddings_computed": compute_embeddings,
        "dense_search": "available" if compute_embeddings else "lexical only until re-indexed with compute_embeddings=true",
    }


# ---------------------------------------------------------------------------
# Workbench: the hands-on demo surface.
#
# Kept apart from the endpoints above on purpose.  Those serve the file
# explorer and answer against whatever the SQLite index currently holds; these
# serve a *versioned* corpus and report what every retrieval pass did, which is
# what the Theme 01 hands-on evaluation actually drives.
# ---------------------------------------------------------------------------

workbench = get_workbench(cache=cache)


class WorkbenchQuery(BaseModel):
    target: str = "self"
    query: str
    version: Optional[str] = None
    limit: int = 10
    verify: bool = True


class WorkbenchCompare(BaseModel):
    target: str = "self"
    query: str
    left: str
    right: str
    limit: int = 10


@app.get("/api/workbench/targets")
def workbench_targets() -> dict[str, Any]:
    return {"targets": workbench.targets(), "state": workbench.state()}


@app.post("/api/workbench/open")
def workbench_open(path: str) -> dict[str, Any]:
    """Register a local directory as a target, then build it separately.

    Opening and building are two calls because opening is instant and building
    is not: the UI can show the folder in the picker, with its file count and
    detected language, before anyone commits to indexing it.
    """
    try:
        return workbench.open_folder(path)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/api/workbench/build")
def workbench_build(
    target: str = "self", versions: int = 4, dense: bool = True, rebuild: bool = False
) -> dict[str, Any]:
    """Build a target; ``rebuild=true`` re-reads an already-built one (P1)."""
    try:
        return workbench.start_build(target, versions=versions, dense=dense, rebuild=rebuild)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/api/workbench/status")
def workbench_status() -> dict[str, Any]:
    return workbench.state()


@app.get("/api/workbench/versions")
def workbench_versions(target: str = "self") -> dict[str, Any]:
    """Versions of a target, and an empty list when it is not indexed yet.

    Not an error: the page asks for this before anything is built, and the
    preset questions are worth showing while the encoder warms up.
    """
    ready = workbench.is_ready(target)
    return {
        "target": target,
        "ready": ready,
        "versions": workbench.versions(target) if ready else [],
        "presets": workbench.presets(target),
    }


@app.post("/api/workbench/search")
def workbench_search(request: WorkbenchQuery) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    try:
        return workbench.search(
            request.target,
            request.query.strip(),
            version=request.version,
            limit=request.limit,
            verify=request.verify,
        )
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/api/workbench/compare")
def workbench_compare(request: WorkbenchCompare) -> dict[str, Any]:
    try:
        return workbench.compare(
            request.target, request.query.strip(), request.left, request.right, limit=request.limit
        )
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/api/workbench/across")
def workbench_across(request: WorkbenchQuery) -> dict[str, Any]:
    try:
        return workbench.across(request.target, request.query.strip(), limit=request.limit)
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error))


FINAL_RESULTS = REPO_ROOT / "results" / "final"


@app.get("/api/workbench/benchmark")
def workbench_benchmark() -> dict[str, Any]:
    """The screening result, read from the artifact rather than typed.

    Every number on the benchmark panel comes out of a result file, so the panel cannot drift
    from it. ``results/final`` (the measured final submission) is preferred when it exists;
    otherwise the first release's artifact at the repository root is shown, labelled as
    historical so the two are never confused.
    """
    payload: dict[str, Any] = {"available": False}
    final = FINAL_RESULTS / "appsretrieval_results.json"
    if final.exists():
        results_path, audit_path = final, FINAL_RESULTS / "execution_audit.json"
        payload.update(kind="final", label="final submission (results/final)")
    else:
        results_path = REPO_ROOT / "appsretrieval_results.json"
        audit_path = REPO_ROOT / "execution_audit.json"
        payload.update(kind="historical", label="first release (historical, frozen configuration)")
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        scores = results.get("scores", {}).get("test", [{}])[0]
        payload.update(
            available=True,
            source=results_path.name,
            task=results.get("task_name"),
            ndcg_at_10=scores.get("ndcg_at_10"),
            mrr_at_10=scores.get("mrr_at_10"),
            recall_at_10=scores.get("recall_at_10"),
            main_score=scores.get("main_score"),
            evaluation_time=results.get("evaluation_time"),
            mteb_version=results.get("mteb_version"),
        )
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            payload["execution"] = {
                "executed": audit.get("candidates_executed"),
                "verified": audit.get("candidates_verified"),
                "queries_with_examples": audit.get("queries_with_examples"),
                "seconds": audit.get("seconds"),
            }
        except Exception:
            pass
    return payload


@app.middleware("http")
async def no_stale_html(request, call_next):
    """Make the browser revalidate the UI on every load.

    The whole interface is one hand-written HTML file with its CSS and JS
    inline, which means a cached copy is a cached *application*. A browser that
    holds one will keep showing an old build against a new backend -- the
    symptom is a page whose colours and progress text belong to a previous
    version while the API responses are current. That is a confusing five
    minutes for anyone, and an expensive one in front of a judge.

    Assets under /assets are content-addressed by name and safe to cache.
    """
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".html", "/")) or path == "":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


# Serve static web files if present
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
