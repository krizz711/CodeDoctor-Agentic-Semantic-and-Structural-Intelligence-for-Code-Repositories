from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Optional, Sequence

from app.indexing.sqlite_index import SQLiteIndex
from app.parser.chunker import chunk_source
from app.retrieval.hybrid import Document, HybridRetriever, RankedDocument


@dataclass(frozen=True)
class VersionedDocument:
    commit: str
    document: Document
    content_hash: str

    @classmethod
    def create(cls, commit: str, document: Document) -> "VersionedDocument":
        digest = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        return cls(commit, document, digest)


@dataclass(frozen=True)
class CrossVersionHit:
    """One piece of code, and every version it appears in.

    ``variants`` counts how many *distinct* contents share this identity: 1
    means the snippet is byte-identical everywhere it appears, higher means it
    genuinely changed and the highest-scoring form is the one reported.
    """

    document: Document
    score: float
    versions: list[str]
    content_hash: str
    variants: int
    sources: tuple[str, ...] = ()

    @property
    def unchanged_across_versions(self) -> bool:
        return self.variants == 1 and len(self.versions) > 1


@dataclass
class CommitMetadata:
    sha: str
    parents: list[str]
    message: str
    author: str
    timestamp: int
    files_changed: list[str]


@dataclass
class DiffHunk:
    commit: str
    file_path: str
    header: str
    content: str
    score: float = 0.0


class GitVersionManager:
    """Helper to read actual git history and diffs via git plumbing."""

    @staticmethod
    def run_git(repo_path: Path, args: list[str]) -> str:
        res = subprocess.run(
            ["git", "-C", str(repo_path)] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()

    @classmethod
    def resolve_sha(cls, repo_path: Path, ref: str = "HEAD") -> str:
        return cls.run_git(repo_path, ["rev-parse", ref])

    @classmethod
    def list_tree(cls, repo_path: Path, commit: str) -> list[str]:
        output = cls.run_git(repo_path, ["ls-tree", "-r", "--name-only", commit])
        return [line.strip() for line in output.splitlines() if line.strip()]

    @classmethod
    def cat_file(cls, repo_path: Path, commit: str, rel_path: str) -> str:
        return cls.run_git(repo_path, ["show", f"{commit}:{rel_path}"])

    @classmethod
    def get_commit_info(cls, repo_path: Path, commit: str) -> CommitMetadata:
        fmt = "%H%x00%P%x00%an%x00%at%x00%s"
        out = cls.run_git(repo_path, ["show", "-s", f"--format={fmt}", commit])
        parts = out.split("\x00")
        sha = parts[0]
        parents = parts[1].split() if parts[1] else []
        author = parts[2]
        timestamp = int(parts[3]) if parts[3].isdigit() else 0
        message = parts[4]

        # Changed files
        files_out = cls.run_git(repo_path, ["diff-tree", "--no-commit-id", "--name-only", "-r", commit])
        files_changed = [f.strip() for f in files_out.splitlines() if f.strip()]

        return CommitMetadata(
            sha=sha,
            parents=parents,
            message=message,
            author=author,
            timestamp=timestamp,
            files_changed=files_changed,
        )

    @classmethod
    def get_diff_hunks(cls, repo_path: Path, commit: str, file_path: Optional[str] = None) -> list[DiffHunk]:
        args = ["show", "--unified=3", commit]
        if file_path:
            args.extend(["--", file_path])
        diff_text = cls.run_git(repo_path, args)

        hunks: list[DiffHunk] = []
        current_file = file_path or ""
        current_header = ""
        current_lines: list[str] = []

        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("@@"):
                if current_lines:
                    hunks.append(
                        DiffHunk(
                            commit=commit,
                            file_path=current_file,
                            header=current_header,
                            content="\n".join(current_lines),
                        )
                    )
                    current_lines = []
                current_header = line
            elif current_header:
                current_lines.append(line)

        if current_lines:
            hunks.append(
                DiffHunk(
                    commit=commit,
                    file_path=current_file,
                    header=current_header,
                    content="\n".join(current_lines),
                )
            )
        return hunks


class VersionedIndex:
    """
    Real git-backed or snapshot-backed versioned index.
    Strict membership decides eligibility:
    A query pinned at commit K sees that commit's chunks and nothing else.
    Unknown commits fail explicitly (never silently falls back to HEAD).
    """

    def __init__(self, encoder: Any = None) -> None:
        self._snapshots: dict[str, list[VersionedDocument]] = {}
        self._commit_metadata: dict[str, CommitMetadata] = {}
        self._embeddings: dict[str, Any] = {}
        self._encoder = encoder

    def add_snapshot(
        self,
        commit: str,
        documents: Iterable[Document],
        embeddings: Any = None,
    ) -> None:
        """Index one version.

        ``embeddings`` is the dense matrix for these documents, one row each.
        Supplying it turns the dense pass on for this snapshot; omitting it
        leaves the snapshot lexical-only, which is a real difference in
        retrieval quality and is reported by :meth:`uses_dense`.
        """
        items = [VersionedDocument.create(commit, doc) for doc in documents]
        self._snapshots[commit] = items
        if embeddings is not None:
            if len(embeddings) != len(items):
                raise ValueError(
                    f"{commit}: {len(embeddings)} embedding rows for {len(items)} documents"
                )
            self._embeddings[commit] = embeddings

    def uses_dense(self, commit: str | None = None) -> bool:
        """Whether retrieval on this index runs the dense pass or lexical only."""
        if commit is not None:
            return commit in self._embeddings
        return bool(self._snapshots) and all(c in self._embeddings for c in self._snapshots)

    def index_git_commit(self, repo_path: str | Path, commit: str) -> None:
        repo = Path(repo_path).resolve()
        sha = GitVersionManager.resolve_sha(repo, commit)
        meta = GitVersionManager.get_commit_info(repo, sha)
        self._commit_metadata[sha] = meta

        files = GitVersionManager.list_tree(repo, sha)
        docs: list[Document] = []
        for rel in files:
            ext = Path(rel).suffix.lower()
            if ext not in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".py"}:
                continue
            try:
                source = GitVersionManager.cat_file(repo, sha, rel)
            except Exception:
                continue

            chunks = chunk_source(source, rel)
            for chunk in chunks:
                docs.append(
                    Document(
                        document_id=chunk.chunk_id,
                        file_path=chunk.file_path,
                        content=chunk.content,
                        name=chunk.name,
                        type=chunk.type,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        start_byte=chunk.start_byte,
                        end_byte=chunk.end_byte,
                        signature=chunk.signature,
                    )
                )

        self.add_snapshot(sha, docs)
        # Also index under short ref if different
        if commit != sha:
            self._snapshots[commit] = self._snapshots[sha]

    def commits(self) -> list[str]:
        return sorted(self._snapshots.keys())

    def retrieve(self, query: str, commit: str, limit: int = 10) -> list[RankedDocument]:
        """
        Retrieves results strictly from the specified commit snapshot.
        Raises KeyError if the snapshot is not indexed.
        """
        if commit not in self._snapshots:
            raise KeyError(f"Snapshot {commit} is not indexed")

        docs = [item.document for item in self._snapshots[commit]]
        if not docs:
            return []
        retriever = HybridRetriever(docs, **self._retriever_kwargs(commit))
        return retriever.search(query, limit=limit)

    def retrieve_across_versions(
        self,
        query: str,
        limit: int = 10,
        collapse_identical: bool = True,
    ) -> list["CrossVersionHit"]:
        """Search every indexed version at once — the Bonus goal.

        The brief states the difficulty directly: across versions the snippets
        "would still be very similar, which would make them hard to rank
        properly".  Searching all snapshots naively makes that worse, not
        better — a function that survived ten commits unchanged returns ten
        near-identical rows and crowds everything else out of the top-k.

        So results are grouped by a *canonical identity* — the file path and
        symbol name, which is what survives a version change — and each group
        collapses to its best-scoring variant, carrying the list of versions it
        appears in.  A caller asking for ten results gets ten distinct pieces
        of code rather than one piece of code ten times.

        ``collapse_identical=False`` returns the ungrouped ranking, which is
        what makes the difference measurable rather than asserted.
        """
        scored: list[tuple[float, str, VersionedDocument, tuple[str, ...]]] = []
        for commit, items in self._snapshots.items():
            docs = [item.document for item in items]
            if not docs:
                continue
            retriever = HybridRetriever(docs, **self._retriever_kwargs(commit))
            by_id = {item.document.document_id: item for item in items}
            # Over-fetch: after collapsing, a version contributes at most one
            # row per identity, so the per-version top-`limit` is not enough.
            for hit in retriever.search(query, limit=limit * 4):
                versioned = by_id.get(hit.document.document_id)
                if versioned is not None:
                    scored.append((hit.score, commit, versioned, hit.sources))

        scored.sort(key=lambda row: row[0], reverse=True)
        if not collapse_identical:
            return [
                CrossVersionHit(
                    document=item.document,
                    score=score,
                    versions=[commit],
                    content_hash=item.content_hash,
                    variants=1,
                    sources=sources,
                )
                for score, commit, item, sources in scored[:limit]
            ]

        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for score, commit, item, sources in scored:
            key = (item.document.file_path, item.document.name or item.document.document_id)
            group = groups.setdefault(
                key,
                {
                    "score": score,
                    "document": item.document,
                    "sources": sources,
                    "versions": [],
                    "hashes": set(),
                    "best_hash": item.content_hash,
                },
            )
            if commit not in group["versions"]:
                group["versions"].append(commit)
            group["hashes"].add(item.content_hash)
            if score > group["score"]:
                group.update(
                    score=score,
                    document=item.document,
                    sources=sources,
                    best_hash=item.content_hash,
                )

        ordered = sorted(groups.values(), key=lambda g: g["score"], reverse=True)
        return [
            CrossVersionHit(
                document=g["document"],
                score=g["score"],
                versions=sorted(g["versions"], key=self._version_order),
                content_hash=g["best_hash"],
                variants=len(g["hashes"]),
                sources=g["sources"],
            )
            for g in ordered[:limit]
        ]

    def _version_order(self, commit: str) -> int:
        """Insertion order, so 'versions' reads oldest-first as indexed."""
        try:
            return list(self._snapshots).index(commit)
        except ValueError:  # pragma: no cover - defensive
            return 0

    def _retriever_kwargs(self, commit: str) -> dict[str, Any]:
        """Dense vectors for this snapshot, when they were supplied.

        Without them ``HybridRetriever`` runs lexical-only, which is a
        materially weaker system than the one P0 is built on — so whether
        embeddings reached this path is worth being explicit about rather than
        discovering from a score.
        """
        embeddings = self._embeddings.get(commit)
        if embeddings is None:
            return {}
        return {"document_embeddings": embeddings, "encoder": self._encoder}

    def verify_no_cross_commit_leakage(self, results: Sequence[RankedDocument], commit: str) -> bool:
        if commit not in self._snapshots:
            return False
        allowed_ids = {item.document.document_id for item in self._snapshots[commit]}
        return all(r.document.document_id in allowed_ids for r in results)

    def find_when_changed(
        self,
        query: str,
        repo_path: Optional[str | Path] = None,
        file_path: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Answers 'when did X change' queries.
        Ranks commits by matching query tokens against commit messages and diff hunks.
        """
        results: list[dict[str, Any]] = []
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))

        for commit_sha in self.commits():
            meta = self._commit_metadata.get(commit_sha)
            score = 0.0
            matching_hunks: list[str] = []

            if meta:
                # BM25-style lexical matching over commit message
                msg_tokens = set(re.findall(r"[a-zA-Z0-9_]+", meta.message.lower()))
                common = tokens.intersection(msg_tokens)
                if common:
                    score += len(common) * 2.0

            # Inspect diff hunks if git repo provided
            if repo_path:
                try:
                    hunks = GitVersionManager.get_diff_hunks(Path(repo_path), commit_sha, file_path)
                    for hunk in hunks:
                        hunk_tokens = set(re.findall(r"[a-zA-Z0-9_]+", hunk.content.lower()))
                        hunk_common = tokens.intersection(hunk_tokens)
                        if hunk_common:
                            score += len(hunk_common) * 1.5
                            matching_hunks.append(hunk.header + "\n" + hunk.content[:200])
                except Exception:
                    pass

            if score > 0 or not tokens:
                results.append(
                    {
                        "commit": commit_sha,
                        "message": meta.message if meta else "",
                        "author": meta.author if meta else "",
                        "score": score,
                        "hunks": matching_hunks[:3],
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def save(self, path: str | Path) -> None:
        payload = {
            commit: [
                {
                    "document_id": item.document.document_id,
                    "file_path": item.document.file_path,
                    "content": item.document.content,
                    "name": item.document.name,
                    "type": item.document.type,
                    "start_line": item.document.start_line,
                    "end_line": item.document.end_line,
                    "start_byte": item.document.start_byte,
                    "end_byte": item.document.end_byte,
                    "signature": item.document.signature,
                    "content_hash": item.content_hash,
                }
                for item in documents
            ]
            for commit, documents in self._snapshots.items()
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VersionedIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        for commit, rows in payload.items():
            documents = [
                Document(
                    document_id=row["document_id"],
                    file_path=row["file_path"],
                    content=row["content"],
                    name=row.get("name", ""),
                    type=row.get("type", ""),
                    start_line=row.get("start_line", 1),
                    end_line=row.get("end_line", 1),
                    start_byte=row.get("start_byte", 0),
                    end_byte=row.get("end_byte", len(row["content"])),
                    signature=row.get("signature", ""),
                )
                for row in rows
            ]
            index.add_snapshot(commit, documents)
        return index
