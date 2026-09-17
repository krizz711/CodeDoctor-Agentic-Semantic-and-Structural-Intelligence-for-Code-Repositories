from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.retrieval.hybrid import Document, HybridRetriever, RankedDocument

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class VersionedDocument:
    commit: str
    document: Document
    content_hash: str

    @classmethod
    def create(cls, commit: str, document: Document) -> VersionedDocument:
        digest = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        return cls(commit, document, digest)


class VersionedIndex:
    def __init__(self) -> None:
        self._snapshots: dict[str, list[VersionedDocument]] = {}
        self._retriever_cache: dict[str, HybridRetriever] = {}

    def add_snapshot(self, commit: str, documents: Iterable[Document]) -> None:
        self._snapshots[commit] = [VersionedDocument.create(commit, document) for document in documents]
        self._retriever_cache.pop(commit, None)  # invalidate cache on update

    def commits(self) -> list[str]:
        return sorted(self._snapshots)

    def retrieve(self, query: str, commit: str, limit: int = 10) -> list[RankedDocument]:
        if commit not in self._snapshots:
            raise KeyError(f"Snapshot {commit} is not indexed")
        if commit not in self._retriever_cache:
            documents = [item.document for item in self._snapshots[commit]]
            self._retriever_cache[commit] = HybridRetriever(documents)
        return self._retriever_cache[commit].search(query, limit=limit)

    def verify_no_cross_commit_leakage(self, results: Sequence[RankedDocument], commit: str) -> bool:
        allowed = {item.document.document_id for item in self._snapshots.get(commit, [])}
        return all(result.document.document_id in allowed for result in results)

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
                    "content_hash": item.content_hash,
                }
                for item in documents
            ]
            for commit, documents in self._snapshots.items()
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VersionedIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        for commit, rows in payload.items():
            documents = [
                Document(
                    row["document_id"],
                    row["file_path"],
                    row["content"],
                    row.get("name", ""),
                    row.get("type", ""),
                    row.get("start_line", 1),
                    row.get("end_line", 1),
                )
                for row in rows
            ]
            index.add_snapshot(commit, documents)
        return index
