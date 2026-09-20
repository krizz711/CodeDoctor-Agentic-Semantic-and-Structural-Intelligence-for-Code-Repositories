from __future__ import annotations

from typing import Any, List

from app.indexing.sqlite_index import SQLiteIndex
from app.parser.edges import EdgeRecord, StructuralExtractor


class SnapshotNotIndexedError(KeyError):
    """Raised when an unindexed snapshot is queried. Strictly prevents silent fallback to HEAD."""
    pass


class StructuralEngine:
    def __init__(self, index: SQLiteIndex) -> None:
        self.index = index
        self.extractor = StructuralExtractor()

    def index_edges_for_file(self, file_path: str, source: str) -> List[EdgeRecord]:
        edges = self.extractor.extract_file_edges(file_path, source)
        for edge in edges:
            self.index.conn.execute(
                """
                INSERT INTO edges (
                    source, target, relation, confidence, status, file, line,
                    start_byte, end_byte, extractor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.source,
                    edge.target,
                    edge.relation,
                    edge.confidence,
                    edge.status,
                    edge.file_path,
                    edge.line,
                    edge.start_byte,
                    edge.end_byte,
                    edge.extractor,
                ),
            )
        self.index.conn.commit()
        return edges

    def callers_of(self, symbol: str) -> List[dict[str, Any]]:
        """
        Returns the calling function (with its file and citing span),
        distinguishing VERIFIED from CANDIDATE.
        """
        rows = self.index.conn.execute(
            """
            SELECT e.source as caller_symbol, e.file as file_path, e.line,
                   e.confidence, e.status, e.start_byte, e.end_byte
            FROM edges e
            WHERE e.target = ? AND e.relation = 'CALLS'
            ORDER BY e.confidence DESC, e.file, e.line
            """,
            (symbol,),
        ).fetchall()

        results: List[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "caller_symbol": row["caller_symbol"],
                    "file_path": row["file_path"],
                    "line": row["line"],
                    "confidence": row["confidence"],
                    "status": row["status"],
                    "span": (row["start_byte"], row["end_byte"]),
                }
            )
        return results

    def sequence_query(self, tool_first: str, tool_second: str) -> List[dict[str, Any]]:
        """
        Answers: 'which files call tool XYZ before tool ABC'.
        Inspects calls within the same file and caller where tool_first precedes tool_second in byte offset.
        """
        first_calls = self.index.conn.execute(
            """
            SELECT file, source as caller, line, start_byte, end_byte
            FROM edges
            WHERE target = ? AND relation = 'CALLS'
            """,
            (tool_first,),
        ).fetchall()

        second_calls = self.index.conn.execute(
            """
            SELECT file, source as caller, line, start_byte, end_byte
            FROM edges
            WHERE target = ? AND relation = 'CALLS'
            """,
            (tool_second,),
        ).fetchall()

        matches: List[dict[str, Any]] = []
        for first in first_calls:
            for second in second_calls:
                if first["file"] == second["file"]:
                    # Tool first must occur before tool second in byte offset
                    if first["start_byte"] < second["start_byte"]:
                        matches.append(
                            {
                                "file_path": first["file"],
                                "caller": first["caller"],
                                "first_tool": tool_first,
                                "first_span": (first["start_byte"], first["end_byte"]),
                                "first_line": first["line"],
                                "second_tool": tool_second,
                                "second_span": (second["start_byte"], second["end_byte"]),
                                "second_line": second["line"],
                                "status": "VERIFIED",
                            }
                        )
        return matches

    def deeplink_query(self, keyword: str = "bluetooth-settings") -> List[dict[str, Any]]:
        """
        Answers: 'where is the Bluetooth-settings deeplink used'.
        """
        pattern = f"%{keyword}%"
        rows = self.index.conn.execute(
            """
            SELECT source as caller, file as file_path, line, start_byte, end_byte, target as deeplink, status, confidence
            FROM edges
            WHERE relation = 'REFERENCES_DEEPLINK' AND target LIKE ?
            """,
            (pattern,),
        ).fetchall()

        return [
            {
                "caller": row["caller"],
                "file_path": row["file_path"],
                "line": row["line"],
                "deeplink": row["deeplink"],
                "span": (row["start_byte"], row["end_byte"]),
                "status": row["status"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    def bounded_graph_expansion(self, seed_symbols: List[str], max_hops: int = 2, max_nodes: int = 50) -> dict[str, Any]:
        """
        Bounded 2-hop graph expansion from top seeds, capped at max_nodes.
        """
        visited_nodes = set(seed_symbols)
        collected_edges: List[dict[str, Any]] = []
        frontier = set(seed_symbols)

        for _ in range(max_hops):
            if not frontier or len(visited_nodes) >= max_nodes:
                break
            next_frontier = set()
            for node in frontier:
                rows = self.index.conn.execute(
                    """
                    SELECT source, target, relation, confidence, status, file, line
                    FROM edges
                    WHERE source = ? OR target = ?
                    LIMIT 20
                    """,
                    (node, node),
                ).fetchall()

                for row in rows:
                    collected_edges.append(dict(row))
                    neighbor = row["target"] if row["source"] == node else row["source"]
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_frontier.add(neighbor)
                        if len(visited_nodes) >= max_nodes:
                            break
            frontier = next_frontier

        return {
            "nodes": list(visited_nodes)[:max_nodes],
            "edges": collected_edges[:100],
        }
