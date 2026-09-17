from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class CallEdge:
    caller: str
    callee: str
    file_path: str
    line: int
    relation: str = "CALLS"
    confidence: float = 1.0


class CodeCallGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_call(self, caller: str, callee: str, file_path: str = "", line: int = 0, confidence: float = 1.0) -> None:
        """Add a caller -> callee edge."""
        # Don't add self-calls
        if caller == callee:
            return
        self.graph.add_edge(caller, callee, file_path=file_path, line=line, confidence=confidence, relation="CALLS")

    def callers_of(self, symbol: str) -> list[str]:
        """Find all functions that call the given symbol."""
        if symbol not in self.graph:
            return []
        return list(self.graph.predecessors(symbol))

    def callees_of(self, symbol: str) -> list[str]:
        """Find all functions called by the given symbol."""
        if symbol not in self.graph:
            return []
        return list(self.graph.successors(symbol))

    def edges_for(self, symbol: str) -> list[CallEdge]:
        """Get all edges (incoming + outgoing) for a symbol."""
        edges = []
        if symbol in self.graph:
            for _, target, data in self.graph.out_edges(symbol, data=True):
                edges.append(
                    CallEdge(
                        caller=symbol,
                        callee=target,
                        **{k: v for k, v in data.items() if k != "relation"},
                        relation=data.get("relation", "CALLS"),
                    )
                )
            for source, _, data in self.graph.in_edges(symbol, data=True):
                edges.append(
                    CallEdge(
                        caller=source,
                        callee=symbol,
                        **{k: v for k, v in data.items() if k != "relation"},
                        relation=data.get("relation", "CALLS"),
                    )
                )
        return edges

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Compute PageRank to find the most important symbols in the codebase."""
        if not self.graph.nodes:
            return {}
        return nx.pagerank(self.graph, alpha=alpha)

    def find_hubs(self, top_k: int = 20) -> list[tuple[str, float]]:
        """Find the top-k most important symbols by PageRank."""
        ranks = self.pagerank()
        return sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def dependency_chain(self, source: str, target: str) -> list[str] | None:
        """Find the shortest call path from source to target."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_cycles(self) -> list[list[str]]:
        """Detect circular call dependencies."""
        return list(nx.simple_cycles(self.graph))

    def connected_components(self) -> list[set[str]]:
        """Find groups of interconnected symbols."""
        return [set(c) for c in nx.weakly_connected_components(self.graph)]

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON/API responses."""
        return nx.node_link_data(self.graph)

    def save(self, path: str | Path) -> None:
        """Save graph to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CodeCallGraph:
        """Load graph from JSON file."""
        with open(path) as f:
            data = json.load(f)
        instance = cls()
        instance.graph = nx.node_link_graph(data)
        return instance

    def subgraph(self, symbol: str, depth: int = 2) -> CodeCallGraph:
        """Extract a subgraph centered on a symbol within given depth."""
        if symbol not in self.graph:
            return CodeCallGraph()
        # BFS to find nodes within depth
        nodes = {symbol}
        frontier = {symbol}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.predecessors(node))
                next_frontier.update(self.graph.successors(node))
            nodes.update(next_frontier)
            frontier = next_frontier
        sub = CodeCallGraph()
        sub.graph = self.graph.subgraph(nodes).copy()
        return sub
