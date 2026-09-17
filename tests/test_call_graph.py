from __future__ import annotations

import tempfile

from app.graph.call_graph import CodeCallGraph
from app.graph.repo_map import generate_repo_map


def test_add_call_and_query():
    g = CodeCallGraph()
    g.add_call("foo", "bar", "test.js", 10)
    assert g.callers_of("bar") == ["foo"]
    assert g.callees_of("foo") == ["bar"]


def test_self_call_ignored():
    g = CodeCallGraph()
    g.add_call("foo", "foo", "test.js", 10)
    assert g.edge_count == 0


def test_pagerank():
    g = CodeCallGraph()
    g.add_call("main", "auth", "app.js", 1)
    g.add_call("main", "db", "app.js", 2)
    g.add_call("auth", "db", "auth.js", 5)
    ranks = g.pagerank()
    assert ranks["db"] > ranks["main"]  # db is called by two functions


def test_find_hubs():
    g = CodeCallGraph()
    g.add_call("a", "hub", "f.js", 1)
    g.add_call("b", "hub", "f.js", 2)
    g.add_call("c", "hub", "f.js", 3)
    hubs = g.find_hubs(1)
    assert hubs[0][0] == "hub"


def test_dependency_chain():
    g = CodeCallGraph()
    g.add_call("a", "b", "f.js", 1)
    g.add_call("b", "c", "f.js", 2)
    chain = g.dependency_chain("a", "c")
    assert chain == ["a", "b", "c"]


def test_dependency_chain_no_path():
    g = CodeCallGraph()
    g.add_call("a", "b", "f.js", 1)
    assert g.dependency_chain("b", "a") is None


def test_find_cycles():
    g = CodeCallGraph()
    g.add_call("a", "b", "f.js", 1)
    g.add_call("b", "a", "f.js", 2)
    cycles = g.find_cycles()
    assert len(cycles) >= 1


def test_save_and_load():
    g = CodeCallGraph()
    g.add_call("foo", "bar", "test.js", 10, confidence=0.9)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    g.save(path)
    loaded = CodeCallGraph.load(path)
    assert loaded.node_count == g.node_count
    assert loaded.edge_count == g.edge_count


def test_subgraph_extraction():
    g = CodeCallGraph()
    g.add_call("a", "b", "f.js", 1)
    g.add_call("b", "c", "f.js", 2)
    g.add_call("c", "d", "f.js", 3)
    g.add_call("x", "y", "f.js", 4)
    sub = g.subgraph("b", depth=1)
    assert "a" in sub.graph.nodes
    assert "c" in sub.graph.nodes
    assert "x" not in sub.graph.nodes


def test_empty_graph():
    g = CodeCallGraph()
    assert g.node_count == 0
    assert g.edge_count == 0
    assert g.callers_of("nonexistent") == []
    assert g.pagerank() == {}
    assert g.find_hubs() == []


def test_repo_map_generation():
    g = CodeCallGraph()
    g.add_call("main", "auth", "app.js", 1)
    g.add_call("auth", "db", "auth.js", 5)
    symbols = {
        "main": {"file_path": "app.js", "type": "FUNCTION", "signature": "main()", "start_line": 1},
        "auth": {"file_path": "auth.js", "type": "FUNCTION", "signature": "auth(user)", "start_line": 5},
        "db": {"file_path": "db.js", "type": "FUNCTION", "signature": "db()", "start_line": 1},
    }
    repo_map = generate_repo_map(g, symbols, max_tokens=500)
    assert "main" in repo_map
    assert "auth" in repo_map
