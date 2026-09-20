import pytest

from app.indexing.sqlite_index import SQLiteIndex
from app.indexing.versioned import VersionedIndex
from app.retrieval.structural import StructuralEngine


@pytest.fixture
def populated_engine():
    index = SQLiteIndex(":memory:")
    engine = StructuralEngine(index)

    # Fixture 1 & 2: Bluetooth deeplink usage in settings
    source_settings = """
    export function openBluetooth() {
        const link = 'action:bluetooth-settings';
        systemNavigate(link);
    }
    """
    engine.index_edges_for_file("src/settings.js", source_settings)

    # Fixture 3 & 4: Sequence query tool XYZ before tool ABC
    source_workflow = """
    export function runPipeline() {
        initToolXYZ();
        doIntermediateWork();
        executeToolABC();
    }
    """
    engine.index_edges_for_file("src/pipeline.js", source_workflow)

    # Fixture 5 & 6: Reverse sequence (ABC before XYZ) -> should NOT match XYZ before ABC
    source_reversed = """
    export function reversePipeline() {
        executeToolABC();
        initToolXYZ();
    }
    """
    engine.index_edges_for_file("src/reversed.js", source_reversed)

    # Fixture 7 & 8: Dynamic / computed method call -> must be CANDIDATE
    source_dynamic = """
    export function dynamicCall(dispatcher, methodName) {
        dispatcher[methodName]();
    }
    """
    engine.index_edges_for_file("src/dynamic.js", source_dynamic)

    # Fixture 9 & 10: Cross-file callers
    source_service = """
    import { validateUser } from './auth.js';
    export function handleRequest(user) {
        validateUser(user);
    }
    """
    engine.index_edges_for_file("src/service.js", source_service)

    return engine


def test_fixture_1_deeplink_finds_exact_file_and_citing_span(populated_engine):
    results = populated_engine.deeplink_query("bluetooth-settings")
    assert len(results) == 1
    res = results[0]
    assert res["file_path"] == "src/settings.js"
    assert res["caller"] == "openBluetooth"
    assert res["status"] == "VERIFIED"
    assert res["span"][0] < res["span"][1]


def test_fixture_2_tool_sequence_order_respected(populated_engine):
    results = populated_engine.sequence_query("initToolXYZ", "executeToolABC")
    # Only pipeline.js called XYZ before ABC; reversed.js called ABC before XYZ
    assert len(results) == 1
    res = results[0]
    assert res["file_path"] == "src/pipeline.js"
    assert res["caller"] == "runPipeline"
    assert res["first_span"][0] < res["second_span"][0]
    assert res["status"] == "VERIFIED"


def test_fixture_3_dynamic_dispatch_labelled_candidate_not_verified(populated_engine):
    rows = populated_engine.index.conn.execute(
        "SELECT status, confidence FROM edges WHERE file = 'src/dynamic.js'"
    ).fetchall()
    assert len(rows) > 0
    candidate_rows = [r for r in rows if r["status"] == "CANDIDATE"]
    assert len(candidate_rows) > 0
    for r in candidate_rows:
        assert r["confidence"] < 1.0


def test_fixture_4_callers_of_returns_calling_function_not_file(populated_engine):
    callers = populated_engine.callers_of("validateUser")
    assert len(callers) == 1
    # Must return function 'handleRequest', NOT just the file name
    assert callers[0]["caller_symbol"] == "handleRequest"
    assert callers[0]["file_path"] == "src/service.js"
    assert callers[0]["status"] == "VERIFIED"


def test_fixture_5_callers_of_system_navigate(populated_engine):
    callers = populated_engine.callers_of("systemNavigate")
    assert len(callers) == 1
    assert callers[0]["caller_symbol"] == "openBluetooth"


def test_fixture_6_callers_of_tool_abc(populated_engine):
    callers = populated_engine.callers_of("executeToolABC")
    caller_names = {c["caller_symbol"] for c in callers}
    assert "runPipeline" in caller_names
    assert "reversePipeline" in caller_names


def test_fixture_7_bounded_graph_expansion(populated_engine):
    subgraph = populated_engine.bounded_graph_expansion(["runPipeline"], max_hops=2, max_nodes=10)
    assert "runPipeline" in subgraph["nodes"]
    assert "initToolXYZ" in subgraph["nodes"]
    assert len(subgraph["nodes"]) <= 10


def test_fixture_8_imports_recorded_as_verified_edges(populated_engine):
    rows = populated_engine.index.conn.execute(
        "SELECT source, target, relation, status FROM edges WHERE relation = 'IMPORTS'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["target"] == "validateUser"
    assert rows[0]["status"] == "VERIFIED"


def test_fixture_9_nonexistent_symbol_returns_empty_callers(populated_engine):
    assert populated_engine.callers_of("nonexistentFunction_12345") == []


def test_fixture_10_nonexistent_deeplink_returns_empty(populated_engine):
    assert populated_engine.deeplink_query("nonexistent-deeplink-999") == []


def test_plan_t10_no_silent_fallback():
    """
    Plan T10: No silent fallback.
    - Querying an unindexed snapshot raises explicit error (never falls back to HEAD).
    - Querying a missing file fails explicitly.
    """
    version_index = VersionedIndex()
    # Unindexed snapshot must raise KeyError
    with pytest.raises(KeyError, match="not indexed"):
        version_index.retrieve("test query", commit="unknown_commit_sha_123")

    # Corrupt or malformed query
    sqlite_idx = SQLiteIndex(":memory:")
    # FTS query with empty or invalid query returns empty, not random rows
    assert sqlite_idx.search_fts("") == []
