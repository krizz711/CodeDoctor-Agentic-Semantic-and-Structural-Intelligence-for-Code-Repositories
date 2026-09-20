from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_api_repo_info():
    response = client.get("/api/repo/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "files_count" in data
    assert "file_tree" in data


def test_api_index_reindex():
    response = client.post("/api/index/reindex?force_rebuild=false")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "stats" in data
    assert "chunks_total" in data["stats"]


def test_api_search_returns_grounded_results_and_facets():
    response = client.post(
        "/api/search",
        json={"query": "command action option", "limit": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "evidence_bundles" in data


def test_api_structural_callers():
    response = client.get("/api/structural/callers?symbol=action")
    assert response.status_code == 200
    data = response.json()
    assert "callers" in data


def test_api_index_status_counters():
    response = client.get("/api/index/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_indexed_chunks" in data
    assert "cached_embeddings" in data
    assert "reused" in data
    assert "re_embedded" in data


def test_api_file_content_and_error_handling():
    # Nonexistent file returns 404
    bad_res = client.get("/api/repo/file?path=nonexistent_file_999.js")
    assert bad_res.status_code == 404


def test_frontend_serves_the_workbench():
    response = client.get("/")
    assert response.status_code == 200
    assert "CodeDoctor — Grounded Code Intelligence" in response.text
    # The three things the hands-on evaluation drives, by their anchors.
    for anchor in ('id="q"', 'id="passes"', 'id="timings"', 'id="view-versions"'):
        assert anchor in response.text


def test_frontend_has_no_external_dependencies():
    """A demo that needs the network to render is a demo that fails in the room."""
    response = client.get("/")
    for scheme in ("src=\"http", "src='http", "href=\"http", "href='http"):
        assert scheme not in response.text


def test_the_old_explorer_is_still_reachable():
    assert client.get("/explorer.html").status_code == 200


def test_workbench_targets_declare_availability():
    data = client.get("/api/workbench/targets").json()
    assert data["targets"]
    for target in data["targets"]:
        assert {"key", "label", "language", "available"} <= set(target)
        assert target["available"] or target["unavailable_reason"]


def test_workbench_versions_before_build_is_not_an_error():
    """The page asks for this on load; an unbuilt corpus is a state, not a fault."""
    data = client.get("/api/workbench/versions?target=commander").json()
    assert data["ready"] in (True, False)
    assert isinstance(data["versions"], list)
    assert data["presets"], "preset questions are worth showing while the index builds"


def test_workbench_search_before_build_explains_the_next_step():
    response = client.post(
        "/api/workbench/search", json={"target": "self", "query": "anything"}
    )
    assert response.status_code in (200, 409)
    if response.status_code == 409:
        assert "build" in response.json()["detail"]


def test_workbench_rejects_an_empty_query():
    response = client.post("/api/workbench/search", json={"target": "self", "query": "   "})
    assert response.status_code == 400


def test_workbench_benchmark_reads_the_submitted_artifact():
    data = client.get("/api/workbench/benchmark").json()
    if not data["available"]:
        return
    assert data["task"] == "AppsRetrieval"
    assert 0.0 < data["ndcg_at_10"] <= 1.0
    assert data["ndcg_at_10"] == data["main_score"]
