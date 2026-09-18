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


def test_frontend_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "CodeDoctor — Grounded Code Intelligence" in response.text
    assert "id=\"stat-chunks\"" in response.text
