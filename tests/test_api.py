from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.server import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_search_without_index():
    response = client.get("/api/v1/search?q=test")
    assert response.status_code == 503  # index not loaded


def test_stats_without_index():
    response = client.get("/api/v1/stats")
    assert response.status_code == 503


def test_symbol_without_index():
    response = client.get("/api/v1/symbol/test")
    assert response.status_code == 503


def test_callers_without_index():
    response = client.get("/api/v1/callers/test")
    assert response.status_code == 503
