"""
Unit tests for FastAPI service endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns HTTP 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_list_nodes_endpoint():
    """Verify /forecast/hierarchy/nodes returns list of node IDs."""
    response = client.get("/forecast/hierarchy/nodes")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) > 0


def test_forecast_node_endpoint():
    """Verify /forecast/India returns national forecast time series."""
    response = client.get("/forecast/India")
    assert response.status_code == 200
    data = response.json()
    assert data["node_id"] == "India"
    assert len(data["forecast"]) > 0


def test_forecast_invalid_node_endpoint():
    """Verify invalid node ID returns HTTP 404."""
    response = client.get("/forecast/InvalidNode123")
    assert response.status_code == 404


def test_ingest_telemetry_endpoint():
    """Verify POST /ingest processes telemetry observation and computes comparison."""
    payload = {
        "node_id": "India",
        "date": "2025-01-05",
        "actual_value": 4200.0
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ingested"
    assert data["comparison"]["actual"] == 4200.0
    assert "anomaly_status" in data["comparison"]


def test_evaluation_summary_endpoint():
    """Verify /evaluation/summary returns evaluation results."""
    response = client.get("/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert "pooled_results" in data or "results_summary" in data
    assert data.get("num_origins") == 6
    assert 7 in data.get("horizons", [])
    assert 14 in data.get("horizons", [])
    assert 30 in data.get("horizons", [])


def test_schedule_recommend_endpoint():
    """Verify POST /schedule/recommend returns cost-optimal schedule."""
    payload = {
        "node_id": "India",
        "date": "2025-01-05",
        "normal_rate": 4000.0
    }
    response = client.post("/schedule/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_schedule_mw" in data
    assert "schedule_bias_mw" in data
    assert data["node_id"] == "India"
    assert data["optimal_schedule_mw"] > 0


def test_pooling_benefit_endpoint():
    """Verify GET /schedule/pooling-benefit returns regional savings."""
    response = client.get("/schedule/pooling-benefit")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "by_region" in data
    assert data["summary"]["total_savings_inr"] >= 0.0
