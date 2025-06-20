"""Test FastAPI application setup."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_app_creation():
    """Test that FastAPI app can be created."""
    app = create_app()
    assert app is not None
    assert app.title == "Yield Arbitrage API"
    assert app.version == "0.1.0"


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "yield-arbitrage"
    assert data["version"] == "0.1.0"


def test_ready_endpoint(client):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data
    assert "database" in data["checks"]
    assert "redis" in data["checks"]
    assert "graph_engine" in data["checks"]


def test_openapi_spec(client):
    """Test that OpenAPI spec is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert spec["info"]["title"] == "Yield Arbitrage API"
    assert spec["info"]["version"] == "0.1.0"