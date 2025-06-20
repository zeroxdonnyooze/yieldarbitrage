"""Integration test for application startup with all services."""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# Import after adding src to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.main import create_app


def test_app_startup_with_mocked_services():
    """Test that the app can start up with mocked database and Redis."""
    
    # Mock database operations
    with patch('yield_arbitrage.database.connection.create_tables') as mock_create_tables, \
         patch('yield_arbitrage.database.connection.close_db') as mock_close_db, \
         patch('yield_arbitrage.cache.redis_client.get_redis') as mock_get_redis, \
         patch('yield_arbitrage.cache.redis_client.close_redis') as mock_close_redis:
        
        # Configure mocks
        mock_create_tables.return_value = None
        mock_close_db.return_value = None
        mock_get_redis.return_value = AsyncMock()
        mock_close_redis.return_value = None
        
        # Create app
        app = create_app()
        
        # Test the app can start
        with TestClient(app) as client:
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "yield-arbitrage"
            assert data["version"] == "0.1.0"


def test_readiness_endpoint_structure():
    """Test the readiness endpoint returns proper structure."""
    with patch('yield_arbitrage.database.connection.create_tables'), \
         patch('yield_arbitrage.database.connection.close_db'), \
         patch('yield_arbitrage.cache.redis_client.get_redis'), \
         patch('yield_arbitrage.cache.redis_client.close_redis'):
        
        app = create_app()
        
        with TestClient(app) as client:
            response = client.get("/ready")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert "checks" in data
            assert "database" in data["checks"]
            assert "redis" in data["checks"] 
            assert "graph_engine" in data["checks"]


if __name__ == "__main__":
    pytest.main([__file__])