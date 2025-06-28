"""Unit tests for database integration."""
import pytest
from unittest.mock import AsyncMock, patch
import asyncio

# Import after adding src to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.database import get_db, create_tables, close_db, Base
from yield_arbitrage.database.connection import engine


class TestDatabaseIntegration:
    """Test database connection and setup."""
    
    def test_base_metadata_exists(self):
        """Test that SQLAlchemy Base is properly configured."""
        assert Base is not None
        assert hasattr(Base, 'metadata')
    
    def test_engine_configuration(self):
        """Test that async engine is properly configured."""
        assert engine is not None
        assert engine.url.drivername == "postgresql+asyncpg"
    
    @pytest.mark.asyncio
    async def test_get_db_dependency(self):
        """Test the FastAPI database dependency function."""
        # Mock the session to avoid actual database connection
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal') as mock_session_maker:
            mock_session = AsyncMock()
            mock_session_maker.return_value.__aenter__.return_value = mock_session
            mock_session_maker.return_value.__aexit__.return_value = None
            
            # Test the dependency
            async for db in get_db():
                assert db == mock_session
                break
    
    @pytest.mark.asyncio 
    async def test_create_tables_function(self):
        """Test the create_tables function structure."""
        # Mock the engine to avoid actual database connection
        with patch('yield_arbitrage.database.connection.engine') as mock_engine:
            mock_conn = AsyncMock()
            mock_engine.begin.return_value.__aenter__.return_value = mock_conn
            mock_engine.begin.return_value.__aexit__.return_value = None
            
            # Should not raise an exception
            await create_tables()
            
            # Verify the function attempted to create tables
            mock_engine.begin.assert_called_once()
            mock_conn.run_sync.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_db_function(self):
        """Test the close_db function."""
        with patch('yield_arbitrage.database.connection.engine') as mock_engine:
            mock_engine.dispose = AsyncMock()
            
            await close_db()
            mock_engine.dispose.assert_called_once()


def test_database_imports():
    """Test that all database components can be imported."""
    from yield_arbitrage.database import (
        AsyncSessionLocal, 
        Base, 
        close_db, 
        create_tables, 
        engine, 
        get_db
    )
    
    # Check all imports are available
    assert AsyncSessionLocal is not None
    assert Base is not None
    assert close_db is not None
    assert create_tables is not None
    assert engine is not None
    assert get_db is not None


if __name__ == "__main__":
    pytest.main([__file__])