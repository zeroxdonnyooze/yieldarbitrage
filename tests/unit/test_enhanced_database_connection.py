"""Unit tests for enhanced database connection and session management."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import DisconnectionError, OperationalError

from yield_arbitrage.database.connection import (
    close_db,
    get_pool_status,
    get_session,
    health_check,
    init_database,
    shutdown_database,
    startup_database,
    transaction,
    execute_with_retry,
    _get_database_url,
    _get_db_config,
)


class TestDatabaseConfiguration:
    """Test database configuration functions."""
    
    def test_get_database_url_with_setting(self):
        """Test database URL retrieval when DATABASE_URL is set."""
        with patch('yield_arbitrage.database.connection.settings') as mock_settings:
            mock_settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
            
            result = _get_database_url()
            assert result == "postgresql+asyncpg://test:test@localhost/test"
    
    def test_get_database_url_fallback(self):
        """Test database URL fallback when DATABASE_URL is not set."""
        with patch('yield_arbitrage.database.connection.settings') as mock_settings:
            mock_settings.database_url = None
            
            result = _get_database_url()
            assert "postgresql+asyncpg://user:pass@localhost/yieldarbitrage" in result
    
    def test_get_db_config(self):
        """Test database configuration retrieval."""
        with patch('yield_arbitrage.database.connection.settings') as mock_settings:
            mock_settings.db_pool_size = 20
            mock_settings.db_max_overflow = 30
            mock_settings.db_pool_timeout = 45
            mock_settings.db_pool_recycle = 7200
            
            config = _get_db_config()
            
            assert config["pool_size"] == 20
            assert config["max_overflow"] == 30
            assert config["pool_timeout"] == 45
            assert config["pool_recycle"] == 7200
            assert config["pool_pre_ping"] is True
            assert "connect_args" in config


class TestDatabaseInitialization:
    """Test database initialization and lifecycle."""
    
    @pytest.fixture
    def mock_engine(self):
        """Mock SQLAlchemy engine."""
        engine = AsyncMock()
        engine.begin = AsyncMock()
        engine.pool = MagicMock()
        engine.pool.size.return_value = 15
        engine.pool.checkedin.return_value = 10
        engine.pool.checkedout.return_value = 5
        engine.pool.overflow.return_value = 0
        engine.pool.invalid.return_value = 0
        return engine
    
    @pytest.mark.asyncio
    async def test_init_database_success(self, mock_engine):
        """Test successful database initialization."""
        with patch('yield_arbitrage.database.connection.create_async_engine') as mock_create:
            with patch('yield_arbitrage.database.connection.async_sessionmaker') as mock_sessionmaker:
                with patch('yield_arbitrage.database.connection.health_check') as mock_health:
                    mock_create.return_value = mock_engine
                    mock_health.return_value = True
                    
                    await init_database()
                    
                    mock_create.assert_called_once()
                    mock_sessionmaker.assert_called_once()
                    mock_health.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_init_database_already_initialized(self, mock_engine):
        """Test database initialization when already initialized."""
        with patch('yield_arbitrage.database.connection.engine', mock_engine):
            with patch('yield_arbitrage.database.connection.logger') as mock_logger:
                await init_database()
                mock_logger.warning.assert_called_with("Database already initialized")
    
    @pytest.mark.asyncio
    async def test_init_database_failure(self):
        """Test database initialization failure."""
        with patch('yield_arbitrage.database.connection.create_async_engine') as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                await init_database()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_engine):
        """Test successful health check."""
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 1
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        with patch('yield_arbitrage.database.connection.engine', mock_engine):
            result = await health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_engine):
        """Test health check failure."""
        mock_engine.begin.side_effect = Exception("Database error")
        
        with patch('yield_arbitrage.database.connection.engine', mock_engine):
            result = await health_check()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_no_engine(self):
        """Test health check with no engine."""
        with patch('yield_arbitrage.database.connection.engine', None):
            result = await health_check()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_pool_status(self, mock_engine):
        """Test pool status retrieval."""
        with patch('yield_arbitrage.database.connection.engine', mock_engine):
            status = await get_pool_status()
            
            assert status["pool_size"] == 15
            assert status["checked_in"] == 10
            assert status["checked_out"] == 5
            assert status["overflow"] == 0
            assert status["invalid"] == 0
            assert status["total_connections"] == 15
            assert status["utilization"] == 33.33333333333333  # 5/15 * 100
    
    @pytest.mark.asyncio
    async def test_get_pool_status_no_engine(self):
        """Test pool status with no engine."""
        with patch('yield_arbitrage.database.connection.engine', None):
            status = await get_pool_status()
            assert status == {"error": "Engine not initialized"}


class TestSessionManagement:
    """Test session management functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = AsyncMock()
        session.in_transaction.return_value = True
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_session_factory(self, mock_session):
        """Mock session factory."""
        factory = MagicMock()
        factory.return_value = mock_session
        return factory
    
    @pytest.mark.asyncio
    async def test_get_session_success(self, mock_session, mock_session_factory):
        """Test successful session creation and management."""
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal', mock_session_factory):
            async with get_session() as session:
                assert session == mock_session
            
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_no_factory(self):
        """Test session creation with no factory initialized."""
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal', None):
            with pytest.raises(RuntimeError, match="Database not initialized"):
                async with get_session():
                    pass
    
    @pytest.mark.asyncio
    async def test_get_session_with_retry(self, mock_session, mock_session_factory):
        """Test session creation with connection retry."""
        # First call fails, second succeeds
        mock_session_factory.side_effect = [
            DisconnectionError("Connection lost", None, None),
            mock_session
        ]
        
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal', mock_session_factory):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                async with get_session() as session:
                    assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_get_session_max_retries_exceeded(self, mock_session_factory):
        """Test session creation when max retries are exceeded."""
        mock_session_factory.side_effect = DisconnectionError("Persistent failure", None, None)
        
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal', mock_session_factory):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(DisconnectionError):
                    async with get_session():
                        pass
    
    @pytest.mark.asyncio
    async def test_get_session_other_exception(self, mock_session, mock_session_factory):
        """Test session creation with non-connection exception."""
        mock_session_factory.return_value = mock_session
        mock_session.commit.side_effect = ValueError("Some other error")
        
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal', mock_session_factory):
            with pytest.raises(ValueError):
                async with get_session():
                    pass
            
            mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transaction_context_manager(self, mock_session, mock_session_factory):
        """Test transaction context manager."""
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal', mock_session_factory):
            async with transaction() as session:
                assert session == mock_session
            
            mock_session.begin.assert_called_once()
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transaction_with_exception(self, mock_session, mock_session_factory):
        """Test transaction context manager with exception."""
        with patch('yield_arbitrage.database.connection.AsyncSessionLocal', mock_session_factory):
            with pytest.raises(ValueError):
                async with transaction():
                    raise ValueError("Transaction error")
            
            mock_session.begin.assert_called_once()
            mock_session.rollback.assert_called_once()


class TestExecuteWithRetry:
    """Test execute with retry functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test successful query execution with retry."""
        mock_query = MagicMock()
        mock_result = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        
        with patch('yield_arbitrage.database.connection.get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await execute_with_retry(mock_query)
            assert result == mock_result
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure_then_success(self):
        """Test query execution with initial failure then success."""
        mock_query = MagicMock()
        mock_result = MagicMock()
        mock_session_fail = AsyncMock()
        mock_session_fail.execute.side_effect = DisconnectionError("Connection lost", None, None)
        mock_session_success = AsyncMock()
        mock_session_success.execute.return_value = mock_result
        
        with patch('yield_arbitrage.database.connection.get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.side_effect = [
                mock_session_fail,
                mock_session_success
            ]
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await execute_with_retry(mock_query)
                assert result == mock_result
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_max_retries_exceeded(self):
        """Test query execution when max retries are exceeded."""
        mock_query = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute.side_effect = OperationalError("Persistent error", None, None)
        
        with patch('yield_arbitrage.database.connection.get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(OperationalError):
                    await execute_with_retry(mock_query, max_retries=2)


class TestDatabaseLifecycle:
    """Test database lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_startup_database(self):
        """Test database startup process."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar.return_value = True  # Tables exist
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        with patch('yield_arbitrage.database.connection.init_database') as mock_init:
            with patch('yield_arbitrage.database.connection.engine', mock_engine):
                with patch('yield_arbitrage.database.connection.get_pool_status') as mock_pool:
                    mock_pool.return_value = {"pool_size": 15}
                    
                    await startup_database()
                    
                    mock_init.assert_called_once()
                    mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_database_create_tables(self):
        """Test database startup with table creation."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar.return_value = False  # Tables don't exist
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        with patch('yield_arbitrage.database.connection.init_database') as mock_init:
            with patch('yield_arbitrage.database.connection.engine', mock_engine):
                with patch('yield_arbitrage.database.connection.create_tables') as mock_create:
                    with patch('yield_arbitrage.database.connection.get_pool_status') as mock_pool:
                        mock_pool.return_value = {"pool_size": 15}
                        
                        await startup_database()
                        
                        mock_init.assert_called_once()
                        mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_database(self):
        """Test database shutdown process."""
        mock_engine = AsyncMock()
        
        with patch('yield_arbitrage.database.connection.engine', mock_engine):
            with patch('yield_arbitrage.database.connection.get_pool_status') as mock_pool:
                with patch('yield_arbitrage.database.connection.close_db') as mock_close:
                    mock_pool.return_value = {"pool_size": 15}
                    
                    await shutdown_database()
                    
                    mock_pool.assert_called_once()
                    mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_db_success(self):
        """Test successful database closure."""
        mock_engine = AsyncMock()
        
        with patch('yield_arbitrage.database.connection.engine', mock_engine):
            await close_db()
            mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_db_no_engine(self):
        """Test database closure with no engine."""
        with patch('yield_arbitrage.database.connection.engine', None):
            with patch('yield_arbitrage.database.connection.logger') as mock_logger:
                await close_db()
                mock_logger.warning.assert_called_with("Database engine was not initialized")
    
    @pytest.mark.asyncio
    async def test_close_db_with_error(self):
        """Test database closure with error."""
        mock_engine = AsyncMock()
        mock_engine.dispose.side_effect = Exception("Disposal error")
        
        with patch('yield_arbitrage.database.connection.engine', mock_engine):
            with patch('yield_arbitrage.database.connection.logger') as mock_logger:
                await close_db()
                mock_logger.error.assert_called()