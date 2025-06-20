"""Unit tests for Redis integration."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

# Import after adding src to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.cache.redis_client import get_redis, init_redis, close_redis, health_check


class TestRedisIntegration:
    """Test Redis connection and client management."""
    
    def setup_method(self):
        """Reset Redis client before each test."""
        with patch('yield_arbitrage.cache.redis_client.redis_client', None):
            pass
    
    @pytest.mark.asyncio
    async def test_init_redis_success(self):
        """Test successful Redis initialization."""
        with patch('yield_arbitrage.cache.redis_client.redis.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis_class.return_value = mock_redis
            
            with patch('yield_arbitrage.cache.redis_client.settings') as mock_settings:
                mock_settings.redis_url = "redis://localhost:6379/0"
                
                await init_redis()
                
                # Verify Redis client was created with correct parameters
                mock_redis_class.assert_called_once()
                call_kwargs = mock_redis_class.call_args[1]
                assert call_kwargs['host'] == 'localhost'
                assert call_kwargs['port'] == 6379
                assert call_kwargs['db'] == 0
                
                # Verify ping was called
                mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_init_redis_with_auth(self):
        """Test Redis initialization with authentication."""
        with patch('yield_arbitrage.cache.redis_client.redis.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis_class.return_value = mock_redis
            
            with patch('yield_arbitrage.cache.redis_client.settings') as mock_settings:
                mock_settings.redis_url = "redis://user:pass@localhost:6379/1"
                
                await init_redis()
                
                call_kwargs = mock_redis_class.call_args[1]
                assert call_kwargs['host'] == 'localhost'
                assert call_kwargs['port'] == 6379
                assert call_kwargs['db'] == 1
                assert call_kwargs['username'] == 'user'
                assert call_kwargs['password'] == 'pass'
    
    @pytest.mark.asyncio
    async def test_init_redis_no_url(self):
        """Test Redis initialization without URL raises error."""
        with patch('yield_arbitrage.cache.redis_client.settings') as mock_settings:
            mock_settings.redis_url = None
            
            with pytest.raises(ValueError, match="Redis URL not configured"):
                await init_redis()
    
    @pytest.mark.asyncio
    async def test_init_redis_connection_error(self):
        """Test Redis initialization with connection error."""
        with patch('yield_arbitrage.cache.redis_client.redis.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(side_effect=ConnectionError("Connection failed"))
            mock_redis_class.return_value = mock_redis
            
            with patch('yield_arbitrage.cache.redis_client.settings') as mock_settings:
                mock_settings.redis_url = "redis://localhost:6379/0"
                
                with pytest.raises(ConnectionError):
                    await init_redis()
    
    @pytest.mark.asyncio
    async def test_get_redis_creates_client(self):
        """Test get_redis creates client if not exists."""
        with patch('yield_arbitrage.cache.redis_client.init_redis') as mock_init:
            with patch('yield_arbitrage.cache.redis_client.redis_client', None):
                mock_init.return_value = None
                
                await get_redis()
                mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_redis(self):
        """Test Redis connection closure."""
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        
        with patch('yield_arbitrage.cache.redis_client.redis_client', mock_redis):
            await close_redis()
            mock_redis.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test Redis health check when healthy."""
        with patch('yield_arbitrage.cache.redis_client.get_redis') as mock_get_redis:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_get_redis.return_value = mock_redis
            
            result = await health_check()
            assert result is True
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test Redis health check when unhealthy."""
        with patch('yield_arbitrage.cache.redis_client.get_redis') as mock_get_redis:
            mock_get_redis.side_effect = ConnectionError("Connection failed")
            
            result = await health_check()
            assert result is False


def test_redis_imports():
    """Test that all Redis components can be imported."""
    from yield_arbitrage.cache import get_redis, close_redis, redis_client
    
    # Check all imports are available
    assert get_redis is not None
    assert close_redis is not None
    # redis_client might be None initially


if __name__ == "__main__":
    pytest.main([__file__])