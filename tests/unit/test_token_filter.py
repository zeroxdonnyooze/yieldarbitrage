"""Unit tests for token filtering system."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta
import json

from yield_arbitrage.protocols.token_filter import (
    TokenCriteria,
    TokenInfo,
    PoolInfo,
    TokenFilter,
    TokenFilterCache,
    ExternalAPIClient,
    default_token_filter
)


class TestTokenCriteria:
    """Test TokenCriteria class."""
    
    def test_default_criteria(self):
        """Test default criteria values."""
        criteria = TokenCriteria()
        
        assert criteria.min_market_cap_usd == 1_000_000
        assert criteria.min_daily_volume_usd == 50_000
        assert criteria.min_pool_tvl_usd == 100_000
        assert criteria.max_price_impact == 0.05
        assert criteria.require_verified is True
        assert isinstance(criteria.blacklisted_tokens, set)
        assert isinstance(criteria.whitelisted_tokens, set)
    
    def test_custom_criteria(self):
        """Test custom criteria values."""
        blacklist = {"0x123", "0x456"}
        whitelist = {"0x789"}
        
        criteria = TokenCriteria(
            min_market_cap_usd=5_000_000,
            min_daily_volume_usd=100_000,
            require_verified=False,
            blacklisted_tokens=blacklist,
            whitelisted_tokens=whitelist
        )
        
        assert criteria.min_market_cap_usd == 5_000_000
        assert criteria.min_daily_volume_usd == 100_000
        assert criteria.require_verified is False
        assert criteria.blacklisted_tokens == blacklist
        assert criteria.whitelisted_tokens == whitelist


class TestTokenInfo:
    """Test TokenInfo class."""
    
    def test_token_info_creation(self):
        """Test creating TokenInfo instance."""
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            market_cap_usd=2_000_000,
            daily_volume_usd=100_000,
            is_verified=True
        )
        
        assert token.address == "0x123"
        assert token.symbol == "TEST"
        assert token.market_cap_usd == 2_000_000
        assert token.is_verified is True
    
    def test_meets_criteria_pass(self):
        """Test token that meets criteria."""
        criteria = TokenCriteria(
            min_market_cap_usd=1_000_000,
            min_daily_volume_usd=50_000,
            require_verified=True
        )
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            market_cap_usd=2_000_000,
            daily_volume_usd=100_000,
            is_verified=True
        )
        
        assert token.meets_criteria(criteria) is True
    
    def test_meets_criteria_fail_market_cap(self):
        """Test token that fails market cap criteria."""
        criteria = TokenCriteria(min_market_cap_usd=1_000_000)
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            market_cap_usd=500_000,  # Too low
            daily_volume_usd=100_000,
            is_verified=True
        )
        
        assert token.meets_criteria(criteria) is False
    
    def test_meets_criteria_fail_volume(self):
        """Test token that fails volume criteria."""
        criteria = TokenCriteria(min_daily_volume_usd=50_000)
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            market_cap_usd=2_000_000,
            daily_volume_usd=25_000,  # Too low
            is_verified=True
        )
        
        assert token.meets_criteria(criteria) is False
    
    def test_meets_criteria_fail_verification(self):
        """Test token that fails verification criteria."""
        criteria = TokenCriteria(require_verified=True)
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            market_cap_usd=2_000_000,
            daily_volume_usd=100_000,
            is_verified=False  # Not verified
        )
        
        assert token.meets_criteria(criteria) is False
    
    def test_meets_criteria_whitelisted(self):
        """Test whitelisted token bypasses criteria."""
        criteria = TokenCriteria(
            min_market_cap_usd=1_000_000,
            require_verified=True,
            whitelisted_tokens={"0x123"}
        )
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            market_cap_usd=100,  # Very low
            daily_volume_usd=10,  # Very low
            is_verified=False  # Not verified
        )
        
        # Should pass because it's whitelisted
        assert token.meets_criteria(criteria) is True
    
    def test_meets_criteria_blacklisted(self):
        """Test blacklisted token is rejected."""
        criteria = TokenCriteria(
            blacklisted_tokens={"0x123"}
        )
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            market_cap_usd=10_000_000,  # Very high
            daily_volume_usd=1_000_000,  # Very high
            is_verified=True
        )
        
        # Should fail because it's blacklisted
        assert token.meets_criteria(criteria) is False


class TestPoolInfo:
    """Test PoolInfo class."""
    
    def test_pool_info_creation(self):
        """Test creating PoolInfo instance."""
        pool = PoolInfo(
            pool_address="0xpool",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            tvl_usd=500_000,
            volume_24h_usd=100_000,
            protocol="uniswap"
        )
        
        assert pool.pool_address == "0xpool"
        assert pool.tvl_usd == 500_000
        assert pool.protocol == "uniswap"
    
    def test_meets_criteria_pass(self):
        """Test pool that meets criteria."""
        criteria = TokenCriteria(min_pool_tvl_usd=100_000)
        
        pool = PoolInfo(
            pool_address="0xpool",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            tvl_usd=500_000
        )
        
        assert pool.meets_criteria(criteria) is True
    
    def test_meets_criteria_fail_tvl(self):
        """Test pool that fails TVL criteria."""
        criteria = TokenCriteria(min_pool_tvl_usd=100_000)
        
        pool = PoolInfo(
            pool_address="0xpool",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            tvl_usd=50_000  # Too low
        )
        
        assert pool.meets_criteria(criteria) is False


class TestTokenFilterCache:
    """Test TokenFilterCache class."""
    
    def test_cache_token(self):
        """Test caching token info."""
        cache = TokenFilterCache(ttl_seconds=3600)
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18
        )
        
        # Cache token
        cache.set_token("0x123", token)
        
        # Retrieve token
        cached_token = cache.get_token("0x123")
        assert cached_token == token
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = TokenFilterCache(ttl_seconds=1)  # 1 second TTL
        
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            decimals=18
        )
        
        # Cache token
        cache.set_token("0x123", token)
        
        # Should be available immediately
        assert cache.get_token("0x123") is not None
        
        # Mock time passage
        with patch('yield_arbitrage.protocols.token_filter.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now(timezone.utc) + timedelta(seconds=2)
            
            # Should be expired
            assert cache.get_token("0x123") is None
    
    def test_cache_pool(self):
        """Test caching pool info."""
        cache = TokenFilterCache(ttl_seconds=3600)
        
        pool = PoolInfo(
            pool_address="0xpool",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            tvl_usd=500_000
        )
        
        # Cache pool
        cache.set_pool("0xpool", pool)
        
        # Retrieve pool
        cached_pool = cache.get_pool("0xpool")
        assert cached_pool == pool
    
    def test_clear_expired(self):
        """Test clearing expired entries."""
        cache = TokenFilterCache(ttl_seconds=1)
        
        token = TokenInfo(address="0x123", symbol="TEST", name="Test", decimals=18)
        cache.set_token("0x123", token)
        
        # Mock time passage to make entry expired
        with patch.object(cache, 'is_expired', return_value=True):
            cache.clear_expired()
        
        # Entry should be gone
        assert "token_0x123" not in cache._token_cache
        assert "token_0x123" not in cache._cache_timestamps


class TestExternalAPIClient:
    """Test ExternalAPIClient class."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = Mock()
        return session
    
    @pytest.fixture
    def api_client(self, mock_session):
        """Create API client with mock session."""
        return ExternalAPIClient(mock_session)
    
    @pytest.mark.asyncio
    async def test_get_token_info_coingecko_success(self, api_client, mock_session):
        """Test successful CoinGecko API call."""
        # Mock response data
        mock_response_data = {
            "id": "test-token",
            "symbol": "test",
            "name": "Test Token",
            "asset_platform_id": "ethereum",
            "detail_platforms": {
                "ethereum": {"decimal_place": 18}
            },
            "market_data": {
                "market_cap": {"usd": 2000000},
                "total_volume": {"usd": 100000},
                "current_price": {"usd": 1.5},
                "price_change_percentage_24h": 5.2
            }
        }
        
        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        # Mock the context manager properly
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_context_manager
        
        # Test API call
        token_info = await api_client.get_token_info_coingecko("0x123", "ethereum")
        
        assert token_info is not None
        assert token_info.symbol == "TEST"
        assert token_info.name == "Test Token"
        assert token_info.market_cap_usd == 2000000
        assert token_info.daily_volume_usd == 100000
        assert token_info.is_verified is True
    
    @pytest.mark.asyncio
    async def test_get_token_info_coingecko_failure(self, api_client, mock_session):
        """Test failed CoinGecko API call."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        # Mock the context manager properly
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_context_manager
        
        # Test API call
        token_info = await api_client.get_token_info_coingecko("0x123", "ethereum")
        
        assert token_info is None


class TestTokenFilter:
    """Test TokenFilter class."""
    
    @pytest.fixture
    def token_filter(self):
        """Create token filter for testing."""
        criteria = TokenCriteria(
            min_market_cap_usd=1_000_000,
            min_daily_volume_usd=50_000
        )
        return TokenFilter(criteria, cache_ttl=3600)
    
    @pytest.mark.asyncio
    async def test_filter_token_pass(self, token_filter):
        """Test filtering token that passes criteria."""
        # Mock the get_token_info method
        mock_token = TokenInfo(
            address="0x123",
            symbol="PASS",
            name="Pass Token",
            decimals=18,
            market_cap_usd=2_000_000,
            daily_volume_usd=100_000,
            is_verified=True
        )
        
        with patch.object(token_filter, 'get_token_info', return_value=mock_token):
            result = await token_filter.filter_token("0x123")
            
            assert result is True
            assert token_filter.stats["tokens_evaluated"] == 1
            assert token_filter.stats["tokens_passed"] == 1
    
    @pytest.mark.asyncio
    async def test_filter_token_fail(self, token_filter):
        """Test filtering token that fails criteria."""
        # Mock the get_token_info method
        mock_token = TokenInfo(
            address="0x123",
            symbol="FAIL",
            name="Fail Token",
            decimals=18,
            market_cap_usd=500_000,  # Too low
            daily_volume_usd=25_000,  # Too low
            is_verified=True
        )
        
        with patch.object(token_filter, 'get_token_info', return_value=mock_token):
            result = await token_filter.filter_token("0x123")
            
            assert result is False
            assert token_filter.stats["tokens_evaluated"] == 1
            assert token_filter.stats["tokens_failed"] == 1
    
    @pytest.mark.asyncio
    async def test_filter_token_no_data(self, token_filter):
        """Test filtering token with no data available."""
        with patch.object(token_filter, 'get_token_info', return_value=None):
            result = await token_filter.filter_token("0x123")
            
            assert result is False
            assert token_filter.stats["tokens_failed"] == 1
    
    @pytest.mark.asyncio
    async def test_filter_tokens_batch(self, token_filter):
        """Test filtering multiple tokens."""
        addresses = ["0x123", "0x456", "0x789"]
        
        # Mock results: first passes, others fail
        async def mock_filter_token(addr, chain="ethereum"):
            return addr == "0x123"
        
        with patch.object(token_filter, 'filter_token', side_effect=mock_filter_token):
            passed_tokens = await token_filter.filter_tokens(addresses)
            
            assert passed_tokens == ["0x123"]
    
    def test_update_criteria(self, token_filter):
        """Test updating filtering criteria."""
        original_market_cap = token_filter.criteria.min_market_cap_usd
        
        token_filter.update_criteria(min_market_cap_usd=5_000_000)
        
        assert token_filter.criteria.min_market_cap_usd == 5_000_000
        assert token_filter.criteria.min_market_cap_usd != original_market_cap
    
    def test_get_stats(self, token_filter):
        """Test getting filter statistics."""
        # Set some stats
        token_filter.stats["tokens_evaluated"] = 10
        token_filter.stats["tokens_passed"] = 6
        token_filter.stats["cache_hits"] = 3
        token_filter.stats["api_calls"] = 7
        
        stats = token_filter.get_stats()
        
        assert stats["tokens_evaluated"] == 10
        assert stats["tokens_passed"] == 6
        assert stats["success_rate"] == 60.0
        assert stats["cache_hit_rate"] == 30.0  # 3/(3+7) * 100
    
    def test_clear_cache(self, token_filter):
        """Test clearing cache."""
        # Add some data to cache
        token_filter.cache.set_token("0x123", TokenInfo("0x123", "TEST", "Test", 18))
        
        assert len(token_filter.cache._token_cache) > 0
        
        token_filter.clear_cache()
        
        assert len(token_filter.cache._token_cache) == 0
        assert len(token_filter.cache._cache_timestamps) == 0


class TestDefaultTokenFilter:
    """Test default token filter instance."""
    
    def test_default_instance_exists(self):
        """Test that default instance exists."""
        assert default_token_filter is not None
        assert isinstance(default_token_filter, TokenFilter)
        assert isinstance(default_token_filter.criteria, TokenCriteria)
    
    def test_default_criteria(self):
        """Test default criteria values."""
        criteria = default_token_filter.criteria
        
        assert criteria.min_market_cap_usd == 1_000_000
        assert criteria.min_daily_volume_usd == 50_000
        assert criteria.min_pool_tvl_usd == 100_000
        assert criteria.require_verified is True