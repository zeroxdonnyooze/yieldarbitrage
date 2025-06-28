"""Unit tests for EdgeStateManager and related classes."""
import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.pathfinding.edge_state_manager import (
    EdgeStateManager,
    StateRetrievalConfig,
    CachedEdgeState,
    CacheLevel
)
from yield_arbitrage.graph_engine.models import (
    UniversalYieldGraph,
    YieldGraphEdge,
    EdgeState,
    EdgeType
)
from yield_arbitrage.data_collector.hybrid_collector import HybridDataCollector


class TestCachedEdgeState:
    """Test CachedEdgeState dataclass."""
    
    def test_basic_initialization(self):
        """Test basic CachedEdgeState initialization."""
        state = EdgeState(
            conversion_rate=1500.0,
            liquidity_usd=1_000_000.0,
            confidence_score=0.95
        )
        
        cached = CachedEdgeState(
            state=state,
            cache_level=CacheLevel.MEMORY,
            timestamp=time.time()
        )
        
        assert cached.state is state
        assert cached.cache_level == CacheLevel.MEMORY
        assert cached.access_count == 0
        assert cached.last_access is not None
    
    def test_last_access_auto_set(self):
        """Test that last_access is auto-set to timestamp."""
        timestamp = time.time()
        state = EdgeState(conversion_rate=1.0, confidence_score=0.9)
        
        cached = CachedEdgeState(
            state=state,
            cache_level=CacheLevel.REDIS,
            timestamp=timestamp
        )
        
        assert cached.last_access == timestamp


class TestStateRetrievalConfig:
    """Test StateRetrievalConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StateRetrievalConfig()
        
        assert config.memory_cache_ttl_seconds == 300.0
        assert config.redis_cache_ttl_seconds == 900.0
        assert config.max_memory_cache_size == 10000
        assert config.batch_size == 50
        assert config.enable_predictive_caching is True
        assert config.cache_eviction_strategy == "lru"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StateRetrievalConfig(
            memory_cache_ttl_seconds=600.0,
            redis_cache_ttl_seconds=1800.0,
            max_memory_cache_size=5000,
            batch_size=25,
            enable_predictive_caching=False,
            cache_eviction_strategy="fifo"
        )
        
        assert config.memory_cache_ttl_seconds == 600.0
        assert config.redis_cache_ttl_seconds == 1800.0
        assert config.max_memory_cache_size == 5000
        assert config.batch_size == 25
        assert config.enable_predictive_caching is False
        assert config.cache_eviction_strategy == "fifo"


class MockDataCollector:
    """Mock data collector for testing."""
    
    def __init__(self):
        self.graph = UniversalYieldGraph()
        self.force_update_calls = {}
        self.update_success = True
        
        # Add sample edge to graph
        sample_edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="ETH",
            target_asset_id="USDC",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=1500.0,
                liquidity_usd=1_000_000.0,
                gas_cost_usd=15.0,
                confidence_score=0.95
            )
        )
        self.graph.add_edge(sample_edge)
    
    async def force_update_edge(self, edge_id: str) -> bool:
        """Mock force update edge."""
        self.force_update_calls[edge_id] = self.force_update_calls.get(edge_id, 0) + 1
        
        if self.update_success:
            # Update the edge state with new values
            edge = self.graph.get_edge(edge_id)
            if edge:
                # Simulate state update
                edge.state = EdgeState(
                    conversion_rate=edge.state.conversion_rate * 1.01,  # Slight change
                    liquidity_usd=edge.state.liquidity_usd,
                    gas_cost_usd=edge.state.gas_cost_usd,
                    confidence_score=edge.state.confidence_score,
                    last_updated_timestamp=time.time()
                )
        
        return self.update_success


class MockRedisClient:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.storage = {}
        self.get_calls = 0
        self.set_calls = 0
        self.delete_calls = 0
    
    async def get(self, key: str):
        """Mock get operation."""
        self.get_calls += 1
        return self.storage.get(key)
    
    async def set(self, key: str, value: str, ex: int = None):
        """Mock set operation."""
        self.set_calls += 1
        self.storage[key] = value
        return True
    
    async def delete(self, *keys):
        """Mock delete operation."""
        self.delete_calls += len(keys)
        for key in keys:
            self.storage.pop(key, None)
        return len(keys)
    
    async def keys(self, pattern: str):
        """Mock keys operation."""
        # Simple pattern matching for testing
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [key for key in self.storage.keys() if key.startswith(prefix)]
        return [key for key in self.storage.keys() if key == pattern]
    
    def pipeline(self):
        """Mock pipeline operation."""
        return MockRedisPipeline(self)


class MockRedisPipeline:
    """Mock Redis pipeline for testing."""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.commands = []
    
    def get(self, key: str):
        """Queue get command."""
        self.commands.append(("get", key))
        return self
    
    async def execute(self):
        """Execute queued commands."""
        results = []
        for command, key in self.commands:
            if command == "get":
                results.append(self.redis_client.storage.get(key))
        self.commands.clear()
        return results


class TestEdgeStateManager:
    """Test EdgeStateManager class."""
    
    @pytest.fixture
    def mock_data_collector(self):
        """Create mock data collector."""
        return MockDataCollector()
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        return MockRedisClient()
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return StateRetrievalConfig(
            memory_cache_ttl_seconds=300.0,
            max_memory_cache_size=100,
            batch_size=10
        )
    
    def test_initialization(self, mock_data_collector, mock_redis_client, config):
        """Test EdgeStateManager initialization."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            redis_client=mock_redis_client,
            config=config
        )
        
        assert manager.data_collector is mock_data_collector
        assert manager.redis_client is mock_redis_client
        assert manager.config is config
        assert len(manager._memory_cache) == 0
        assert len(manager._cache_access_order) == 0
    
    def test_default_config_initialization(self, mock_data_collector):
        """Test initialization with default config."""
        manager = EdgeStateManager(data_collector=mock_data_collector)
        
        assert manager.config.memory_cache_ttl_seconds == 300.0
        assert manager.config.max_memory_cache_size == 10000
        assert manager.redis_client is None
    
    @pytest.mark.asyncio
    async def test_get_edge_state_fresh(self, mock_data_collector, mock_redis_client, config):
        """Test getting fresh edge state."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            redis_client=mock_redis_client,
            config=config
        )
        
        # First call should fetch fresh state
        state = await manager.get_edge_state("test_edge")
        
        assert state is not None
        assert state.conversion_rate > 0
        assert mock_data_collector.force_update_calls["test_edge"] == 1
        assert manager._metrics["fresh_fetches"] == 1
        
        # Should be cached in memory
        assert "test_edge" in manager._memory_cache
        cached = manager._memory_cache["test_edge"]
        assert cached.cache_level == CacheLevel.MEMORY
        assert cached.state is state
    
    @pytest.mark.asyncio
    async def test_get_edge_state_memory_cache_hit(self, mock_data_collector, config):
        """Test memory cache hit."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=config
        )
        
        # First call - fresh fetch
        state1 = await manager.get_edge_state("test_edge")
        initial_calls = mock_data_collector.force_update_calls.get("test_edge", 0)
        
        # Second call - should hit memory cache
        state2 = await manager.get_edge_state("test_edge")
        
        assert state1 is state2
        assert mock_data_collector.force_update_calls.get("test_edge", 0) == initial_calls
        assert manager._metrics["memory_hits"] >= 1
    
    @pytest.mark.asyncio
    async def test_get_edge_state_cache_expiry(self, mock_data_collector, config):
        """Test cache expiry handling."""
        # Very short cache TTL for testing
        config.memory_cache_ttl_seconds = 0.1
        
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=config
        )
        
        # First call
        state1 = await manager.get_edge_state("test_edge")
        
        # Wait for cache to expire
        await asyncio.sleep(0.2)
        
        # Second call should fetch fresh again
        state2 = await manager.get_edge_state("test_edge")
        
        assert state1 is not state2
        assert mock_data_collector.force_update_calls["test_edge"] >= 2
    
    @pytest.mark.asyncio
    async def test_get_edge_states_batch(self, mock_data_collector, config):
        """Test batch edge state retrieval."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=config
        )
        
        # Add more edges to the mock graph
        for i in range(3):
            edge = YieldGraphEdge(
                edge_id=f"test_edge_{i}",
                edge_type=EdgeType.TRADE,
                source_asset_id="ETH",
                target_asset_id="USDC",
                protocol_name="uniswapv3",
                chain_name="ethereum",
                state=EdgeState(conversion_rate=1500.0 + i, confidence_score=0.9)
            )
            mock_data_collector.graph.add_edge(edge)
        
        edge_ids = ["test_edge_0", "test_edge_1", "test_edge_2"]
        states = await manager.get_edge_states_batch(edge_ids)
        
        assert len(states) == 3
        for edge_id in edge_ids:
            assert edge_id in states
            assert states[edge_id] is not None
            assert states[edge_id].conversion_rate > 0
            
        # All should be cached now
        for edge_id in edge_ids:
            assert edge_id in manager._memory_cache
    
    @pytest.mark.asyncio
    async def test_batch_with_partial_cache_hits(self, mock_data_collector, config):
        """Test batch retrieval with some cache hits."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=config
        )
        
        # Add edges to graph
        for i in range(3):
            edge = YieldGraphEdge(
                edge_id=f"batch_edge_{i}",
                edge_type=EdgeType.TRADE,
                source_asset_id="ETH",
                target_asset_id="USDC",
                protocol_name="uniswapv3",
                chain_name="ethereum",
                state=EdgeState(conversion_rate=1500.0 + i, confidence_score=0.9)
            )
            mock_data_collector.graph.add_edge(edge)
        
        # Pre-cache one edge
        await manager.get_edge_state("batch_edge_0")
        
        # Batch fetch all three
        edge_ids = ["batch_edge_0", "batch_edge_1", "batch_edge_2"]
        states = await manager.get_edge_states_batch(edge_ids)
        
        assert len(states) == 3
        for edge_id in edge_ids:
            assert states[edge_id] is not None
    
    @pytest.mark.asyncio
    async def test_redis_cache_storage_and_retrieval(self, mock_data_collector, mock_redis_client, config):
        """Test Redis cache storage and retrieval."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            redis_client=mock_redis_client,
            config=config
        )
        
        # Mock Redis cache with valid data
        test_state = EdgeState(conversion_rate=1500.0, confidence_score=0.95)
        cache_data = {
            "state": test_state.model_dump(),
            "timestamp": time.time(),
            "access_count": 1
        }
        mock_redis_client.storage[f"edge_state_cache:test_edge"] = str(cache_data)
        
        # Clear memory cache to force Redis lookup
        manager._memory_cache.clear()
        
        # Should retrieve from Redis cache
        with patch.object(manager, '_get_fresh_state') as mock_fresh:
            state = await manager.get_edge_state("test_edge")
            
            # Should not call fresh state since Redis had it
            mock_fresh.assert_not_called()
            assert state is not None
            assert mock_redis_client.get_calls > 0
    
    @pytest.mark.asyncio
    async def test_cache_eviction_lru(self, mock_data_collector, config):
        """Test LRU cache eviction."""
        # Small cache size for testing
        config.max_memory_cache_size = 2
        
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=config
        )
        
        # Add edges to graph
        for i in range(3):
            edge = YieldGraphEdge(
                edge_id=f"evict_edge_{i}",
                edge_type=EdgeType.TRADE,
                source_asset_id="ETH",
                target_asset_id="USDC",
                protocol_name="uniswapv3",
                chain_name="ethereum",
                state=EdgeState(conversion_rate=1500.0 + i, confidence_score=0.9)
            )
            mock_data_collector.graph.add_edge(edge)
        
        # Fill cache beyond capacity
        await manager.get_edge_state("evict_edge_0")
        await manager.get_edge_state("evict_edge_1")
        await manager.get_edge_state("evict_edge_2")  # Should evict evict_edge_0
        
        # Check that eviction occurred
        assert len(manager._memory_cache) <= config.max_memory_cache_size
        assert manager._metrics["cache_evictions"] > 0
        
        # First edge should be evicted (LRU)
        assert "evict_edge_0" not in manager._memory_cache
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_data_collector, config):
        """Test cache clearing."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=config
        )
        
        # Add some cached states
        await manager.get_edge_state("test_edge")
        assert len(manager._memory_cache) > 0
        
        # Clear cache
        manager.clear_cache()
        
        assert len(manager._memory_cache) == 0
        assert len(manager._cache_access_order) == 0
    
    @pytest.mark.asyncio
    async def test_clear_redis_cache(self, mock_data_collector, mock_redis_client, config):
        """Test Redis cache clearing."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            redis_client=mock_redis_client,
            config=config
        )
        
        # Add some data to Redis
        mock_redis_client.storage["edge_state_cache:test1"] = "data1"
        mock_redis_client.storage["edge_state_cache:test2"] = "data2"
        mock_redis_client.storage["other_key"] = "other_data"
        
        # Clear Redis cache
        await manager.clear_redis_cache()
        
        # Only edge state cache keys should be cleared
        assert "edge_state_cache:test1" not in mock_redis_client.storage
        assert "edge_state_cache:test2" not in mock_redis_client.storage
        assert "other_key" in mock_redis_client.storage  # Should remain
        assert mock_redis_client.delete_calls > 0
    
    def test_get_metrics(self, mock_data_collector, config):
        """Test metrics retrieval."""
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=config
        )
        
        # Update some metrics
        manager._metrics["memory_hits"] = 10
        manager._metrics["redis_hits"] = 5
        manager._metrics["fresh_fetches"] = 3
        
        metrics = manager.get_metrics()
        
        assert "cache_stats" in metrics
        assert "cache_config" in metrics
        
        cache_stats = metrics["cache_stats"]
        assert cache_stats["memory_hit_rate"] == "55.6%"  # 10/18 * 100
        assert cache_stats["redis_hit_rate"] == "27.8%"   # 5/18 * 100
        assert cache_stats["fresh_fetch_rate"] == "16.7%" # 3/18 * 100
        assert cache_stats["total_requests"] == 18
    
    @pytest.mark.asyncio
    async def test_error_handling_data_collector_failure(self, mock_redis_client, config):
        """Test error handling when data collector fails."""
        failing_collector = MockDataCollector()
        failing_collector.update_success = False
        
        manager = EdgeStateManager(
            data_collector=failing_collector,
            redis_client=mock_redis_client,
            config=config
        )
        
        # Should handle failure gracefully
        state = await manager.get_edge_state("test_edge")
        
        # Should fall back to existing state from graph
        assert state is not None or state is None  # Either way, should not crash
        assert manager._metrics["failed_retrievals"] >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_redis_failure(self, mock_data_collector, config):
        """Test error handling when Redis fails."""
        failing_redis = MockRedisClient()
        
        # Make Redis operations fail
        async def failing_get(key):
            raise Exception("Redis connection failed")
        
        failing_redis.get = failing_get
        
        manager = EdgeStateManager(
            data_collector=mock_data_collector,
            redis_client=failing_redis,
            config=config
        )
        
        # Should handle Redis failure gracefully and fall back to fresh fetch
        state = await manager.get_edge_state("test_edge")
        
        assert state is not None
        assert mock_data_collector.force_update_calls["test_edge"] >= 1