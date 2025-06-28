"""Unit tests for HybridDataCollector and related classes."""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import time

from yield_arbitrage.data_collector.hybrid_collector import (
    HybridDataCollector,
    EdgePriority,
    EdgeUpdateConfig,
    EdgePriorityClassifier,
    CollectorStats
)
from yield_arbitrage.graph_engine.models import (
    UniversalYieldGraph,
    YieldGraphEdge,
    EdgeState,
    EdgeType,
    EdgeConstraints
)
from yield_arbitrage.protocols.base_adapter import ProtocolAdapter


class TestEdgeUpdateConfig:
    """Test EdgeUpdateConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EdgeUpdateConfig(
            priority=EdgePriority.NORMAL,
            update_interval_seconds=300
        )
        
        assert config.priority == EdgePriority.NORMAL
        assert config.update_interval_seconds == 300
        assert config.use_websocket is False
        assert config.confidence_threshold == 0.5
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EdgeUpdateConfig(
            priority=EdgePriority.CRITICAL,
            update_interval_seconds=1,
            use_websocket=True,
            confidence_threshold=0.8,
            max_retries=5,
            timeout_seconds=60
        )
        
        assert config.priority == EdgePriority.CRITICAL
        assert config.update_interval_seconds == 1
        assert config.use_websocket is True
        assert config.confidence_threshold == 0.8
        assert config.max_retries == 5
        assert config.timeout_seconds == 60


class TestCollectorStats:
    """Test CollectorStats dataclass."""
    
    def test_default_stats(self):
        """Test default statistics values."""
        stats = CollectorStats()
        
        assert stats.total_edges == 0
        assert stats.updates_performed == 0
        assert stats.updates_failed == 0
        assert stats.websocket_connections == 0
        assert stats.cache_hits == 0
        assert stats.avg_update_latency_ms == 0.0
        assert stats.last_update_cycle is None
        
        # Check edges_by_priority is properly initialized
        assert len(stats.edges_by_priority) == len(EdgePriority)
        for priority in EdgePriority:
            assert stats.edges_by_priority[priority] == 0
    
    def test_custom_stats(self):
        """Test custom statistics values."""
        custom_edges_by_priority = {
            EdgePriority.CRITICAL: 5,
            EdgePriority.HIGH: 10,
            EdgePriority.NORMAL: 20,
            EdgePriority.LOW: 15
        }
        
        stats = CollectorStats(
            total_edges=50,
            edges_by_priority=custom_edges_by_priority,
            updates_performed=100,
            updates_failed=5,
            websocket_connections=3,
            avg_update_latency_ms=125.5
        )
        
        assert stats.total_edges == 50
        assert stats.edges_by_priority == custom_edges_by_priority
        assert stats.updates_performed == 100
        assert stats.updates_failed == 5
        assert stats.websocket_connections == 3
        assert stats.avg_update_latency_ms == 125.5


class TestEdgePriorityClassifier:
    """Test EdgePriorityClassifier."""
    
    def test_critical_priority_classification(self):
        """Test classification of critical priority edges."""
        classifier = EdgePriorityClassifier()
        
        # Create edge with high liquidity
        edge = YieldGraphEdge(
            edge_id="test_critical_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                liquidity_usd=15_000_000.0  # Above critical threshold
            )
        )
        
        priority = classifier.classify_edge(edge)
        assert priority == EdgePriority.CRITICAL
    
    def test_high_priority_classification(self):
        """Test classification of high priority edges."""
        classifier = EdgePriorityClassifier()
        
        edge = YieldGraphEdge(
            edge_id="test_high_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                liquidity_usd=2_000_000.0  # Above high threshold, below critical
            )
        )
        
        priority = classifier.classify_edge(edge)
        assert priority == EdgePriority.HIGH
    
    def test_normal_priority_classification(self):
        """Test classification of normal priority edges."""
        classifier = EdgePriorityClassifier()
        
        edge = YieldGraphEdge(
            edge_id="test_normal_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                liquidity_usd=500_000.0  # Above normal threshold, below high
            )
        )
        
        priority = classifier.classify_edge(edge)
        assert priority == EdgePriority.NORMAL
    
    def test_low_priority_classification(self):
        """Test classification of low priority edges."""
        classifier = EdgePriorityClassifier()
        
        edge = YieldGraphEdge(
            edge_id="test_low_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                liquidity_usd=50_000.0  # Below normal threshold
            )
        )
        
        priority = classifier.classify_edge(edge)
        assert priority == EdgePriority.LOW
    
    def test_classification_with_missing_liquidity(self):
        """Test classification when liquidity data is missing."""
        classifier = EdgePriorityClassifier()
        
        edge = YieldGraphEdge(
            edge_id="test_missing_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                liquidity_usd=None  # Missing liquidity data
            )
        )
        
        priority = classifier.classify_edge(edge)
        assert priority == EdgePriority.LOW
    
    def test_classification_error_handling(self):
        """Test error handling in edge classification."""
        classifier = EdgePriorityClassifier()
        
        # Create edge with invalid state
        edge = Mock()
        edge.edge_id = "test_error_edge"
        edge.state = None  # This will cause an error
        
        priority = classifier.classify_edge(edge)
        assert priority == EdgePriority.NORMAL  # Default fallback


class MockAdapter(ProtocolAdapter):
    """Mock adapter for testing."""
    
    PROTOCOL_NAME = "MockProtocol"
    
    def __init__(self, chain_name: str, edges_to_discover: list = None):
        super().__init__(chain_name, None)  # Initialize base class
        self.chain_name = chain_name
        self.protocol_name = "MockProtocol"  # Add protocol_name property
        self.edges_to_discover = edges_to_discover or []
        self.update_calls = 0
        
        # Default update behavior
        async def default_update(edge):
            self.update_calls += 1
            print(f"MockAdapter.update_edge_state called for {edge.edge_id}")
            return EdgeState(
                conversion_rate=0.0006,
                liquidity_usd=1_200_000.0,
                gas_cost_usd=15.0,
                last_updated_timestamp=time.time(),
                confidence_score=0.95
            )
        
        self.update_edge_state_mock = default_update
    
    async def discover_edges(self):
        """Return mock edges."""
        return self.edges_to_discover
    
    async def update_edge_state(self, edge):
        """Mock edge state update."""
        if hasattr(self, 'update_edge_state_mock'):
            return await self.update_edge_state_mock(edge)
        else:
            self.update_calls += 1
            return EdgeState(
                conversion_rate=0.0006,
                liquidity_usd=1_200_000.0,
                gas_cost_usd=15.0,
                last_updated_timestamp=time.time(),
                confidence_score=0.95
            )


class TestHybridDataCollector:
    """Test HybridDataCollector class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.set = AsyncMock(return_value=True)
        return redis_mock
    
    @pytest.fixture
    def mock_graph(self):
        """Mock UniversalYieldGraph."""
        return UniversalYieldGraph()
    
    @pytest.fixture
    def sample_edge(self):
        """Create a sample edge for testing."""
        return YieldGraphEdge(
            edge_id="test_edge_1",
            edge_type=EdgeType.TRADE,
            source_asset_id="ethereum_TOKEN_weth",
            target_asset_id="ethereum_TOKEN_usdc",
            protocol_name="MockProtocol",
            chain_name="ethereum",
            constraints=EdgeConstraints(
                min_input_amount=1.0,
                max_input_amount=1000000.0
            ),
            state=EdgeState(
                conversion_rate=0.0005,
                liquidity_usd=1_000_000.0,
                gas_cost_usd=15.0,
                last_updated_timestamp=time.time(),
                confidence_score=0.9
            )
        )
    
    @pytest.fixture
    def mock_adapter(self, sample_edge):
        """Create mock adapter with sample edge."""
        adapter = MockAdapter("ethereum", [sample_edge])
        
        # Create a proper async mock that returns EdgeState
        async def mock_update_edge_state(edge):
            return EdgeState(
                conversion_rate=0.0006,
                liquidity_usd=1_200_000.0,
                gas_cost_usd=15.0,
                last_updated_timestamp=time.time(),
                confidence_score=0.95
            )
        
        adapter.update_edge_state_mock = mock_update_edge_state
        adapter.update_edge_state = mock_update_edge_state
        return adapter
    
    def test_initialization(self, mock_graph, mock_redis, mock_adapter):
        """Test HybridDataCollector initialization."""
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[mock_adapter],
            enable_websockets=True
        )
        
        assert collector.graph is mock_graph
        assert collector.redis_client is mock_redis
        assert len(collector.adapters) == 1
        assert collector.enable_websockets is True
        assert len(collector.edge_configs) == 0
        assert len(collector.last_update_times) == 0
        assert collector.stats.total_edges == 0
        assert not collector._running
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_graph, mock_redis, mock_adapter, sample_edge):
        """Test successful initialization."""
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[mock_adapter],
            enable_websockets=False
        )
        
        result = await collector.initialize()
        
        assert result is True
        assert collector.stats.total_edges == 1
        assert len(collector.edge_configs) == 1
        assert sample_edge.edge_id in collector.edge_configs
        
        # Check edge was added to graph
        assert sample_edge.edge_id in mock_graph.edges
        
        # Check adapter was called (at least once for initialization)
        assert mock_adapter.update_calls >= 1
        
        # Check Redis was called to store state
        mock_redis.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_initialize_with_adapter_failure(self, mock_graph, mock_redis):
        """Test initialization when adapter fails."""
        # Create adapter that raises exception
        failing_adapter = MockAdapter("ethereum", [])
        failing_adapter.discover_edges = AsyncMock(side_effect=Exception("Adapter failed"))
        
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[failing_adapter],
            enable_websockets=False
        )
        
        result = await collector.initialize()
        
        # Should still succeed even if one adapter fails
        assert result is True
        assert collector.stats.total_edges == 0
    
    @pytest.mark.asyncio
    async def test_edge_classification_and_configuration(self, mock_graph, mock_redis, sample_edge):
        """Test edge classification and configuration creation."""
        # Create edges with different liquidity levels
        critical_edge = YieldGraphEdge(
            edge_id="critical_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="MockProtocol",
            chain_name="ethereum",
            state=EdgeState(liquidity_usd=15_000_000.0)
        )
        
        normal_edge = YieldGraphEdge(
            edge_id="normal_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_c",
            target_asset_id="asset_d",
            protocol_name="MockProtocol",
            chain_name="ethereum",
            state=EdgeState(liquidity_usd=500_000.0)
        )
        
        adapter = MockAdapter("ethereum", [critical_edge, normal_edge])
        
        # Create specific mock function that preserves edge liquidity
        async def preserve_liquidity_update(edge):
            adapter.update_calls += 1
            return EdgeState(
                conversion_rate=0.0006,
                liquidity_usd=edge.state.liquidity_usd,  # Preserve original liquidity
                gas_cost_usd=15.0,
                last_updated_timestamp=time.time(),
                confidence_score=0.95
            )
        
        adapter.update_edge_state_mock = preserve_liquidity_update
        
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[adapter],
            enable_websockets=False
        )
        
        await collector.initialize()
        
        # Check edge classifications
        critical_config = collector.edge_configs["critical_edge"]
        normal_config = collector.edge_configs["normal_edge"]
        
        assert critical_config.priority == EdgePriority.CRITICAL
        assert critical_config.update_interval_seconds == 1
        
        assert normal_config.priority == EdgePriority.NORMAL
        assert normal_config.update_interval_seconds == 300
        
        # Check statistics
        assert collector.stats.edges_by_priority[EdgePriority.CRITICAL] == 1
        assert collector.stats.edges_by_priority[EdgePriority.NORMAL] == 1
    
    @pytest.mark.asyncio
    async def test_single_edge_update(self, mock_graph, mock_redis, mock_adapter, sample_edge):
        """Test updating a single edge."""
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[mock_adapter],
            enable_websockets=False
        )
        
        await collector.initialize()
        
        # Check that initialization worked
        assert sample_edge.edge_id in mock_graph.edges
        assert sample_edge.edge_id in collector.edge_configs
        
        # Force update the edge
        result = await collector.force_update_edge(sample_edge.edge_id)
        
        assert result is True
        assert collector.stats.updates_performed > 0
        
        # Check adapter was called for update
        # Note: During initialization, the adapter should be called once
        # During force update, it should be called again
        assert mock_adapter.update_calls >= 2  # Initial + forced
        
        # Check Redis was called to store updated state
        assert mock_redis.set.call_count >= 2  # Initial + update
    
    @pytest.mark.asyncio
    async def test_edge_update_failure_handling(self, mock_graph, mock_redis, sample_edge):
        """Test handling of edge update failures."""
        # Create adapter that fails on updates
        failing_adapter = MockAdapter("ethereum", [sample_edge])
        
        async def failing_update(edge):
            failing_adapter.update_calls += 1
            raise Exception("Update failed")
        
        failing_adapter.update_edge_state_mock = failing_update
        
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[failing_adapter],
            enable_websockets=False
        )
        
        await collector.initialize()
        
        # Try to update the edge
        result = await collector.force_update_edge(sample_edge.edge_id)
        
        assert result is False
        assert collector.stats.updates_failed > 0
        
        # Check that degraded state was created
        updated_edge = mock_graph.get_edge(sample_edge.edge_id)
        assert updated_edge.state.confidence_score < sample_edge.state.confidence_score
    
    @pytest.mark.asyncio
    async def test_batch_update_edges(self, mock_graph, mock_redis, mock_adapter):
        """Test batch updating multiple edges."""
        # Create multiple edges
        edges = []
        for i in range(3):
            edge = YieldGraphEdge(
                edge_id=f"test_edge_{i}",
                edge_type=EdgeType.TRADE,
                source_asset_id=f"asset_a_{i}",
                target_asset_id=f"asset_b_{i}",
                protocol_name="MockProtocol",
                chain_name="ethereum",
                state=EdgeState(
                    conversion_rate=0.001 * (i + 1),
                    liquidity_usd=100_000.0 * (i + 1),
                    gas_cost_usd=15.0,
                    confidence_score=0.8
                )
            )
            edges.append(edge)
        
        adapter = MockAdapter("ethereum", edges)
        
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[adapter],
            enable_websockets=False
        )
        
        await collector.initialize()
        
        # Force update all edges
        edge_ids = [edge.edge_id for edge in edges]
        await collector._batch_update_edges(edge_ids)
        
        assert collector.stats.updates_performed >= len(edges)
        
        # Check all adapters were called
        assert adapter.update_calls >= len(edges) * 2  # Initial + batch
    
    @pytest.mark.asyncio
    async def test_redis_state_storage_and_retrieval(self, mock_graph, mock_redis, mock_adapter, sample_edge):
        """Test Redis state storage and retrieval."""
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[mock_adapter],
            enable_websockets=False
        )
        
        # Test storing state
        test_state = EdgeState(
            conversion_rate=0.001,
            liquidity_usd=500_000.0,
            gas_cost_usd=20.0,
            confidence_score=0.85
        )
        
        await collector._store_edge_state("test_edge", test_state)
        
        # Check Redis set was called with correct parameters
        mock_redis.set.assert_called()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "edge_state:test_edge"
        assert "0.001" in call_args[0][1]  # Check conversion_rate in JSON
    
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_graph, mock_redis, mock_adapter, sample_edge):
        """Test getting collector statistics."""
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[mock_adapter],
            enable_websockets=False
        )
        
        await collector.initialize()
        
        stats = collector.get_stats()
        
        assert "total_edges" in stats
        assert "edges_by_priority" in stats
        assert "updates_performed" in stats
        assert "updates_failed" in stats
        assert "success_rate" in stats
        assert "websocket_connections" in stats
        assert "avg_update_latency_ms" in stats
        assert "adapters_count" in stats
        assert "background_tasks" in stats
        assert "running" in stats
        
        assert stats["total_edges"] == 1
        assert stats["adapters_count"] == 1
        assert stats["running"] is False
    
    @pytest.mark.asyncio
    async def test_force_update_all(self, mock_graph, mock_redis, mock_adapter):
        """Test forcing update of all edges."""
        # Create multiple edges
        edges = []
        for i in range(2):
            edge = YieldGraphEdge(
                edge_id=f"test_edge_{i}",
                edge_type=EdgeType.TRADE,
                source_asset_id=f"asset_a_{i}",
                target_asset_id=f"asset_b_{i}",
                protocol_name="MockProtocol",
                chain_name="ethereum",
                state=EdgeState(conversion_rate=0.001, liquidity_usd=100_000.0)
            )
            edges.append(edge)
        
        adapter = MockAdapter("ethereum", edges)
        
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[adapter],
            enable_websockets=False
        )
        
        await collector.initialize()
        
        # Force update all
        results = await collector.force_update_all()
        
        assert len(results) == len(edges)
        assert all(results.values())  # All updates should succeed
        
        # Check last update times were cleared
        assert len(collector.last_update_times) == 0
    
    def test_find_adapter_for_edge(self, mock_graph, mock_redis, sample_edge):
        """Test finding the correct adapter for an edge."""
        adapter1 = MockAdapter("ethereum")
        adapter1.PROTOCOL_NAME = "MockProtocol"
        
        adapter2 = MockAdapter("arbitrum")
        adapter2.PROTOCOL_NAME = "OtherProtocol"
        
        collector = HybridDataCollector(
            graph=mock_graph,
            redis_client=mock_redis,
            adapters=[adapter1, adapter2],
            enable_websockets=False
        )
        
        # Test finding correct adapter
        found_adapter = collector._find_adapter_for_edge(sample_edge)
        assert found_adapter is adapter1
        
        # Test with edge that has no matching adapter
        no_match_edge = YieldGraphEdge(
            edge_id="no_match",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="NonExistentProtocol",
            chain_name="ethereum"
        )
        
        found_adapter = collector._find_adapter_for_edge(no_match_edge)
        assert found_adapter is None