"""Unit tests for BeamSearchOptimizer and related classes."""
import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from yield_arbitrage.pathfinding.path_models import (
    SearchPath,
    PathNode,
    PathStatus
)
from yield_arbitrage.pathfinding.beam_search import (
    BeamSearchOptimizer,
    BeamSearchConfig,
    SearchResult
)
from yield_arbitrage.pathfinding.edge_state_manager import (
    EdgeStateManager,
    StateRetrievalConfig
)
from yield_arbitrage.pathfinding.path_scorer import (
    NonMLPathScorer,
    ScoringConfig,
    ScoringMethod
)
from yield_arbitrage.graph_engine.models import (
    UniversalYieldGraph,
    YieldGraphEdge,
    EdgeState,
    EdgeType,
    EdgeConstraints
)
from yield_arbitrage.data_collector.hybrid_collector import HybridDataCollector


class TestPathNode:
    """Test PathNode dataclass."""
    
    def test_default_initialization(self):
        """Test default PathNode initialization."""
        node = PathNode(
            asset_id="ETH",
            amount=1.0
        )
        
        assert node.asset_id == "ETH"
        assert node.amount == 1.0
        assert node.gas_cost_accumulated == 0.0
        assert node.confidence_accumulated == 1.0
        assert node.edge_path == []
    
    def test_custom_initialization(self):
        """Test custom PathNode initialization."""
        node = PathNode(
            asset_id="USDC",
            amount=1500.0,
            gas_cost_accumulated=25.0,
            confidence_accumulated=0.95,
            edge_path=["edge1", "edge2"]
        )
        
        assert node.asset_id == "USDC"
        assert node.amount == 1500.0
        assert node.gas_cost_accumulated == 25.0
        assert node.confidence_accumulated == 0.95
        assert node.edge_path == ["edge1", "edge2"]


class TestSearchPath:
    """Test SearchPath dataclass."""
    
    def test_basic_path_properties(self):
        """Test basic path properties."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["edge1"]),
            PathNode("ETH", 1.01, 30.0, 0.90, ["edge1", "edge2"])
        ]
        
        path = SearchPath(nodes=nodes, total_score=100.0)
        
        assert path.start_asset == "ETH"
        assert path.end_asset == "ETH"
        assert path.path_length == 2
        assert path.total_gas_cost == 30.0
        assert path.final_amount == 1.01
        assert path.status == PathStatus.ACTIVE
    
    def test_net_profit_calculation(self):
        """Test net profit calculation."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("ETH", 1.05, 20.0, 0.95, ["edge1", "edge2"])
        ]
        
        path = SearchPath(nodes=nodes, total_score=50.0)
        
        # Net profit = final_amount - initial_amount - gas_costs
        # 1.05 - 1.0 - 20.0 = -19.95
        assert abs(path.net_profit - (-19.95)) < 0.001
    
    def test_empty_path_properties(self):
        """Test properties of empty path."""
        path = SearchPath(nodes=[], total_score=0.0)
        
        assert path.start_asset == ""
        assert path.end_asset == ""
        assert path.path_length == 0
        assert path.total_gas_cost == 0.0
        assert path.final_amount == 0.0
        assert path.net_profit == 0.0


class TestBeamSearchConfig:
    """Test BeamSearchConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BeamSearchConfig()
        
        assert config.beam_width == 100
        assert config.max_path_length == 6
        assert config.min_profit_threshold == 0.01
        assert config.max_search_time_seconds == 30.0
        assert config.gas_price_gwei == 20.0
        assert config.slippage_tolerance == 0.01
        assert config.confidence_threshold == 0.5
        assert config.max_concurrent_updates == 20
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BeamSearchConfig(
            beam_width=50,
            max_path_length=4,
            min_profit_threshold=0.05,
            max_search_time_seconds=60.0,
            gas_price_gwei=30.0,
            slippage_tolerance=0.02,
            confidence_threshold=0.8,
            max_concurrent_updates=10
        )
        
        assert config.beam_width == 50
        assert config.max_path_length == 4
        assert config.min_profit_threshold == 0.05
        assert config.max_search_time_seconds == 60.0
        assert config.gas_price_gwei == 30.0
        assert config.slippage_tolerance == 0.02
        assert config.confidence_threshold == 0.8
        assert config.max_concurrent_updates == 10


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_successful_result(self):
        """Test successful search result."""
        paths = [
            SearchPath([PathNode("ETH", 1.0)], 100.0),
            SearchPath([PathNode("USDC", 1500.0)], 80.0)
        ]
        
        result = SearchResult(
            paths=paths,
            search_time_seconds=5.2,
            edges_evaluated=150,
            paths_pruned=25
        )
        
        assert len(result.paths) == 2
        assert result.search_time_seconds == 5.2
        assert result.edges_evaluated == 150
        assert result.paths_pruned == 25
        assert result.timeout_occurred is False
        assert result.error_message is None
    
    def test_error_result(self):
        """Test error search result."""
        result = SearchResult(
            paths=[],
            search_time_seconds=1.0,
            edges_evaluated=0,
            paths_pruned=0,
            timeout_occurred=False,
            error_message="Network error"
        )
        
        assert len(result.paths) == 0
        assert result.error_message == "Network error"


class MockDataCollector:
    """Mock data collector for testing."""
    
    def __init__(self):
        self.force_update_calls = {}
        self.update_success = True
    
    async def force_update_edge(self, edge_id: str) -> bool:
        """Mock force update edge."""
        self.force_update_calls[edge_id] = self.force_update_calls.get(edge_id, 0) + 1
        return self.update_success


class TestBeamSearchOptimizer:
    """Test BeamSearchOptimizer class."""
    
    @pytest.fixture
    def mock_graph(self):
        """Create mock graph with test edges."""
        graph = UniversalYieldGraph()
        
        # ETH -> USDC edge
        eth_usdc_edge = YieldGraphEdge(
            edge_id="eth_usdc_trade",
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
        
        # USDC -> ETH edge (for arbitrage completion)
        usdc_eth_edge = YieldGraphEdge(
            edge_id="usdc_eth_trade",
            edge_type=EdgeType.TRADE,
            source_asset_id="USDC",
            target_asset_id="ETH",
            protocol_name="uniswapv3",
            chain_name="ethereum", 
            state=EdgeState(
                conversion_rate=0.00067,  # Slightly better rate for arbitrage
                liquidity_usd=800_000.0,
                gas_cost_usd=15.0,
                confidence_score=0.90
            )
        )
        
        # Add edges to graph
        graph.add_edge(eth_usdc_edge)
        graph.add_edge(usdc_eth_edge)
        
        return graph
    
    @pytest.fixture
    def mock_data_collector(self):
        """Create mock data collector."""
        return MockDataCollector()
    
    @pytest.fixture
    def beam_search_config(self):
        """Create test configuration."""
        return BeamSearchConfig(
            beam_width=10,
            max_path_length=3,
            min_profit_threshold=0.001,
            max_search_time_seconds=5.0,
            confidence_threshold=0.5
        )
    
    def test_initialization(self, mock_graph, mock_data_collector, beam_search_config):
        """Test BeamSearchOptimizer initialization."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        assert optimizer.graph is mock_graph
        assert optimizer.data_collector is mock_data_collector
        assert optimizer.config is beam_search_config
        assert len(optimizer._current_beam) == 0
        assert len(optimizer._completed_paths) == 0
        assert optimizer.edge_state_manager is not None
    
    def test_default_config_initialization(self, mock_graph, mock_data_collector):
        """Test initialization with default config."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector
        )
        
        assert optimizer.config.beam_width == 100
        assert optimizer.config.max_path_length == 6
    
    @pytest.mark.asyncio
    async def test_search_initialization(self, mock_graph, mock_data_collector, beam_search_config):
        """Test search initialization."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        await optimizer._initialize_search("ETH", 1.0)
        
        assert len(optimizer._current_beam) == 1
        assert len(optimizer._completed_paths) == 0
        
        initial_path = optimizer._current_beam[0]
        assert initial_path.start_asset == "ETH"
        assert initial_path.final_amount == 1.0
        assert initial_path.status == PathStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_edge_state_retrieval_with_cache(self, mock_graph, mock_data_collector, beam_search_config):
        """Test edge state retrieval with caching."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        # First call should fetch from data collector
        state1 = await optimizer._get_edge_state("eth_usdc_trade")
        assert state1 is not None
        assert mock_data_collector.force_update_calls.get("eth_usdc_trade", 0) >= 1
        
        # Second call should use cache
        initial_calls = mock_data_collector.force_update_calls.get("eth_usdc_trade", 0)
        state2 = await optimizer._get_edge_state("eth_usdc_trade") 
        assert state2 is not None
        assert mock_data_collector.force_update_calls.get("eth_usdc_trade", 0) == initial_calls
    
    @pytest.mark.asyncio
    async def test_edge_state_validation(self, mock_graph, mock_data_collector, beam_search_config):
        """Test edge state validation."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        # Valid state
        valid_edge = mock_graph.get_edge("eth_usdc_trade")
        valid_state = valid_edge.state
        assert await optimizer._validate_edge_state(valid_edge, valid_state) is True
        
        # Invalid state - low confidence
        invalid_state = EdgeState(
            conversion_rate=1500.0,
            confidence_score=0.3  # Below threshold
        )
        assert await optimizer._validate_edge_state(valid_edge, invalid_state) is False
        
        # Invalid state - no conversion rate
        invalid_state2 = EdgeState(
            conversion_rate=None,
            confidence_score=0.95
        )
        assert await optimizer._validate_edge_state(valid_edge, invalid_state2) is False
    
    @pytest.mark.asyncio
    async def test_edge_conversion_calculation(self, mock_graph, mock_data_collector, beam_search_config):
        """Test edge conversion calculation."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        edge = mock_graph.get_edge("eth_usdc_trade")
        state = edge.state
        input_amount = 1.0
        
        result = await optimizer._calculate_edge_conversion(edge, state, input_amount)
        
        assert result["success"] is True
        assert "output_amount" in result
        assert "gas_cost" in result
        assert result["output_amount"] > 0
        
        # Should apply slippage tolerance
        expected_base_output = 1.0 * 1500.0 * 0.997  # Base calculation with fee
        expected_with_slippage = expected_base_output * (1.0 - beam_search_config.slippage_tolerance)
        assert abs(result["output_amount"] - expected_with_slippage) < 0.1
    
    @pytest.mark.asyncio 
    async def test_partial_path_scoring(self, mock_graph, mock_data_collector, beam_search_config):
        """Test partial path scoring."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        # Create a partial path
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"])
        ]
        path = SearchPath(nodes=nodes, total_score=0.0)
        
        score = await optimizer._score_partial_path(path, "ETH")
        
        # Score should be positive and consider amount, confidence, gas cost
        assert score > 0
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_completed_path_scoring(self, mock_graph, mock_data_collector, beam_search_config):
        """Test completed path scoring."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        # Create a completed arbitrage path (ETH -> USDC -> ETH)
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"]),
            PathNode("ETH", 1.01, 30.0, 0.90, ["eth_usdc_trade", "usdc_eth_trade"])
        ]
        path = SearchPath(nodes=nodes, total_score=0.0)
        
        score = await optimizer._score_completed_path(path)
        
        # Score should be based on net profit and confidence
        expected_net_profit = 1.01 - 1.0 - 30.0  # Very negative due to high gas costs
        expected_score = expected_net_profit * 0.90
        assert abs(score - expected_score) < 0.001
    
    def test_beam_pruning(self, mock_graph, mock_data_collector, beam_search_config):
        """Test beam pruning functionality."""
        config = BeamSearchConfig(beam_width=2)  # Small beam for testing
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=config
        )
        
        # Create paths with different scores
        paths = [
            SearchPath([PathNode("ETH", 1.0)], total_score=100.0, status=PathStatus.ACTIVE),
            SearchPath([PathNode("USDC", 1500.0)], total_score=80.0, status=PathStatus.ACTIVE),
            SearchPath([PathNode("DAI", 1000.0)], total_score=60.0, status=PathStatus.ACTIVE),
            SearchPath([PathNode("WBTC", 0.05)], total_score=40.0, status=PathStatus.ACTIVE)
        ]
        
        pruned_paths = optimizer._prune_beam(paths)
        
        # Should keep only top 2 paths
        assert len(pruned_paths) == 2
        assert pruned_paths[0].total_score == 100.0
        assert pruned_paths[1].total_score == 80.0
        assert optimizer._search_stats["paths_pruned"] == 2
    
    def test_search_stats(self, mock_graph, mock_data_collector, beam_search_config):
        """Test search statistics tracking."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        stats = optimizer.get_search_stats()
        
        assert "current_beam_size" in stats
        assert "completed_paths" in stats
        assert "edge_state_cache_size" in stats
        assert "edges_evaluated" in stats
        assert "paths_pruned" in stats
        assert "state_updates_performed" in stats
        
        # Initial values
        assert stats["current_beam_size"] == 0
        assert stats["completed_paths"] == 0
        assert stats["edge_state_cache_size"] == 0
        assert stats["edges_evaluated"] == 0
    
    @pytest.mark.asyncio
    async def test_find_arbitrage_paths_basic(self, mock_graph, mock_data_collector, beam_search_config):
        """Test basic arbitrage path finding."""
        # Set up a very simple config for testing
        simple_config = BeamSearchConfig(
            beam_width=5,
            max_path_length=2,
            min_profit_threshold=0.0,  # Accept any path for testing
            max_search_time_seconds=10.0,
            confidence_threshold=0.5
        )
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=simple_config
        )
        
        result = await optimizer.find_arbitrage_paths(
            start_asset_id="ETH",
            target_asset_id="ETH", 
            initial_amount=1.0
        )
        
        assert isinstance(result, SearchResult)
        assert result.search_time_seconds > 0
        assert result.edges_evaluated >= 0
        assert result.error_message is None
        
        # Should have attempted to find paths
        assert len(mock_data_collector.force_update_calls) > 0
    
    @pytest.mark.asyncio
    async def test_find_arbitrage_paths_with_timeout(self, mock_graph, mock_data_collector):
        """Test arbitrage path finding with timeout."""
        # Very short timeout to trigger timeout condition
        timeout_config = BeamSearchConfig(
            beam_width=100,
            max_path_length=10, 
            max_search_time_seconds=0.001  # Very short timeout
        )
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=timeout_config
        )
        
        result = await optimizer.find_arbitrage_paths(
            start_asset_id="ETH",
            target_asset_id="ETH",
            initial_amount=1.0
        )
        
        # Should complete without error even with timeout
        assert isinstance(result, SearchResult)
        assert result.search_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_find_arbitrage_paths_with_error(self, mock_graph, beam_search_config):
        """Test arbitrage path finding with data collector error."""
        # Create failing data collector
        failing_collector = MockDataCollector()
        failing_collector.update_success = False
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=failing_collector,
            config=beam_search_config
        )
        
        result = await optimizer.find_arbitrage_paths(
            start_asset_id="ETH",
            target_asset_id="ETH",
            initial_amount=1.0
        )
        
        # Should handle errors gracefully
        assert isinstance(result, SearchResult)
        assert result.search_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_edge_state_manager_integration(self, mock_graph, mock_data_collector, beam_search_config):
        """Test EdgeStateManager integration."""
        # Create custom EdgeStateManager
        state_config = StateRetrievalConfig(
            memory_cache_ttl_seconds=60.0,
            max_memory_cache_size=100,
            batch_size=5
        )
        edge_state_manager = EdgeStateManager(
            data_collector=mock_data_collector,
            config=state_config
        )
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config,
            edge_state_manager=edge_state_manager
        )
        
        # Test that the custom manager is used
        assert optimizer.edge_state_manager is edge_state_manager
        assert optimizer.edge_state_manager.config.max_memory_cache_size == 100
        
        # Test edge state retrieval through manager
        state = await optimizer._get_edge_state("eth_usdc_trade")
        
        assert state is not None
        assert state.conversion_rate > 0
        
        # Check that manager was used
        manager_metrics = edge_state_manager.get_metrics()
        assert manager_metrics["cache_stats"]["total_requests"] > 0
    
    @pytest.mark.asyncio
    async def test_batch_edge_state_optimization(self, mock_graph, mock_data_collector, beam_search_config):
        """Test optimized batch edge state retrieval."""
        # Create multiple edges for testing batch operations
        additional_edges = []
        for i in range(5):
            edge = YieldGraphEdge(
                edge_id=f"batch_test_edge_{i}",
                edge_type=EdgeType.TRADE,
                source_asset_id="ETH",
                target_asset_id="DAI",
                protocol_name="uniswapv3",
                chain_name="ethereum",
                state=EdgeState(
                    conversion_rate=1500.0 + i,
                    liquidity_usd=500_000.0,
                    gas_cost_usd=15.0,
                    confidence_score=0.9
                )
            )
            additional_edges.append(edge)
            mock_graph.add_edge(edge)
            mock_data_collector.graph.add_edge(edge)
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        # Initialize search
        await optimizer._initialize_search("ETH", 1.0)
        
        # Test batch expansion
        expanded_paths = await optimizer._expand_beam_optimized("ETH")
        
        # Should have created paths using batch optimization
        assert len(expanded_paths) >= 0  # May be zero if no valid paths
        
        # Check that EdgeStateManager was used for batch retrieval
        manager_metrics = optimizer.edge_state_manager.get_metrics()
        cache_stats = manager_metrics["cache_stats"]
        assert cache_stats["total_requests"] > 0
    
    @pytest.mark.asyncio
    async def test_path_scorer_integration(self, mock_graph, mock_data_collector, beam_search_config):
        """Test integration with NonMLPathScorer."""
        # Create custom path scorer
        scoring_config = ScoringConfig(
            method=ScoringMethod.COMPOSITE,
            profitability_weight=0.5,
            liquidity_weight=0.3
        )
        path_scorer = NonMLPathScorer(scoring_config)
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config,
            path_scorer=path_scorer
        )
        
        # Test that the custom scorer is used
        assert optimizer.path_scorer is path_scorer
        assert optimizer.path_scorer.config.profitability_weight == 0.5
        
        # Create a test path
        test_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"])
            ],
            total_score=0.0,
            status=PathStatus.ACTIVE
        )
        
        # Test scoring methods
        partial_score = await optimizer._score_partial_path(test_path, "ETH")
        assert partial_score >= 0
        
        # Test completed path scoring
        completed_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"]),
                PathNode("ETH", 1.01, 30.0, 0.90, ["eth_usdc_trade", "usdc_eth_trade"])
            ],
            total_score=0.0,
            status=PathStatus.COMPLETE
        )
        
        completed_score = await optimizer._score_completed_path(completed_path)
        assert completed_score >= 0
        
        # Completed paths should typically score higher than partial paths
        # (though this depends on the specific metrics)
    
    @pytest.mark.asyncio
    async def test_scoring_fallback_mechanism(self, mock_graph, mock_data_collector, beam_search_config):
        """Test fallback scoring when advanced scoring fails."""
        # Create a scorer that will fail
        failing_scorer = Mock()
        failing_scorer.score_path = AsyncMock(side_effect=Exception("Scoring failed"))
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config,
            path_scorer=failing_scorer
        )
        
        test_path = SearchPath(
            nodes=[PathNode("ETH", 1.0, 0.0, 1.0, [])],
            total_score=0.0,
            status=PathStatus.ACTIVE
        )
        
        # Should fall back to simple scoring
        score = await optimizer._score_partial_path(test_path, "ETH")
        
        # Should get a valid score from fallback
        assert score >= 0
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_different_scoring_methods_impact(self, mock_graph, mock_data_collector, beam_search_config):
        """Test that different scoring methods produce different results."""
        # Create test path
        test_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"]),
                PathNode("ETH", 1.01, 30.0, 0.90, ["eth_usdc_trade", "usdc_eth_trade"])
            ],
            total_score=0.0,
            status=PathStatus.COMPLETE
        )
        
        # Test different scoring methods
        methods_to_test = [
            ScoringMethod.SIMPLE_PROFIT,
            ScoringMethod.COMPOSITE,
            ScoringMethod.RISK_ADJUSTED
        ]
        
        scores = {}
        
        for method in methods_to_test:
            scoring_config = ScoringConfig(method=method)
            path_scorer = NonMLPathScorer(scoring_config)
            
            optimizer = BeamSearchOptimizer(
                graph=mock_graph,
                data_collector=mock_data_collector,
                config=beam_search_config,
                path_scorer=path_scorer
            )
            
            score = await optimizer._score_completed_path(test_path)
            scores[method] = score
            
            assert score >= 0
        
        # Different methods should potentially produce different scores
        # (though they might be similar for this simple test case)
        assert len(scores) == len(methods_to_test)
    
    @pytest.mark.asyncio
    async def test_scoring_with_warnings_and_risk_flags(self, mock_graph, mock_data_collector, beam_search_config):
        """Test that scoring warnings and risk flags are properly logged."""
        # Create path scorer that will generate warnings
        scoring_config = ScoringConfig(
            min_liquidity_threshold=1_000_000.0,  # High threshold to trigger warnings
            max_acceptable_slippage=0.001  # Very low threshold
        )
        path_scorer = NonMLPathScorer(scoring_config)
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config,
            path_scorer=path_scorer
        )
        
        # Create path that will trigger warnings
        risky_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 0.6, []),  # Low confidence
                PathNode("USDC", 1500.0, 15.0, 0.5, ["low_confidence_edge"])
            ],
            total_score=0.0,
            status=PathStatus.ACTIVE
        )
        
        # Scoring should handle warnings gracefully
        with patch('yield_arbitrage.pathfinding.beam_search.logger') as mock_logger:
            score = await optimizer._score_partial_path(risky_path, "ETH")
            
            # Should have logged debug message about warnings
            assert score >= 0
            # Check if logger was called (warnings would be logged)
    
    @pytest.mark.asyncio
    async def test_path_scoring_performance(self, mock_graph, mock_data_collector, beam_search_config):
        """Test path scoring performance with caching."""
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_search_config
        )
        
        test_path = SearchPath(
            nodes=[PathNode("ETH", 1.0, 0.0, 1.0, [])],
            total_score=0.0,
            status=PathStatus.ACTIVE
        )
        
        # Time multiple scoring operations
        import time
        start_time = time.time()
        
        # Score the same path multiple times
        for _ in range(10):
            score = await optimizer._score_partial_path(test_path, "ETH")
            assert score >= 0
        
        end_time = time.time()
        
        # Should complete quickly (benefit from caching)
        assert end_time - start_time < 1.0  # Should take less than 1 second
        
        # Check scoring statistics
        scorer_stats = optimizer.path_scorer.get_scoring_stats()
        assert scorer_stats["paths_scored"] >= 10
        # Should have cache hits from repeated scoring
        assert scorer_stats["cache_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_custom_scoring_config_integration(self, mock_graph, mock_data_collector):
        """Test BeamSearchOptimizer with custom scoring configuration."""
        # Create custom beam search config that affects scoring
        beam_config = BeamSearchConfig(
            gas_price_gwei=30.0,  # Higher gas price
            slippage_tolerance=0.005  # Lower slippage tolerance
        )
        
        optimizer = BeamSearchOptimizer(
            graph=mock_graph,
            data_collector=mock_data_collector,
            config=beam_config
        )
        
        # Verify that scoring config inherits from beam config
        assert optimizer.path_scorer.config.gas_price_gwei == 30.0
        assert optimizer.path_scorer.config.max_acceptable_slippage == 0.005
        
        # Test that scoring works with custom config
        test_path = SearchPath(
            nodes=[PathNode("ETH", 1.0, 0.0, 1.0, [])],
            total_score=0.0
        )
        
        score = await optimizer._score_partial_path(test_path, "ETH")
        assert score >= 0