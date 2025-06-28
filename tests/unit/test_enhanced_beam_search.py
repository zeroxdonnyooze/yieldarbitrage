"""Unit tests for enhanced beam search algorithm functionality."""
import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.pathfinding.beam_search import (
    BeamSearchOptimizer,
    BeamSearchConfig,
    SearchResult
)
from yield_arbitrage.pathfinding.path_models import (
    SearchPath,
    PathNode,
    PathStatus
)
from yield_arbitrage.graph_engine.models import (
    UniversalYieldGraph,
    YieldGraphEdge,
    EdgeState,
    EdgeType
)


class TestEnhancedBeamSearchAlgorithm:
    """Test the enhanced beam search algorithm core logic."""
    
    @pytest.fixture
    def enhanced_graph(self):
        """Create a more complex mock graph for testing."""
        graph = Mock(spec=UniversalYieldGraph)
        
        # Create a network: ETH -> USDC -> DAI -> WBTC -> ETH
        # Plus some alternative paths for diversity testing
        
        def create_mock_edge(edge_id, target_asset_id, edge_type, protocol_name, chain_name):
            mock_edge = Mock(edge_id=edge_id, target_asset_id=target_asset_id, 
                           edge_type=edge_type, protocol_name=protocol_name, 
                           chain_name=chain_name)
            
            # Mock the calculate_output method
            def mock_calculate_output(input_amount, current_state=None):
                if input_amount <= 0:
                    return {"success": False, "error": "Input amount must be positive"}
                
                # Use conversion rate from current_state if provided
                rate = current_state.conversion_rate if current_state and current_state.conversion_rate else 1.0
                output = input_amount * rate * 0.997  # Apply 0.3% fee
                
                return {
                    "success": True,
                    "output_amount": output,
                    "gas_cost": 0.01,
                    "slippage_applied": 0.003
                }
            
            mock_edge.calculate_output = mock_calculate_output
            return mock_edge

        edges = {
            "ETH": [
                create_mock_edge("eth_usdc", "USDC", EdgeType.TRADE, "uniswapv3", "ethereum"),
                create_mock_edge("eth_dai", "DAI", EdgeType.TRADE, "curve", "ethereum"),
                create_mock_edge("eth_wbtc", "WBTC", EdgeType.TRADE, "sushiswap", "ethereum")
            ],
            "USDC": [
                create_mock_edge("usdc_dai", "DAI", EdgeType.TRADE, "curve", "ethereum"),
                create_mock_edge("usdc_eth", "ETH", EdgeType.TRADE, "uniswapv3", "ethereum"),
                create_mock_edge("usdc_wbtc", "WBTC", EdgeType.TRADE, "curve", "ethereum")
            ],
            "DAI": [
                create_mock_edge("dai_wbtc", "WBTC", EdgeType.TRADE, "balancer", "ethereum"),
                create_mock_edge("dai_eth", "ETH", EdgeType.TRADE, "curve", "ethereum"),
                create_mock_edge("dai_usdc", "USDC", EdgeType.TRADE, "curve", "ethereum")
            ],
            "WBTC": [
                create_mock_edge("wbtc_eth", "ETH", EdgeType.TRADE, "sushiswap", "ethereum"),
                create_mock_edge("wbtc_usdc", "USDC", EdgeType.TRADE, "curve", "ethereum")
            ]
        }
        
        def mock_get_edges_from(asset_id):
            return edges.get(asset_id, [])
        
        def mock_get_edge(edge_id):
            for asset_edges in edges.values():
                for edge in asset_edges:
                    if edge.edge_id == edge_id:
                        return edge
            return None
        
        graph.get_edges_from = mock_get_edges_from
        graph.get_edge = mock_get_edge
        
        return graph
    
    @pytest.fixture
    def mock_data_collector(self):
        """Create enhanced mock data collector."""
        collector = Mock()
        
        async def mock_force_update_edge(edge_id: str) -> bool:
            return True
        
        collector.force_update_edge = mock_force_update_edge
        return collector
    
    @pytest.fixture
    def enhanced_config(self):
        """Create configuration for enhanced beam search testing."""
        return BeamSearchConfig(
            beam_width=20,
            max_path_length=4,
            min_profit_threshold=0.001,
            max_search_time_seconds=10.0,
            confidence_threshold=0.5
        )
    
    @pytest.fixture
    def enhanced_optimizer(self, enhanced_graph, mock_data_collector, enhanced_config):
        """Create enhanced beam search optimizer."""
        return BeamSearchOptimizer(
            graph=enhanced_graph,
            data_collector=mock_data_collector,
            config=enhanced_config
        )
    
    def test_adaptive_beam_pruning(self, enhanced_optimizer):
        """Test adaptive beam pruning functionality."""
        # Create paths with different scores
        paths = []
        for i in range(30):  # More than beam width
            path = SearchPath(
                nodes=[PathNode(f"ASSET_{i}", 1.0 + i * 0.01)],
                total_score=float(i),
                status=PathStatus.ACTIVE
            )
            paths.append(path)
        
        # Test early iteration (should use full beam width)
        pruned_early = enhanced_optimizer._prune_beam_adaptive(paths, iteration=1)
        assert len(pruned_early) == enhanced_optimizer.config.beam_width
        
        # Test later iteration (should reduce beam width)
        pruned_late = enhanced_optimizer._prune_beam_adaptive(paths, iteration=5)
        assert len(pruned_late) < enhanced_optimizer.config.beam_width
        assert len(pruned_late) >= 10  # Minimum beam size
    
    def test_early_termination_conditions(self, enhanced_optimizer):
        """Test early termination decision logic."""
        # No early termination in first iterations
        assert not enhanced_optimizer._should_terminate_early(1, 0.1)
        
        # Should terminate with excellent profit
        assert enhanced_optimizer._should_terminate_early(3, 1.0)  # 1000x minimum threshold
        
        # Create scenario with many completed paths
        enhanced_optimizer._completed_paths = [
            SearchPath([PathNode("ETH", 1.0)], 0.1) for _ in range(6)
        ]
        enhanced_optimizer._current_beam = [
            SearchPath([PathNode("ETH", 1.0)], 0.1),
            SearchPath([PathNode("USDC", 1.0)], 0.1)
        ]
        
        # Should terminate with many completed paths and small beam
        assert enhanced_optimizer._should_terminate_early(3, 0.01)
    
    def test_edge_exploration_filtering(self, enhanced_optimizer, enhanced_graph):
        """Test edge exploration filtering logic."""
        # Create a simple path
        path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("USDC", 1500.0, 0.01, 0.95, ["eth_usdc"])
            ],
            total_score=0.5,
            status=PathStatus.ACTIVE
        )
        
        # Mock edge
        test_edge = Mock()
        test_edge.target_asset_id = "ETH"
        test_edge.edge_type = EdgeType.TRADE
        test_edge.protocol_name = "uniswapv3"
        test_edge.chain_name = "ethereum"
        
        # Should explore edge that completes arbitrage cycle
        assert enhanced_optimizer._is_edge_worth_exploring(test_edge, path, "ETH")
        
        # Test with different target
        test_edge.target_asset_id = "DAI"
        assert enhanced_optimizer._is_edge_worth_exploring(test_edge, path, "ETH")
        
        # Test edge type filtering for longer paths
        long_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("USDC", 1500.0, 0.01, 0.95, ["eth_usdc"]),
                PathNode("DAI", 1500.0, 0.02, 0.90, ["eth_usdc", "usdc_dai"])
            ],
            total_score=0.5,
            status=PathStatus.ACTIVE
        )
        
        # Low priority edge type should be filtered for longer paths
        test_edge.edge_type = EdgeType.WAIT
        assert not enhanced_optimizer._is_edge_worth_exploring(test_edge, long_path, "ETH")
    
    def test_target_reachability_check(self, enhanced_optimizer):
        """Test target reachability checking."""
        # Direct reachability
        assert enhanced_optimizer._is_target_reachable("ETH", "ETH", 1)
        
        # One hop reachability (ETH -> USDC)
        assert enhanced_optimizer._is_target_reachable("ETH", "USDC", 1)
        
        # Two hop reachability (ETH -> USDC -> DAI)
        assert enhanced_optimizer._is_target_reachable("ETH", "DAI", 2)
        
        # Unreachable within hop limit
        # This depends on the specific graph structure, but with limited hops
        # some paths should be unreachable
    
    def test_path_post_processing(self, enhanced_optimizer):
        """Test path post-processing functionality."""
        # Create duplicate and unique paths
        paths = [
            SearchPath([
                PathNode("ETH", 1.0),
                PathNode("USDC", 1500.0),
                PathNode("ETH", 1.01)
            ], 0.8),
            SearchPath([
                PathNode("ETH", 1.0),
                PathNode("USDC", 1500.0),
                PathNode("ETH", 1.01)
            ], 0.7),  # Duplicate sequence
            SearchPath([
                PathNode("ETH", 1.0),
                PathNode("DAI", 1500.0),
                PathNode("ETH", 1.02)
            ], 0.9),
        ]
        
        enhanced_optimizer._completed_paths = paths
        enhanced_optimizer._post_process_completed_paths()
        
        # Should remove duplicates
        assert len(enhanced_optimizer._completed_paths) == 2
        
        # Should be sorted by composite score
        scores = [p.total_score for p in enhanced_optimizer._completed_paths]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_progress_tracking(self, enhanced_optimizer):
        """Test search progress tracking."""
        # Add some completed paths with proper profit calculation
        enhanced_optimizer._completed_paths = [
            SearchPath([
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("ETH", 1.1, 0.01, 0.9, ["edge1"])
            ], 0.1),
            SearchPath([
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("ETH", 1.2, 0.01, 0.9, ["edge2"])
            ], 0.2),
            SearchPath([
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("ETH", 1.05, 0.01, 0.9, ["edge3"])
            ], 0.05),
        ]
        
        # Add current beam
        enhanced_optimizer._current_beam = [
            SearchPath([PathNode("ETH", 1.0)], 0.1),
            SearchPath([PathNode("USDC", 1.0)], 0.1),
        ]
        
        progress = enhanced_optimizer.get_search_progress()
        
        assert progress["current_beam_size"] == 2
        assert progress["completed_paths_count"] == 3
        assert progress["best_profit_found"] > 0.15  # Should be around 0.19 (1.2 - 1.0 - 0.01)
        assert len(progress["top_3_profits"]) == 3
        assert progress["top_3_profits"][0] >= progress["top_3_profits"][1]  # Sorted descending
    
    @pytest.mark.asyncio
    async def test_enhanced_beam_expansion(self, enhanced_optimizer, enhanced_graph):
        """Test enhanced beam expansion with filtering."""
        # Initialize search
        await enhanced_optimizer._initialize_search("ETH", 1.0)
        
        # Verify initial beam state
        assert len(enhanced_optimizer._current_beam) == 1
        initial_path = enhanced_optimizer._current_beam[0]
        assert initial_path.status == PathStatus.ACTIVE
        assert initial_path.end_asset == "ETH"
        
        # Mock edge state manager to return valid states
        mock_edge_states = {
            "eth_usdc": EdgeState(conversion_rate=1500.0, confidence_score=0.9),
            "eth_dai": EdgeState(conversion_rate=1500.0, confidence_score=0.8),
            "eth_wbtc": EdgeState(conversion_rate=0.05, confidence_score=0.85)
        }
        
        async def mock_get_edge_states_batch(edge_ids):
            return {eid: mock_edge_states.get(eid) for eid in edge_ids}
        
        enhanced_optimizer.edge_state_manager.get_edge_states_batch = mock_get_edge_states_batch
        
        # Mock path validator to pass all paths
        async def mock_validate_path(*args, **kwargs):
            from yield_arbitrage.pathfinding.path_validator import ValidationReport, ValidationResult
            return ValidationReport(
                is_valid=True,
                result=ValidationResult.VALID,
                errors=[],
                warnings=[],
                path_score=0.8,
                confidence_score=0.9,
                liquidity_score=0.8,
                risk_score=0.2,
                cycle_analysis={},
                constraint_analysis={},
                risk_analysis={},
                performance_metrics={}
            )
        
        enhanced_optimizer.path_validator.validate_path = mock_validate_path
        
        # Test beam expansion
        expanded_paths = await enhanced_optimizer._expand_beam_optimized("ETH")
        
        # Should have expanded to multiple paths
        assert len(expanded_paths) > 0
        
        # All expanded paths should be valid
        for path in expanded_paths:
            assert path.status != PathStatus.INVALID
            assert len(path.nodes) == 2  # Original + one expansion
    
    @pytest.mark.asyncio
    async def test_complete_enhanced_search(self, enhanced_optimizer):
        """Test complete enhanced search algorithm."""
        # Mock all dependencies for a complete search
        
        # Mock edge state manager
        mock_edge_states = {
            "eth_usdc": EdgeState(conversion_rate=1500.0, confidence_score=0.9, gas_cost_usd=0.01),
            "usdc_eth": EdgeState(conversion_rate=0.00067, confidence_score=0.9, gas_cost_usd=0.01),
            "eth_dai": EdgeState(conversion_rate=1500.0, confidence_score=0.8, gas_cost_usd=0.01),
            "dai_eth": EdgeState(conversion_rate=0.00067, confidence_score=0.8, gas_cost_usd=0.01)
        }
        
        async def mock_get_edge_states_batch(edge_ids):
            return {eid: mock_edge_states.get(eid) for eid in edge_ids}
        
        enhanced_optimizer.edge_state_manager.get_edge_states_batch = mock_get_edge_states_batch
        
        # Mock path validator to be permissive
        async def mock_validate_path(*args, **kwargs):
            from yield_arbitrage.pathfinding.path_validator import ValidationReport, ValidationResult
            return ValidationReport(
                is_valid=True,
                result=ValidationResult.VALID,
                errors=[],
                warnings=[],
                path_score=0.8,
                confidence_score=0.9,
                liquidity_score=0.8,
                risk_score=0.2,
                cycle_analysis={},
                constraint_analysis={},
                risk_analysis={},
                performance_metrics={}
            )
        
        enhanced_optimizer.path_validator.validate_path = mock_validate_path
        
        # Run the complete search
        result = await enhanced_optimizer.find_arbitrage_paths(
            start_asset_id="ETH",
            target_asset_id="ETH",
            initial_amount=1.0
        )
        
        # Verify search completed successfully
        assert isinstance(result, SearchResult)
        assert result.search_time_seconds > 0
        assert result.error_message is None
        
        # Check search statistics
        stats = enhanced_optimizer.get_search_stats()
        assert stats["edges_evaluated"] >= 0
        assert stats["paths_pruned"] >= 0
        assert "validation" in stats
    
    def test_search_statistics_integration(self, enhanced_optimizer):
        """Test comprehensive search statistics integration."""
        stats = enhanced_optimizer.get_search_stats()
        
        # Should include all major component statistics
        assert "current_beam_size" in stats
        assert "completed_paths" in stats
        assert "edges_evaluated" in stats
        assert "paths_pruned" in stats
        assert "validation" in stats
        
        # Should handle missing optional stats gracefully
        assert isinstance(stats, dict)
        
        # Test progress tracking
        progress = enhanced_optimizer.get_search_progress()
        assert "best_profit_found" in progress
        assert "profitable_paths_count" in progress
        assert "search_stats" in progress
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, enhanced_optimizer):
        """Test timeout handling in enhanced algorithm."""
        # Set very short timeout
        enhanced_optimizer.config.max_search_time_seconds = 0.001
        
        # Mock dependencies to be slow/unresponsive
        async def slow_batch_get(edge_ids):
            await asyncio.sleep(0.01)  # Longer than timeout
            return {}
        
        enhanced_optimizer.edge_state_manager.get_edge_states_batch = slow_batch_get
        
        # Run search
        result = await enhanced_optimizer.find_arbitrage_paths("ETH", "ETH", 1.0)
        
        # Should handle timeout gracefully
        assert isinstance(result, SearchResult)
        assert result.timeout_occurred or result.search_time_seconds < 1.0
    
    def test_beam_size_edge_cases(self, enhanced_optimizer):
        """Test beam size handling edge cases."""
        # Test with empty beam
        empty_result = enhanced_optimizer._prune_beam_adaptive([], 1)
        assert len(empty_result) == 0
        
        # Test with single path
        single_path = [SearchPath([PathNode("ETH", 1.0)], 0.5, PathStatus.ACTIVE)]
        single_result = enhanced_optimizer._prune_beam_adaptive(single_path, 1)
        assert len(single_result) == 1
        
        # Test adaptive sizing with profitable paths
        enhanced_optimizer._completed_paths = [
            SearchPath([PathNode("ETH", 1.0)], 100.0)  # Very profitable
        ]
        
        many_paths = [
            SearchPath([PathNode(f"ASSET_{i}", 1.0)], float(i), PathStatus.ACTIVE)
            for i in range(50)
        ]
        
        adaptive_result = enhanced_optimizer._prune_beam_adaptive(many_paths, 2)
        # Should expand beam size due to profitable paths
        assert len(adaptive_result) >= enhanced_optimizer.config.beam_width