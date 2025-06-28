"""
Unit tests for the Path Segment Analyzer.

Tests various path configurations to ensure correct segmentation based on
edge execution properties and constraints.
"""
import pytest
from typing import List
from unittest.mock import MagicMock

from yield_arbitrage.pathfinding.path_segment_analyzer import (
    PathSegmentAnalyzer, PathSegment, SegmentType, SegmentBoundary
)
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState
)


def create_test_edge(
    edge_id: str,
    source_asset: str,
    target_asset: str,
    edge_type: EdgeType = EdgeType.TRADE,
    chain: str = "ethereum",
    supports_synchronous: bool = True,
    requires_time_delay: int = None,
    requires_bridge: bool = False,
    requires_capital_holding: bool = False,
    gas_estimate: int = 100000,
    mev_sensitivity: float = 0.3,
    min_liquidity: float = 10000.0
) -> YieldGraphEdge:
    """Create a test edge with specified properties."""
    return YieldGraphEdge(
        edge_id=edge_id,
        source_asset_id=source_asset,
        target_asset_id=target_asset,
        edge_type=edge_type,
        protocol_name="test_protocol",
        chain_name=chain,
        execution_properties=EdgeExecutionProperties(
            supports_synchronous=supports_synchronous,
            requires_time_delay=requires_time_delay,
            requires_bridge=requires_bridge,
            requires_capital_holding=requires_capital_holding,
            gas_estimate=gas_estimate,
            mev_sensitivity=mev_sensitivity,
            min_liquidity_required=min_liquidity
        ),
        constraints=EdgeConstraints(),
        state=EdgeState()
    )


class TestPathSegmentAnalyzer:
    """Test suite for PathSegmentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a PathSegmentAnalyzer instance."""
        return PathSegmentAnalyzer(max_segment_gas=5_000_000, max_mev_sensitivity=0.7)
    
    def test_empty_path(self, analyzer):
        """Test analyzing an empty path."""
        segments = analyzer.analyze_path([])
        assert len(segments) == 0
    
    def test_single_edge_path(self, analyzer):
        """Test analyzing a path with a single edge."""
        edge = create_test_edge("edge1", "USDC", "WETH")
        segments = analyzer.analyze_path([edge])
        
        assert len(segments) == 1
        assert segments[0].segment_type == SegmentType.ATOMIC
        assert segments[0].edge_count == 1
        assert segments[0].get_input_asset() == "USDC"
        assert segments[0].get_output_asset() == "WETH"
    
    def test_simple_atomic_path(self, analyzer):
        """Test a simple path that can be executed atomically."""
        edges = [
            create_test_edge("edge1", "USDC", "WETH"),
            create_test_edge("edge2", "WETH", "DAI"),
            create_test_edge("edge3", "DAI", "USDC")
        ]
        segments = analyzer.analyze_path(edges)
        
        assert len(segments) == 1
        assert segments[0].segment_type == SegmentType.ATOMIC
        assert segments[0].edge_count == 3
        assert segments[0].is_atomic
        assert not segments[0].requires_flash_loan
    
    def test_time_delayed_segmentation(self, analyzer):
        """Test segmentation with time delay requirement."""
        edges = [
            create_test_edge("edge1", "USDC", "WETH"),
            create_test_edge("edge2", "WETH", "stETH", 
                           edge_type=EdgeType.STAKE, requires_time_delay=3600),
            create_test_edge("edge3", "stETH", "USDC")
        ]
        segments = analyzer.analyze_path(edges)
        
        assert len(segments) == 3
        assert segments[0].segment_type == SegmentType.ATOMIC
        assert segments[1].segment_type == SegmentType.TIME_DELAYED
        assert segments[1].requires_delay_seconds == 3600
        assert segments[2].segment_type == SegmentType.ATOMIC
    
    def test_bridge_segmentation(self, analyzer):
        """Test segmentation with bridge requirement."""
        edges = [
            create_test_edge("edge1", "ETH_USDC", "ETH_WETH", chain="ethereum"),
            create_test_edge("edge2", "ETH_WETH", "ARB_WETH", 
                           edge_type=EdgeType.BRIDGE, chain="arbitrum", requires_bridge=True),
            create_test_edge("edge3", "ARB_WETH", "ARB_USDC", chain="arbitrum")
        ]
        segments = analyzer.analyze_path(edges)
        
        assert len(segments) == 3
        assert segments[0].segment_type == SegmentType.ATOMIC
        assert segments[1].segment_type == SegmentType.BRIDGED
        assert segments[2].segment_type == SegmentType.ATOMIC
        assert segments[0].source_chain == "ethereum"
        assert segments[2].source_chain == "arbitrum"
    
    def test_gas_limit_segmentation(self, analyzer):
        """Test segmentation due to gas limit."""
        # Create edges with high gas usage
        edges = []
        for i in range(10):
            edges.append(create_test_edge(
                f"edge{i}", 
                f"TOKEN{i}", 
                f"TOKEN{i+1}",
                gas_estimate=600_000  # High gas per edge
            ))
        
        segments = analyzer.analyze_path(edges)
        
        # Should split into multiple segments due to gas limit
        assert len(segments) > 1
        for segment in segments:
            assert segment.max_gas_estimate <= analyzer.max_segment_gas
    
    def test_mev_sensitivity_segmentation(self, analyzer):
        """Test segmentation due to MEV sensitivity."""
        edges = [
            create_test_edge("edge1", "USDC", "WETH", mev_sensitivity=0.3),
            create_test_edge("edge2", "WETH", "RARE_TOKEN", mev_sensitivity=0.6),
            create_test_edge("edge3", "RARE_TOKEN", "USDC", mev_sensitivity=0.5)
        ]
        
        segments = analyzer.analyze_path(edges)
        
        # Should create segments based on MEV sensitivity
        assert all(s.total_mev_sensitivity <= analyzer.max_mev_sensitivity for s in segments)
    
    def test_flash_loan_detection(self, analyzer):
        """Test flash loan requirement detection."""
        edges = [
            create_test_edge("flash", "USDC", "USDC_LOAN", 
                           edge_type=EdgeType.FLASH_LOAN, min_liquidity=1_000_000),
            create_test_edge("edge1", "USDC_LOAN", "WETH"),
            create_test_edge("edge2", "WETH", "USDC")
        ]
        
        segments = analyzer.analyze_path(edges)
        
        assert len(segments) == 1
        assert segments[0].segment_type == SegmentType.FLASH_LOAN_ATOMIC
        assert segments[0].requires_flash_loan
        assert segments[0].flash_loan_asset == "USDC"
    
    def test_capital_holding_segmentation(self, analyzer):
        """Test segmentation with capital holding requirement."""
        edges = [
            create_test_edge("edge1", "USDC", "WETH"),
            create_test_edge("edge2", "WETH", "aWETH", 
                           edge_type=EdgeType.LEND, requires_capital_holding=True),
            create_test_edge("edge3", "aWETH", "USDC")
        ]
        
        segments = analyzer.analyze_path(edges)
        
        assert len(segments) == 3
        assert segments[1].segment_type == SegmentType.CAPITAL_HOLDING
    
    def test_complex_mixed_path(self, analyzer):
        """Test complex path with multiple segment types."""
        edges = [
            # Initial atomic segment
            create_test_edge("edge1", "USDC", "WETH"),
            create_test_edge("edge2", "WETH", "DAI"),
            
            # Time delay
            create_test_edge("edge3", "DAI", "sDAI", 
                           edge_type=EdgeType.STAKE, requires_time_delay=7200),
            
            # Bridge
            create_test_edge("edge4", "sDAI", "ARB_sDAI", 
                           edge_type=EdgeType.BRIDGE, chain="arbitrum", requires_bridge=True),
            
            # Final atomic segment on Arbitrum
            create_test_edge("edge5", "ARB_sDAI", "ARB_USDC", chain="arbitrum"),
            create_test_edge("edge6", "ARB_USDC", "ARB_WETH", chain="arbitrum")
        ]
        
        segments = analyzer.analyze_path(edges)
        
        # Should have 4 segments based on boundaries
        assert len(segments) == 4
        assert segments[0].segment_type == SegmentType.ATOMIC
        assert any(s.segment_type == SegmentType.TIME_DELAYED for s in segments)
        assert any(s.segment_type == SegmentType.BRIDGED for s in segments)
    
    def test_segment_connectivity_validation(self, analyzer):
        """Test segment connectivity validation."""
        edges = [
            create_test_edge("edge1", "USDC", "WETH"),
            create_test_edge("edge2", "WETH", "DAI"),
            create_test_edge("edge3", "DAI", "USDC")
        ]
        
        segments = analyzer.analyze_path(edges)
        assert analyzer.validate_segment_connectivity(segments)
    
    def test_non_synchronous_edge_segmentation(self, analyzer):
        """Test segmentation with non-synchronous edges."""
        edges = [
            create_test_edge("edge1", "USDC", "WETH"),
            create_test_edge("edge2", "WETH", "YIELD_TOKEN", 
                           supports_synchronous=False),
            create_test_edge("edge3", "YIELD_TOKEN", "USDC")
        ]
        
        segments = analyzer.analyze_path(edges)
        
        # Non-synchronous edge should create a boundary
        assert len(segments) > 1
    
    def test_segment_summary(self, analyzer):
        """Test segment summary generation."""
        edge = create_test_edge("edge1", "USDC", "WETH", gas_estimate=150000)
        segments = analyzer.analyze_path([edge])
        
        summary = analyzer.get_segment_summary(segments[0])
        
        assert summary["segment_id"] == "seg_0"
        assert summary["type"] == SegmentType.ATOMIC
        assert summary["edge_count"] == 1
        assert summary["is_atomic"] is True
        assert summary["input_asset"] == "USDC"
        assert summary["output_asset"] == "WETH"
        assert summary["gas_estimate"] == 150000
    
    def test_statistics_tracking(self, analyzer):
        """Test statistics tracking."""
        # Create various types of paths
        paths = [
            [create_test_edge("e1", "A", "B")],  # Simple atomic
            [create_test_edge("e2", "B", "C", requires_time_delay=3600)],  # Time delayed
            [create_test_edge("e3", "C", "D", requires_bridge=True)]  # Bridged
        ]
        
        for path in paths:
            analyzer.analyze_path(path)
        
        stats = analyzer.get_statistics()
        
        assert stats["paths_analyzed"] == 3
        assert stats["segments_created"] == 3
        assert stats["atomic_segments"] == 1
        assert stats["delayed_segments"] == 1
        assert stats["bridged_segments"] == 1
    
    def test_high_liquidity_flash_loan_detection(self, analyzer):
        """Test flash loan detection based on liquidity requirements."""
        edges = [
            create_test_edge("edge1", "USDC", "WETH", min_liquidity=500_000),
            create_test_edge("edge2", "WETH", "USDC")
        ]
        
        segments = analyzer.analyze_path(edges)
        
        assert segments[0].requires_flash_loan
        assert segments[0].segment_type == SegmentType.FLASH_LOAN_ATOMIC
        assert segments[0].flash_loan_amount == 500_000


@pytest.mark.asyncio
async def test_segment_analyzer_integration():
    """Integration test with real edge data structures."""
    analyzer = PathSegmentAnalyzer()
    
    # Create a realistic arbitrage path
    edges = [
        create_test_edge("uni_swap_1", "ETH_USDC", "ETH_WETH", 
                       edge_type=EdgeType.TRADE, gas_estimate=120_000),
        create_test_edge("curve_swap", "ETH_WETH", "ETH_stETH", 
                       edge_type=EdgeType.TRADE, gas_estimate=180_000),
        create_test_edge("lido_stake", "ETH_stETH", "ETH_wstETH", 
                       edge_type=EdgeType.STAKE, gas_estimate=90_000),
        create_test_edge("uni_swap_2", "ETH_wstETH", "ETH_USDC", 
                       edge_type=EdgeType.TRADE, gas_estimate=120_000)
    ]
    
    segments = analyzer.analyze_path(edges)
    
    # Should create one atomic segment since all edges support synchronous execution
    assert len(segments) == 1
    assert segments[0].is_atomic
    assert segments[0].max_gas_estimate == 510_000  # Sum of all gas estimates
    
    # Validate the complete path
    assert analyzer.validate_segment_connectivity(segments)