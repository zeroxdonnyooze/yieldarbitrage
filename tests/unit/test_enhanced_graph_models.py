"""Unit tests for enhanced graph models with execution properties."""

import pytest
from yield_arbitrage.graph_engine.models import (
    EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState,
    YieldGraphEdge, FlashLoanEdge, BackRunEdge, UniversalYieldGraph
)


class TestEdgeExecutionProperties:
    """Test EdgeExecutionProperties functionality."""
    
    def test_default_properties(self):
        """Test default execution properties."""
        props = EdgeExecutionProperties()
        
        assert props.supports_synchronous is True
        assert props.requires_time_delay is None
        assert props.requires_bridge is False
        assert props.requires_capital_holding is False
        assert props.max_slippage == 0.05
        assert props.mev_sensitivity == 0.5
        assert props.supports_private_mempool is True
        assert props.gas_estimate == 100000
        assert props.requires_approval is True
        assert props.min_liquidity_required == 10000.0
        assert props.max_impact_allowed == 0.01
    
    def test_custom_properties(self):
        """Test custom execution properties."""
        props = EdgeExecutionProperties(
            supports_synchronous=False,
            requires_time_delay=3600,
            requires_capital_holding=True,
            mev_sensitivity=0.8,
            gas_estimate=200000
        )
        
        assert props.supports_synchronous is False
        assert props.requires_time_delay == 3600
        assert props.requires_capital_holding is True
        assert props.mev_sensitivity == 0.8
        assert props.gas_estimate == 200000
    
    def test_validation(self):
        """Test property validation."""
        # Test invalid mev_sensitivity
        with pytest.raises(ValueError):
            EdgeExecutionProperties(mev_sensitivity=1.5)
        
        with pytest.raises(ValueError):
            EdgeExecutionProperties(mev_sensitivity=-0.1)
        
        # Test invalid slippage
        with pytest.raises(ValueError):
            EdgeExecutionProperties(max_slippage=1.5)
        
        # Test negative gas estimate
        with pytest.raises(ValueError):
            EdgeExecutionProperties(gas_estimate=-1000)


class TestEnhancedYieldGraphEdge:
    """Test enhanced YieldGraphEdge with execution properties."""
    
    def test_edge_with_default_execution_properties(self):
        """Test edge creation with default execution properties."""
        edge = YieldGraphEdge(
            edge_id="ETH_MAINNET_UNISWAPV3_TRADE_WETH_USDC",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="UniswapV3",
            chain_name="ethereum"
        )
        
        assert edge.execution_properties.supports_synchronous is True
        assert edge.execution_properties.mev_sensitivity == 0.5
        assert edge.execution_properties.gas_estimate == 100000
    
    def test_edge_with_custom_execution_properties(self):
        """Test edge creation with custom execution properties."""
        custom_props = EdgeExecutionProperties(
            supports_synchronous=False,
            requires_time_delay=86400,  # 24 hours
            requires_capital_holding=True,
            mev_sensitivity=0.2
        )
        
        edge = YieldGraphEdge(
            edge_id="ETH_MAINNET_AAVE_LEND_USDC",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_AUSDC",
            edge_type=EdgeType.LEND,
            protocol_name="AaveV3",
            chain_name="ethereum",
            execution_properties=custom_props
        )
        
        assert edge.execution_properties.supports_synchronous is False
        assert edge.execution_properties.requires_time_delay == 86400
        assert edge.execution_properties.requires_capital_holding is True
        assert edge.execution_properties.mev_sensitivity == 0.2
    
    def test_new_edge_types(self):
        """Test new edge types FLASH_LOAN and BACK_RUN."""
        assert EdgeType.FLASH_LOAN == "FLASH_LOAN"
        assert EdgeType.BACK_RUN == "BACK_RUN"
        
        # Test creating edges with new types
        flash_edge = YieldGraphEdge(
            edge_id="ETH_AAVE_FLASH_LOAN_WETH",
            source_asset_id="ETH_FLASH_WETH",
            target_asset_id="ETH_WETH",
            edge_type=EdgeType.FLASH_LOAN,
            protocol_name="aave",
            chain_name="ethereum"
        )
        
        backrun_edge = YieldGraphEdge(
            edge_id="ETH_BACKRUN_0x123_WETH_USDC",
            source_asset_id="ETH_WETH",
            target_asset_id="ETH_USDC",
            edge_type=EdgeType.BACK_RUN,
            protocol_name="MEV_BACKRUN",
            chain_name="ethereum"
        )
        
        assert flash_edge.edge_type == EdgeType.FLASH_LOAN
        assert backrun_edge.edge_type == EdgeType.BACK_RUN


class TestFlashLoanEdge:
    """Test FlashLoanEdge special edge type."""
    
    def test_flash_loan_edge_creation(self):
        """Test flash loan edge creation."""
        flash_edge = FlashLoanEdge(
            chain_name="ethereum",
            provider="aave",
            asset="WETH",
            max_amount=1000.0
        )
        
        assert flash_edge.edge_type == EdgeType.FLASH_LOAN
        assert flash_edge.provider == "aave"
        assert flash_edge.asset == "WETH"
        assert flash_edge.max_amount == 1000.0
        assert flash_edge.fee_percentage == 0.0009  # Default fee
        
        # Check execution properties
        assert flash_edge.execution_properties.supports_synchronous is True
        assert flash_edge.execution_properties.requires_capital_holding is False
        assert flash_edge.execution_properties.mev_sensitivity == 0.3
        assert flash_edge.execution_properties.requires_approval is False
    
    def test_flash_loan_edge_with_custom_fee(self):
        """Test flash loan edge with custom fee."""
        flash_edge = FlashLoanEdge(
            chain_name="ethereum",
            provider="balancer",
            asset="USDC",
            max_amount=500000.0,
            fee_percentage=0.0
        )
        
        assert flash_edge.provider == "balancer"
        assert flash_edge.fee_percentage == 0.0
    
    def test_flash_loan_calculation(self):
        """Test flash loan output calculation."""
        flash_edge = FlashLoanEdge(
            chain_name="ethereum",
            provider="aave",
            asset="WETH",
            max_amount=1000.0,
            fee_percentage=0.001  # 0.1%
        )
        
        # Test normal calculation
        result = flash_edge.calculate_output(100.0)
        assert result["output_amount"] == 100.0
        assert result["flash_loan_fee"] == 0.1  # 100 * 0.001
        assert result["repayment_required"] == 100.1
        assert result["effective_rate"] == 1.0
        assert "gas_cost_usd" in result
    
    def test_flash_loan_max_amount_validation(self):
        """Test flash loan max amount validation."""
        flash_edge = FlashLoanEdge(
            chain_name="ethereum",
            provider="aave",
            asset="WETH",
            max_amount=1000.0
        )
        
        # Test exceeding max amount
        result = flash_edge.calculate_output(2000.0)
        assert result["output_amount"] == 0.0
        assert "error" in result
        assert "exceeds max flash loan" in result["error"]
    
    def test_flash_loan_invalid_input(self):
        """Test flash loan with invalid input."""
        flash_edge = FlashLoanEdge(
            chain_name="ethereum",
            provider="aave",
            asset="WETH",
            max_amount=1000.0
        )
        
        # Test negative input
        result = flash_edge.calculate_output(-100.0)
        assert result["output_amount"] == 0.0
        assert "error" in result


class TestBackRunEdge:
    """Test BackRunEdge special edge type."""
    
    def test_backrun_edge_creation(self):
        """Test back-run edge creation."""
        backrun_edge = BackRunEdge(
            chain_name="ethereum",
            target_transaction="0x123456789abcdef",
            source_asset="WETH",
            target_asset="USDC",
            expected_profit=50.0
        )
        
        assert backrun_edge.edge_type == EdgeType.BACK_RUN
        assert backrun_edge.target_transaction == "0x123456789abcdef"
        assert backrun_edge.expected_profit == 50.0
        assert backrun_edge.protocol_name == "MEV_BACKRUN"
        
        # Check execution properties
        assert backrun_edge.execution_properties.supports_synchronous is True
        assert backrun_edge.execution_properties.mev_sensitivity == 0.0
        assert backrun_edge.execution_properties.requires_capital_holding is False
    
    def test_backrun_calculation(self):
        """Test back-run output calculation."""
        backrun_edge = BackRunEdge(
            chain_name="ethereum",
            target_transaction="0x123456789abcdef",
            source_asset="WETH",
            target_asset="USDC",
            expected_profit=100.0
        )
        
        result = backrun_edge.calculate_output(1000.0)
        assert result["output_amount"] > 1000.0  # Should include profit
        assert result["expected_profit_usd"] == 100.0
        assert result["target_transaction"] == "0x123456789abcdef"
        assert result["confidence"] == 0.7
    
    def test_backrun_invalid_input(self):
        """Test back-run with invalid input."""
        backrun_edge = BackRunEdge(
            chain_name="ethereum",
            target_transaction="0x123456789abcdef",
            source_asset="WETH",
            target_asset="USDC",
            expected_profit=50.0
        )
        
        result = backrun_edge.calculate_output(-100.0)
        assert result["output_amount"] == 0.0
        assert "error" in result


class TestUniversalYieldGraphWithEnhancedEdges:
    """Test UniversalYieldGraph with enhanced edges."""
    
    def test_graph_with_enhanced_edges(self):
        """Test graph operations with enhanced edges."""
        graph = UniversalYieldGraph()
        
        # Add regular trade edge
        trade_edge = YieldGraphEdge(
            edge_id="ETH_UNISWAP_TRADE_WETH_USDC",
            source_asset_id="ETH_WETH",
            target_asset_id="ETH_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="UniswapV3",
            chain_name="ethereum"
        )
        
        # Add flash loan edge
        flash_edge = FlashLoanEdge(
            chain_name="ethereum",
            provider="aave",
            asset="WETH",
            max_amount=1000.0
        )
        
        # Add back-run edge
        backrun_edge = BackRunEdge(
            chain_name="ethereum",
            target_transaction="0x123456789abcdef",
            source_asset="WETH",
            target_asset="USDC",
            expected_profit=25.0
        )
        
        # Add all edges to graph
        assert graph.add_edge(trade_edge) is True
        assert graph.add_edge(flash_edge) is True
        assert graph.add_edge(backrun_edge) is True
        
        # Test graph stats
        assert len(graph) == 3
        stats = graph.get_stats()
        assert stats["total_edges"] == 3
        assert EdgeType.TRADE.value in stats["edge_types"]
        assert EdgeType.FLASH_LOAN.value in stats["edge_types"]
        assert EdgeType.BACK_RUN.value in stats["edge_types"]
    
    def test_graph_edge_retrieval(self):
        """Test retrieving enhanced edges from graph."""
        graph = UniversalYieldGraph()
        
        flash_edge = FlashLoanEdge(
            chain_name="ethereum",
            provider="aave",
            asset="WETH",
            max_amount=1000.0
        )
        
        graph.add_edge(flash_edge)
        
        # Test edge retrieval
        retrieved_edge = graph.get_edge(flash_edge.edge_id)
        assert retrieved_edge is not None
        assert isinstance(retrieved_edge, FlashLoanEdge)
        assert retrieved_edge.provider == "aave"
        assert retrieved_edge.max_amount == 1000.0
    
    def test_graph_edge_filtering_by_execution_properties(self):
        """Test filtering edges by execution properties."""
        graph = UniversalYieldGraph()
        
        # Add synchronous edge
        sync_edge = YieldGraphEdge(
            edge_id="ETH_SYNC_TRADE",
            source_asset_id="ETH_WETH",
            target_asset_id="ETH_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="UniswapV3",
            chain_name="ethereum"
        )
        
        # Add asynchronous edge  
        async_props = EdgeExecutionProperties(
            supports_synchronous=False,
            requires_time_delay=3600
        )
        async_edge = YieldGraphEdge(
            edge_id="ETH_ASYNC_LEND",
            source_asset_id="ETH_USDC",
            target_asset_id="ETH_AUSDC",
            edge_type=EdgeType.LEND,
            protocol_name="AaveV3",
            chain_name="ethereum",
            execution_properties=async_props
        )
        
        graph.add_edge(sync_edge)
        graph.add_edge(async_edge)
        
        # Filter synchronous edges
        synchronous_edges = [
            edge for edge in graph.edges.values()
            if edge.execution_properties.supports_synchronous
        ]
        
        asynchronous_edges = [
            edge for edge in graph.edges.values()
            if not edge.execution_properties.supports_synchronous
        ]
        
        assert len(synchronous_edges) == 1
        assert len(asynchronous_edges) == 1
        assert synchronous_edges[0].edge_id == "ETH_SYNC_TRADE"
        assert asynchronous_edges[0].edge_id == "ETH_ASYNC_LEND"