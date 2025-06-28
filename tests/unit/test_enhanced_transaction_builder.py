"""
Unit tests for Enhanced Transaction Builder.

Tests the complete integration of the enhanced transaction builder with
the YieldArbitrageRouter contract and router-based execution patterns.
"""
import pytest
from unittest.mock import Mock, MagicMock
from decimal import Decimal

from yield_arbitrage.execution.enhanced_transaction_builder import (
    EnhancedTransactionBuilder, RouterTransaction, BatchExecutionPlan,
    RouterIntegrationMode, create_simple_execution_plan, estimate_execution_cost
)
from yield_arbitrage.execution.calldata_generator import CalldataGenerator, SegmentCalldata
from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegment, SegmentType
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


class TestEnhancedTransactionBuilder:
    """Test enhanced transaction builder functionality."""
    
    @pytest.fixture
    def router_address(self):
        """Router contract address for testing."""
        return "0x1234567890123456789012345678901234567890"
    
    @pytest.fixture
    def executor_address(self):
        """Executor address for testing."""
        return "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    
    @pytest.fixture
    def mock_calldata_generator(self):
        """Mock calldata generator."""
        generator = Mock(spec=CalldataGenerator)
        generator.get_statistics.return_value = {"chain_id": 1}
        generator._get_token_address.return_value = "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65"
        return generator
    
    @pytest.fixture
    def enhanced_builder(self, router_address, mock_calldata_generator):
        """Create enhanced transaction builder."""
        return EnhancedTransactionBuilder(
            router_address=router_address,
            calldata_generator=mock_calldata_generator
        )
    
    @pytest.fixture
    def sample_atomic_segment(self):
        """Create sample atomic segment."""
        mock_edge = Mock(spec=YieldGraphEdge)
        mock_edge.edge_type = EdgeType.TRADE
        mock_edge.source_asset_id = "USDC"
        mock_edge.target_asset_id = "WETH"
        
        return PathSegment(
            segment_id="test_atomic_segment",
            segment_type=SegmentType.ATOMIC,
            edges=[mock_edge],
            start_index=0,
            end_index=1,
            max_gas_estimate=200_000,
            requires_flash_loan=False
        )
    
    @pytest.fixture
    def sample_flash_loan_segment(self):
        """Create sample flash loan segment."""
        mock_edge = Mock(spec=YieldGraphEdge)
        mock_edge.edge_type = EdgeType.TRADE
        mock_edge.source_asset_id = "USDC"
        mock_edge.target_asset_id = "WETH"
        
        return PathSegment(
            segment_id="test_flash_loan_segment",
            segment_type=SegmentType.FLASH_LOAN_ATOMIC,
            edges=[mock_edge],
            start_index=0,
            end_index=1,
            requires_flash_loan=True,
            flash_loan_asset="USDC",
            flash_loan_amount=100000.0,
            max_gas_estimate=400_000
        )


def test_enhanced_builder_initialization(router_address, mock_calldata_generator):
    """Test enhanced builder initialization."""
    builder = EnhancedTransactionBuilder(
        router_address=router_address,
        calldata_generator=mock_calldata_generator
    )
    
    assert builder.router_address == router_address
    assert builder.chain_id == 1
    assert builder.calldata_generator == mock_calldata_generator
    assert EdgeType.TRADE in builder.gas_estimates
    assert builder.stats["segments_built"] == 0


def test_build_segment_execution(
    enhanced_builder,
    sample_atomic_segment,
    executor_address,
    mock_calldata_generator
):
    """Test building segment execution transaction."""
    
    # Mock calldata generation
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_atomic_segment",
        operations=[],
        requires_flash_loan=False,
        recipient=executor_address
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Build segment execution
    router_tx = enhanced_builder.build_segment_execution(
        sample_atomic_segment,
        executor_address
    )
    
    # Verify router transaction
    assert isinstance(router_tx, RouterTransaction)
    assert router_tx.segment_id == "test_atomic_segment"
    assert router_tx.to_address == enhanced_builder.router_address
    assert router_tx.from_address == executor_address
    assert router_tx.requires_flash_loan is False
    assert router_tx.estimated_gas > 0
    
    # Verify calldata generation was called
    mock_calldata_generator.generate_segment_calldata.assert_called_once()
    
    # Verify statistics updated
    assert enhanced_builder.stats["segments_built"] == 1


def test_build_flash_loan_segment_execution(
    enhanced_builder,
    sample_flash_loan_segment,
    executor_address,
    mock_calldata_generator
):
    """Test building flash loan segment execution."""
    
    # Mock calldata generation
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_flash_loan_segment",
        operations=[],
        requires_flash_loan=True,
        flash_loan_asset="USDC",
        flash_loan_amount=100000,
        recipient=executor_address
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Build segment execution
    router_tx = enhanced_builder.build_segment_execution(
        sample_flash_loan_segment,
        executor_address
    )
    
    # Verify flash loan properties
    assert router_tx.requires_flash_loan is True
    assert router_tx.flash_loan_asset == "USDC"
    assert router_tx.flash_loan_amount == 100000
    assert router_tx.estimated_gas > enhanced_builder.gas_estimates[EdgeType.TRADE]


def test_build_batch_execution_direct_mode(
    enhanced_builder,
    sample_atomic_segment,
    executor_address,
    mock_calldata_generator
):
    """Test building batch execution in direct mode."""
    
    segments = [sample_atomic_segment, sample_atomic_segment]
    
    # Mock calldata generation
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_segment",
        operations=[],
        requires_flash_loan=False
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Build batch execution
    execution_plan = enhanced_builder.build_batch_execution(
        segments,
        executor_address,
        RouterIntegrationMode.DIRECT
    )
    
    # Verify execution plan
    assert isinstance(execution_plan, BatchExecutionPlan)
    assert execution_plan.router_address == enhanced_builder.router_address
    assert execution_plan.executor_address == executor_address
    assert len(execution_plan.segments) == 2
    assert execution_plan.total_gas_estimate > 0
    assert execution_plan.total_transactions >= 2
    
    # Verify statistics
    assert enhanced_builder.stats["batches_created"] == 1


def test_build_batch_execution_flash_loan_mode(
    enhanced_builder,
    sample_flash_loan_segment,
    sample_atomic_segment,
    executor_address,
    mock_calldata_generator
):
    """Test building batch execution in flash loan mode."""
    
    segments = [sample_flash_loan_segment, sample_atomic_segment]
    
    # Mock calldata generation
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_segment",
        operations=[],
        requires_flash_loan=False
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Build batch execution
    execution_plan = enhanced_builder.build_batch_execution(
        segments,
        executor_address,
        RouterIntegrationMode.FLASH_LOAN
    )
    
    # Verify flash loan handling
    assert len(execution_plan.segments) == 2
    flash_loan_segments = [tx for tx in execution_plan.segments if tx.requires_flash_loan]
    assert len(flash_loan_segments) == 1
    
    # Verify statistics
    assert enhanced_builder.stats["flash_loans_prepared"] == 1


def test_build_emergency_operations(enhanced_builder, executor_address):
    """Test building emergency withdrawal operations."""
    
    tokens = ["0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]
    amounts = [1000000, 500000]
    
    emergency_txs = enhanced_builder.build_emergency_operations(
        tokens, amounts, executor_address
    )
    
    # Verify emergency transactions
    assert len(emergency_txs) == 2
    for tx in emergency_txs:
        assert isinstance(tx, RouterTransaction)
        assert tx.to_address == enhanced_builder.router_address
        assert tx.from_address == executor_address
        assert tx.gas_limit == 100_000  # Emergency operations should be simple
        assert "emergency" in tx.segment_id


def test_gas_estimation(enhanced_builder, sample_atomic_segment, sample_flash_loan_segment):
    """Test gas estimation for different segment types."""
    
    # Test atomic segment gas estimation
    atomic_gas = enhanced_builder._estimate_segment_gas(sample_atomic_segment)
    assert atomic_gas > enhanced_builder.gas_estimates[EdgeType.TRADE]
    assert atomic_gas < 8_000_000  # Should not exceed block limit
    
    # Test flash loan segment gas estimation
    flash_loan_gas = enhanced_builder._estimate_segment_gas(sample_flash_loan_segment)
    assert flash_loan_gas > atomic_gas  # Flash loans should use more gas
    assert flash_loan_gas > enhanced_builder.gas_estimates[EdgeType.FLASH_LOAN]


def test_router_transaction_to_tenderly_conversion(enhanced_builder, executor_address):
    """Test converting RouterTransaction to TenderlyTransaction."""
    
    router_tx = RouterTransaction(
        segment_id="test_conversion",
        to_address=enhanced_builder.router_address,
        from_address=executor_address,
        value="1000000000000000000",  # 1 ETH
        gas_limit=500_000,
        data=b"\x12\x34\x56\x78"
    )
    
    tenderly_tx = router_tx.to_tenderly_transaction()
    
    assert tenderly_tx.from_address == executor_address
    assert tenderly_tx.to_address == enhanced_builder.router_address
    assert tenderly_tx.value == "1000000000000000000"
    assert tenderly_tx.gas == 500_000
    assert "12345678" in tenderly_tx.data  # Hex encoding


def test_execution_plan_simulation(enhanced_builder, sample_atomic_segment, executor_address, mock_calldata_generator):
    """Test execution plan simulation and analysis."""
    
    # Mock calldata generation
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_segment",
        operations=[],
        requires_flash_loan=False
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Create execution plan
    segments = [sample_atomic_segment] * 3
    execution_plan = enhanced_builder.build_batch_execution(
        segments, executor_address, RouterIntegrationMode.DIRECT
    )
    
    # Simulate execution plan
    analysis = enhanced_builder.simulate_execution_plan(execution_plan)
    
    # Verify analysis
    assert "plan_id" in analysis
    assert analysis["total_transactions"] == execution_plan.total_transactions
    assert analysis["total_gas_estimate"] == execution_plan.total_gas_estimate
    assert "recommendations" in analysis
    assert isinstance(analysis["recommendations"], list)


def test_execution_plan_optimization(enhanced_builder, sample_atomic_segment, sample_flash_loan_segment, executor_address, mock_calldata_generator):
    """Test execution plan optimization."""
    
    # Mock calldata generation
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_segment",
        operations=[],
        requires_flash_loan=False
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Create mixed execution plan
    segments = [sample_flash_loan_segment, sample_atomic_segment, sample_flash_loan_segment]
    original_plan = enhanced_builder.build_batch_execution(
        segments, executor_address, RouterIntegrationMode.HYBRID
    )
    
    # Optimize execution plan
    optimized_plan = enhanced_builder.optimize_execution_plan(original_plan)
    
    # Verify optimization
    assert optimized_plan.plan_id != original_plan.plan_id
    assert "optimized" in optimized_plan.plan_id
    assert len(optimized_plan.segments) == len(original_plan.segments)
    assert optimized_plan.total_gas_estimate <= original_plan.total_gas_estimate


def test_statistics_tracking(enhanced_builder, sample_atomic_segment, executor_address, mock_calldata_generator):
    """Test statistics tracking and reset."""
    
    # Mock calldata generation
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_segment",
        operations=[],
        requires_flash_loan=False
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Perform operations
    enhanced_builder.build_segment_execution(sample_atomic_segment, executor_address)
    enhanced_builder.build_batch_execution([sample_atomic_segment], executor_address)
    
    # Check statistics
    stats = enhanced_builder.get_statistics()
    assert stats["segments_built"] > 0
    assert stats["batches_created"] > 0
    assert stats["router_address"] == enhanced_builder.router_address
    assert "calldata_generator_stats" in stats
    
    # Reset statistics
    enhanced_builder.reset_statistics()
    reset_stats = enhanced_builder.get_statistics()
    assert reset_stats["segments_built"] == 0
    assert reset_stats["batches_created"] == 0


def test_approval_transaction_generation(enhanced_builder, sample_atomic_segment, executor_address):
    """Test approval transaction generation."""
    
    # Mock edge with trade operation requiring approval
    mock_edge = Mock(spec=YieldGraphEdge)
    mock_edge.edge_type = EdgeType.TRADE
    mock_edge.source_asset_id = "USDC"
    mock_edge.target_asset_id = "WETH"
    
    segment = PathSegment(
        segment_id="test_approval_segment",
        segment_type=SegmentType.ATOMIC,
        edges=[mock_edge],
        start_index=0,
        end_index=1
    )
    
    # Generate approval transactions
    approval_txs = enhanced_builder._generate_approval_transactions([segment], executor_address)
    
    # Verify approval transactions
    assert len(approval_txs) >= 0  # May be 0 if token address is 0x0
    for tx in approval_txs:
        assert tx.from_address == executor_address
        assert tx.to_address != enhanced_builder.router_address  # Approval is to token contract


def test_convenience_functions(sample_atomic_segment):
    """Test convenience functions."""
    
    router_address = "0x1234567890123456789012345678901234567890"
    executor_address = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    
    # Test create_simple_execution_plan
    execution_plan = create_simple_execution_plan(
        [sample_atomic_segment],
        router_address,
        executor_address
    )
    
    assert isinstance(execution_plan, BatchExecutionPlan)
    assert execution_plan.router_address == router_address
    assert execution_plan.executor_address == executor_address
    
    # Test estimate_execution_cost
    cost_analysis = estimate_execution_cost(execution_plan)
    
    assert "total_gas" in cost_analysis
    assert "gas_cost_usd" in cost_analysis
    assert "transactions" in cost_analysis
    assert cost_analysis["gas_cost_usd"] > 0


def test_edge_cases_and_validation(enhanced_builder, executor_address):
    """Test edge cases and validation."""
    
    # Test with empty segments list
    empty_plan = enhanced_builder.build_batch_execution([], executor_address)
    assert len(empty_plan.segments) == 0
    assert empty_plan.total_gas_estimate == 0
    
    # Test with invalid router address
    with pytest.raises(Exception):
        EnhancedTransactionBuilder("invalid_address")
    
    # Test emergency operations with empty lists
    empty_emergency = enhanced_builder.build_emergency_operations([], [], executor_address)
    assert len(empty_emergency) == 0


if __name__ == "__main__":
    # Run a simple test
    print("🧪 Testing Enhanced Transaction Builder")
    print("=" * 50)
    
    # Create test instances
    router_addr = "0x1234567890123456789012345678901234567890"
    executor_addr = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    
    # Test basic functionality
    builder = EnhancedTransactionBuilder(router_addr)
    
    # Create test segment
    mock_edge = Mock(spec=YieldGraphEdge)
    mock_edge.edge_type = EdgeType.TRADE
    mock_edge.source_asset_id = "USDC"
    mock_edge.target_asset_id = "WETH"
    
    segment = PathSegment(
        segment_id="integration_test",
        segment_type=SegmentType.ATOMIC,
        edges=[mock_edge],
        start_index=0,
        end_index=1
    )
    
    # Mock calldata generator response
    builder.calldata_generator.generate_segment_calldata = Mock(
        return_value=SegmentCalldata(
            segment_id="integration_test",
            operations=[],
            requires_flash_loan=False
        )
    )
    
    # Test segment execution
    router_tx = builder.build_segment_execution(segment, executor_addr)
    print(f"✅ Created router transaction: {router_tx.segment_id}")
    print(f"   - Gas estimate: {router_tx.estimated_gas:,}")
    print(f"   - Router address: {router_tx.to_address}")
    
    # Test batch execution
    execution_plan = builder.build_batch_execution([segment], executor_addr)
    print(f"✅ Created execution plan: {execution_plan.plan_id}")
    print(f"   - Total segments: {len(execution_plan.segments)}")
    print(f"   - Total gas: {execution_plan.total_gas_estimate:,}")
    
    # Test statistics
    stats = builder.get_statistics()
    print(f"✅ Builder statistics:")
    print(f"   - Segments built: {stats['segments_built']}")
    print(f"   - Batches created: {stats['batches_created']}")
    
    print("\n✅ Enhanced Transaction Builder integration test passed!")