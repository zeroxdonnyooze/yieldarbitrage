"""
Unit tests for Router Simulation Integration with Tenderly.

Tests the complete integration of router simulation, pre-execution validation,
and Tenderly API for validating atomic execution viability.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from yield_arbitrage.execution.router_simulator import (
    RouterSimulator, RouterSimulationParams, RouterSimulationResult, 
    SimulationStatus, TenderlyNetworkId
)
from yield_arbitrage.execution.pre_execution_validator import (
    PreExecutionValidator, ExecutionValidationReport, ValidationResult
)
from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient, TenderlySimulationResult, TenderlyTransaction
)
from yield_arbitrage.execution.calldata_generator import CalldataGenerator
from yield_arbitrage.pathfinding.path_segment_analyzer import (
    PathSegmentAnalyzer, PathSegment, SegmentType
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


class TestRouterSimulationIntegration:
    """Test complete router simulation integration."""
    
    @pytest.fixture
    def mock_tenderly_client(self):
        """Create mock Tenderly client."""
        client = Mock(spec=TenderlyClient)
        client.initialize = AsyncMock()
        client.simulate_transaction = AsyncMock()
        client.create_virtual_testnet = AsyncMock()
        client.close = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_calldata_generator(self):
        """Create mock calldata generator."""
        generator = Mock(spec=CalldataGenerator)
        generator.generate_segment_calldata = Mock()
        return generator
    
    @pytest.fixture
    def router_simulator(self, mock_tenderly_client, mock_calldata_generator):
        """Create router simulator with mocked dependencies."""
        return RouterSimulator(
            tenderly_client=mock_tenderly_client,
            calldata_generator=mock_calldata_generator,
            default_router_address="0x1234567890123456789012345678901234567890"
        )
    
    @pytest.fixture
    def path_analyzer(self):
        """Create path segment analyzer."""
        return PathSegmentAnalyzer()
    
    @pytest.fixture
    def pre_execution_validator(self, router_simulator, path_analyzer):
        """Create pre-execution validator."""
        return PreExecutionValidator(
            router_simulator=router_simulator,
            path_analyzer=path_analyzer
        )
    
    @pytest.fixture
    def sample_atomic_segment(self):
        """Create a sample atomic segment for testing."""
        return PathSegment(
            segment_id="test_segment_1",
            segment_type=SegmentType.ATOMIC,
            edges=[],
            start_index=0,
            end_index=1,
            requires_flash_loan=False,
            max_gas_estimate=500_000
        )
    
    @pytest.fixture
    def sample_flash_loan_segment(self):
        """Create a sample flash loan segment for testing."""
        return PathSegment(
            segment_id="test_segment_2",
            segment_type=SegmentType.FLASH_LOAN_ATOMIC,
            edges=[],
            start_index=0,
            end_index=2,
            requires_flash_loan=True,
            flash_loan_amount=100000.0,
            flash_loan_asset="USDC",
            max_gas_estimate=800_000
        )
    
    @pytest.fixture
    def simulation_params(self):
        """Create simulation parameters."""
        return RouterSimulationParams(
            router_contract_address="0x1234567890123456789012345678901234567890",
            network_id=TenderlyNetworkId.ETHEREUM,
            executor_address="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            gas_limit=2_000_000,
            gas_price_gwei=20.0,
            initial_token_balances={
                "USDC": Decimal("10000"),
                "WETH": Decimal("5")
            }
        )


def test_router_simulation_params_creation():
    """Test RouterSimulationParams creation and defaults."""
    params = RouterSimulationParams(
        router_contract_address="0x1234567890123456789012345678901234567890"
    )
    
    assert params.router_contract_address == "0x1234567890123456789012345678901234567890"
    assert params.network_id == TenderlyNetworkId.ETHEREUM
    assert params.gas_limit == 8_000_000
    assert params.gas_price_gwei == 20.0
    assert len(params.initial_token_balances) == 0


@pytest.mark.asyncio
async def test_simulate_successful_segment_execution(
    router_simulator, 
    sample_atomic_segment, 
    simulation_params,
    mock_tenderly_client,
    mock_calldata_generator
):
    """Test successful segment execution simulation."""
    
    # Mock successful Tenderly simulation
    mock_tenderly_result = TenderlySimulationResult(
        success=True,
        gas_used=450_000,
        transaction_hash="0xabcd1234",
        block_number=18_500_000
    )
    mock_tenderly_client.simulate_transaction.return_value = mock_tenderly_result
    
    # Mock calldata generation
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_segment_1",
        operations=[],
        requires_flash_loan=False
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Execute simulation
    result = await router_simulator.simulate_segment_execution(
        sample_atomic_segment, simulation_params
    )
    
    # Verify results
    assert result.status == SimulationStatus.SUCCESS
    assert result.segment_id == "test_segment_1"
    assert result.gas_used == 450_000
    assert result.success is True
    assert result.transaction_hash == "0xabcd1234"
    
    # Verify mocks were called
    mock_tenderly_client.simulate_transaction.assert_called_once()
    mock_calldata_generator.generate_segment_calldata.assert_called_once()


@pytest.mark.asyncio
async def test_simulate_failed_segment_execution(
    router_simulator,
    sample_atomic_segment,
    simulation_params,
    mock_tenderly_client,
    mock_calldata_generator
):
    """Test failed segment execution simulation."""
    
    # Mock failed Tenderly simulation
    mock_tenderly_result = TenderlySimulationResult(
        success=False,
        gas_used=800_000,
        error_message="Execution reverted",
        revert_reason="Insufficient output amount"
    )
    mock_tenderly_client.simulate_transaction.return_value = mock_tenderly_result
    
    # Mock calldata generation
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata
    mock_segment_calldata = SegmentCalldata(
        segment_id="test_segment_1",
        operations=[],
        requires_flash_loan=False
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Execute simulation
    result = await router_simulator.simulate_segment_execution(
        sample_atomic_segment, simulation_params
    )
    
    # Verify results
    assert result.status == SimulationStatus.REVERTED
    assert result.segment_id == "test_segment_1"
    assert result.gas_used == 800_000
    assert result.success is False
    assert result.error_message == "Execution reverted"
    assert result.revert_reason == "Insufficient output amount"


@pytest.mark.asyncio
async def test_batch_execution_simulation(
    router_simulator,
    sample_atomic_segment,
    sample_flash_loan_segment,
    simulation_params,
    mock_tenderly_client,
    mock_calldata_generator
):
    """Test batch execution simulation with multiple segments."""
    
    segments = [sample_atomic_segment, sample_flash_loan_segment]
    
    # Mock simulation results for both segments
    successful_result = TenderlySimulationResult(
        success=True,
        gas_used=450_000
    )
    failed_result = TenderlySimulationResult(
        success=False,
        gas_used=200_000,
        error_message="Flash loan failed"
    )
    
    mock_tenderly_client.simulate_transaction.side_effect = [
        successful_result,
        failed_result
    ]
    
    # Mock calldata generation
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata
    mock_calldata_generator.generate_segment_calldata.return_value = SegmentCalldata(
        segment_id="mock",
        operations=[],
        requires_flash_loan=False
    )
    
    # Execute batch simulation
    batch_result = await router_simulator.simulate_batch_execution(
        segments, simulation_params
    )
    
    # Verify batch results
    assert batch_result.total_segments == 2
    assert batch_result.successful_segments == 1
    assert batch_result.failed_segments == 1
    assert batch_result.total_gas_used == 650_000
    assert batch_result.success_rate == 50.0
    
    # Verify individual results
    assert len(batch_result.segment_results) == 2
    assert batch_result.segment_results[0].status == SimulationStatus.SUCCESS
    assert batch_result.segment_results[1].status == SimulationStatus.FAILED


@pytest.mark.asyncio
async def test_validate_atomic_execution(
    router_simulator,
    sample_atomic_segment,
    simulation_params,
    mock_tenderly_client,
    mock_calldata_generator
):
    """Test atomic execution validation."""
    
    segments = [sample_atomic_segment]
    
    # Mock successful simulation
    mock_tenderly_client.simulate_transaction.return_value = TenderlySimulationResult(
        success=True,
        gas_used=400_000
    )
    
    # Mock calldata generation
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata
    mock_calldata_generator.generate_segment_calldata.return_value = SegmentCalldata(
        segment_id="test",
        operations=[],
        requires_flash_loan=False
    )
    
    # Validate atomicity
    is_atomic, issues = await router_simulator.validate_atomic_execution(
        segments, simulation_params
    )
    
    # Should be atomic since all segments are atomic and simulation succeeds
    assert is_atomic is True
    assert len(issues) == 0


@pytest.mark.asyncio
async def test_validate_atomic_execution_with_non_atomic_segment(
    router_simulator,
    simulation_params
):
    """Test atomic execution validation with non-atomic segment."""
    
    non_atomic_segment = PathSegment(
        segment_id="non_atomic",
        segment_type=SegmentType.TIME_DELAYED,
        edges=[],
        start_index=0,
        end_index=1,
        requires_delay_seconds=3600  # 1 hour delay
    )
    
    segments = [non_atomic_segment]
    
    # Validate atomicity
    is_atomic, issues = await router_simulator.validate_atomic_execution(
        segments, simulation_params
    )
    
    # Should not be atomic due to time delay
    assert is_atomic is False
    assert len(issues) > 0
    assert "not atomic" in issues[0]


@pytest.mark.asyncio  
async def test_pre_execution_validation_complete_flow(
    pre_execution_validator,
    router_simulator,
    mock_tenderly_client,
    mock_calldata_generator,
    simulation_params
):
    """Test complete pre-execution validation flow."""
    
    # Create mock edges
    mock_edges = [
        Mock(spec=YieldGraphEdge, edge_type=EdgeType.TRADE),
        Mock(spec=YieldGraphEdge, edge_type=EdgeType.TRADE)
    ]
    
    # Mock path analyzer to return atomic segments
    with patch.object(pre_execution_validator.path_analyzer, 'analyze_path') as mock_analyze:
        atomic_segment = PathSegment(
            segment_id="validation_test",
            segment_type=SegmentType.ATOMIC,
            edges=mock_edges,
            start_index=0,
            end_index=2,
            max_gas_estimate=600_000
        )
        mock_analyze.return_value = [atomic_segment]
        
        # Mock successful simulation
        mock_tenderly_client.simulate_transaction.return_value = TenderlySimulationResult(
            success=True,
            gas_used=550_000
        )
        
        # Mock calldata generation
        from yield_arbitrage.execution.calldata_generator import SegmentCalldata
        mock_calldata_generator.generate_segment_calldata.return_value = SegmentCalldata(
            segment_id="validation_test",
            operations=[],
            requires_flash_loan=False
        )
        
        # Execute validation
        report = await pre_execution_validator.validate_execution_plan(
            mock_edges,
            Decimal("1000"),
            simulation_params
        )
        
        # Verify validation report
        assert isinstance(report, ExecutionValidationReport)
        assert report.validation_result in [ValidationResult.VALID, ValidationResult.WARNING]
        assert report.total_segments == 1
        assert report.estimated_gas_usage > 0
        assert report.validation_time_ms > 0


def test_router_simulation_statistics(router_simulator):
    """Test router simulation statistics tracking."""
    
    # Get initial stats
    initial_stats = router_simulator.get_stats()
    
    assert initial_stats["simulations_run"] == 0
    assert initial_stats["successful_simulations"] == 0
    assert initial_stats["failed_simulations"] == 0
    assert initial_stats["success_rate"] == 0.0


@pytest.mark.asyncio
async def test_gas_efficiency_validation(
    pre_execution_validator,
    sample_atomic_segment,
    simulation_params,
    mock_tenderly_client,
    mock_calldata_generator
):
    """Test gas efficiency validation across different gas prices."""
    
    # Mock simulation results for different gas prices
    def mock_simulate_batch(segments, params):
        from yield_arbitrage.execution.router_simulator import BatchSimulationResult
        return BatchSimulationResult(
            total_segments=1,
            successful_segments=1,
            failed_segments=0,
            total_gas_used=500_000,
            total_gas_cost_usd=params.gas_price_gwei * 0.5,  # Simplified cost
            estimated_profit=Decimal("100") - Decimal(str(params.gas_price_gwei))
        )
    
    # Patch the batch simulation method
    with patch.object(
        pre_execution_validator.router_simulator, 
        'simulate_batch_execution',
        side_effect=mock_simulate_batch
    ):
        # Execute gas efficiency validation
        efficiency_data = await pre_execution_validator.validate_gas_efficiency(
            [sample_atomic_segment], simulation_params
        )
        
        # Verify results
        assert "gas_efficiency_curve" in efficiency_data
        assert "optimal_gas_price" in efficiency_data
        assert "gas_elasticity" in efficiency_data
        assert len(efficiency_data["gas_efficiency_curve"]) == 5  # 5 gas prices tested


@pytest.mark.asyncio
async def test_flash_loan_requirements_validation(
    pre_execution_validator,
    sample_flash_loan_segment
):
    """Test flash loan requirements validation."""
    
    segments = [sample_flash_loan_segment]
    
    # Execute flash loan validation
    validation_results = await pre_execution_validator.validate_flash_loan_requirements(
        segments
    )
    
    # Verify results
    assert validation_results["total_segments"] == 1
    assert validation_results["flash_loan_segments"] == 1
    assert len(validation_results["flash_loan_requirements"]) == 1
    
    requirement = validation_results["flash_loan_requirements"][0]
    assert requirement["segment_id"] == "test_segment_2"
    assert requirement["asset"] == "USDC"
    assert requirement["amount"] == 100000.0


def test_validation_issue_creation():
    """Test ValidationIssue creation and properties."""
    from yield_arbitrage.execution.pre_execution_validator import ValidationIssue
    
    issue = ValidationIssue(
        severity="error",
        category="gas",
        message="Gas limit exceeded",
        segment_id="test_segment",
        suggested_fix="Reduce operations per segment"
    )
    
    assert issue.severity == "error"
    assert issue.category == "gas"
    assert issue.message == "Gas limit exceeded"
    assert issue.segment_id == "test_segment"
    assert issue.suggested_fix == "Reduce operations per segment"


if __name__ == "__main__":
    # Run a simple test
    import asyncio
    
    async def run_basic_test():
        print("🧪 Testing Router Simulation Integration")
        print("=" * 50)
        
        # Test basic parameter creation
        params = RouterSimulationParams(
            router_contract_address="0x1234567890123456789012345678901234567890"
        )
        print(f"✅ Created simulation params: {params.network_id.value}")
        
        # Test atomic segment creation
        segment = PathSegment(
            segment_id="test_segment",
            segment_type=SegmentType.ATOMIC,
            edges=[],
            start_index=0,
            end_index=1
        )
        print(f"✅ Created atomic segment: {segment.segment_id}")
        print(f"   Is atomic: {segment.is_atomic}")
        
        print("\n✅ Router simulation integration components working!")
    
    asyncio.run(run_basic_test())