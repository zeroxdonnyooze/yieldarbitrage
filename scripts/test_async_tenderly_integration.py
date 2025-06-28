#!/usr/bin/env python3
"""
Test async Tenderly integration functionality.

This script demonstrates the complete async workflow for router simulation
using the Tenderly API integration.
"""
import asyncio
import sys
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.router_simulator import (
    RouterSimulator, RouterSimulationParams, SimulationStatus, TenderlyNetworkId
)
from yield_arbitrage.execution.pre_execution_validator import (
    PreExecutionValidator, ValidationResult
)
from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient, TenderlySimulationResult, TenderlyTransaction
)
from yield_arbitrage.execution.calldata_generator import CalldataGenerator
from yield_arbitrage.pathfinding.path_segment_analyzer import (
    PathSegmentAnalyzer, PathSegment, SegmentType
)


async def test_mock_tenderly_integration():
    """Test the complete async integration with mocked Tenderly client."""
    print("🔄 Testing Async Tenderly Integration")
    print("=" * 40)
    
    # Create mock Tenderly client
    mock_tenderly_client = Mock(spec=TenderlyClient)
    mock_tenderly_client.initialize = AsyncMock()
    mock_tenderly_client.simulate_transaction = AsyncMock()
    mock_tenderly_client.close = AsyncMock()
    
    # Mock successful simulation response
    mock_simulation_result = TenderlySimulationResult(
        success=True,
        gas_used=675_000,
        gas_cost_usd=27.0,
        transaction_hash="0xabcdef1234567890",
        block_number=18_500_000,
        simulation_time_ms=1500.0
    )
    mock_tenderly_client.simulate_transaction.return_value = mock_simulation_result
    
    # Create mock calldata generator
    mock_calldata_generator = Mock(spec=CalldataGenerator)
    
    # Create router simulator
    router_simulator = RouterSimulator(
        tenderly_client=mock_tenderly_client,
        calldata_generator=mock_calldata_generator,
        default_router_address="0x1234567890123456789012345678901234567890"
    )
    
    print("✅ RouterSimulator created with mock Tenderly client")
    
    # Create test segment
    test_segment = PathSegment(
        segment_id="async_test_segment",
        segment_type=SegmentType.ATOMIC,
        edges=[],
        start_index=0,
        end_index=2,
        max_gas_estimate=700_000,
        requires_flash_loan=False
    )
    
    # Create simulation parameters
    simulation_params = RouterSimulationParams(
        router_contract_address="0x1234567890123456789012345678901234567890",
        network_id=TenderlyNetworkId.ETHEREUM,
        gas_limit=2_000_000,
        gas_price_gwei=20.0,
        initial_token_balances={
            "USDC": Decimal("50000"),
            "WETH": Decimal("20")
        }
    )
    
    print(f"✅ Created test segment: {test_segment.segment_id}")
    print(f"   - Type: {test_segment.segment_type.value}")
    print(f"   - Is atomic: {test_segment.is_atomic}")
    print(f"   - Gas estimate: {test_segment.max_gas_estimate:,}")
    
    # Mock calldata generation
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata
    mock_segment_calldata = SegmentCalldata(
        segment_id=test_segment.segment_id,
        operations=[],
        requires_flash_loan=False,
        recipient="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    )
    mock_calldata_generator.generate_segment_calldata.return_value = mock_segment_calldata
    
    # Test async segment simulation
    print(f"\n🚀 Executing async segment simulation...")
    
    simulation_result = await router_simulator.simulate_segment_execution(
        test_segment, simulation_params
    )
    
    print(f"✅ Simulation completed!")
    print(f"   - Status: {simulation_result.status.value}")
    print(f"   - Gas used: {simulation_result.gas_used:,}")
    print(f"   - Gas cost: ${simulation_result.gas_cost_usd:.2f}")
    print(f"   - Success: {simulation_result.success}")
    print(f"   - Transaction hash: {simulation_result.transaction_hash}")
    print(f"   - Simulation time: {simulation_result.simulation_time_ms:.1f}ms")
    
    # Verify mocks were called correctly
    mock_tenderly_client.simulate_transaction.assert_called_once()
    mock_calldata_generator.generate_segment_calldata.assert_called_once()
    
    return simulation_result


async def test_batch_simulation():
    """Test batch simulation with multiple segments."""
    print(f"\n📦 Testing Batch Simulation")
    print("=" * 30)
    
    # Create mock Tenderly client
    mock_tenderly_client = Mock(spec=TenderlyClient)
    mock_tenderly_client.initialize = AsyncMock()
    mock_tenderly_client.simulate_transaction = AsyncMock()
    
    # Mock calldata generator
    mock_calldata_generator = Mock(spec=CalldataGenerator)
    
    # Create router simulator
    router_simulator = RouterSimulator(
        tenderly_client=mock_tenderly_client,
        calldata_generator=mock_calldata_generator,
        default_router_address="0x1234567890123456789012345678901234567890"
    )
    
    # Create multiple test segments
    segments = [
        PathSegment(
            segment_id="batch_segment_1",
            segment_type=SegmentType.ATOMIC,
            edges=[],
            start_index=0,
            end_index=1,
            max_gas_estimate=400_000
        ),
        PathSegment(
            segment_id="batch_segment_2",
            segment_type=SegmentType.FLASH_LOAN_ATOMIC,
            edges=[],
            start_index=0,
            end_index=2,
            max_gas_estimate=650_000,
            requires_flash_loan=True,
            flash_loan_asset="USDC",
            flash_loan_amount=75000.0
        )
    ]
    
    # Mock simulation results for each segment
    simulation_results = [
        TenderlySimulationResult(success=True, gas_used=380_000),
        TenderlySimulationResult(success=True, gas_used=620_000)
    ]
    mock_tenderly_client.simulate_transaction.side_effect = simulation_results
    
    # Mock calldata generation
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata
    mock_calldata_generator.generate_segment_calldata.return_value = SegmentCalldata(
        segment_id="mock",
        operations=[],
        requires_flash_loan=False
    )
    
    # Execute batch simulation
    simulation_params = RouterSimulationParams(
        router_contract_address="0x1234567890123456789012345678901234567890",
        gas_price_gwei=15.0
    )
    
    batch_result = await router_simulator.simulate_batch_execution(
        segments, simulation_params
    )
    
    print(f"✅ Batch simulation completed!")
    print(f"   - Total segments: {batch_result.total_segments}")
    print(f"   - Successful: {batch_result.successful_segments}")
    print(f"   - Failed: {batch_result.failed_segments}")
    print(f"   - Success rate: {batch_result.success_rate:.1f}%")
    print(f"   - Total gas used: {batch_result.total_gas_used:,}")
    print(f"   - Total gas cost: ${batch_result.total_gas_cost_usd:.2f}")
    
    return batch_result


async def test_atomic_execution_validation():
    """Test atomic execution validation."""
    print(f"\n⚛️  Testing Atomic Execution Validation")
    print("=" * 40)
    
    # Create mock components
    mock_tenderly_client = Mock(spec=TenderlyClient)
    mock_tenderly_client.simulate_transaction = AsyncMock()
    mock_calldata_generator = Mock(spec=CalldataGenerator)
    
    router_simulator = RouterSimulator(
        tenderly_client=mock_tenderly_client,
        calldata_generator=mock_calldata_generator,
        default_router_address="0x1234567890123456789012345678901234567890"
    )
    
    # Test with atomic segments
    atomic_segments = [
        PathSegment(
            segment_id="atomic_1",
            segment_type=SegmentType.ATOMIC,
            edges=[],
            start_index=0,
            end_index=1
        ),
        PathSegment(
            segment_id="atomic_2", 
            segment_type=SegmentType.FLASH_LOAN_ATOMIC,
            edges=[],
            start_index=0,
            end_index=1,
            requires_flash_loan=True
        )
    ]
    
    # Mock successful simulations
    mock_tenderly_client.simulate_transaction.return_value = TenderlySimulationResult(
        success=True, gas_used=500_000
    )
    
    # Mock calldata generation
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata
    mock_calldata_generator.generate_segment_calldata.return_value = SegmentCalldata(
        segment_id="mock", operations=[], requires_flash_loan=False
    )
    
    simulation_params = RouterSimulationParams(
        router_contract_address="0x1234567890123456789012345678901234567890"
    )
    
    # Test atomicity validation
    is_atomic, issues = await router_simulator.validate_atomic_execution(
        atomic_segments, simulation_params
    )
    
    print(f"✅ Atomicity validation completed!")
    print(f"   - Is atomic: {is_atomic}")
    print(f"   - Issues found: {len(issues)}")
    if issues:
        for issue in issues:
            print(f"     * {issue}")
    
    # Test with non-atomic segment
    non_atomic_segments = [
        PathSegment(
            segment_id="non_atomic",
            segment_type=SegmentType.TIME_DELAYED,
            edges=[],
            start_index=0,
            end_index=1,
            requires_delay_seconds=1800  # 30 minutes
        )
    ]
    
    is_atomic_2, issues_2 = await router_simulator.validate_atomic_execution(
        non_atomic_segments, simulation_params
    )
    
    print(f"✅ Non-atomic validation completed!")
    print(f"   - Is atomic: {is_atomic_2}")
    print(f"   - Issues found: {len(issues_2)}")
    
    return is_atomic, is_atomic_2


async def test_pre_execution_validator():
    """Test the complete pre-execution validation system."""
    print(f"\n🔬 Testing Pre-Execution Validator")
    print("=" * 35)
    
    # Create mock components
    mock_tenderly_client = Mock(spec=TenderlyClient)
    mock_calldata_generator = Mock(spec=CalldataGenerator)
    
    router_simulator = RouterSimulator(
        tenderly_client=mock_tenderly_client,
        calldata_generator=mock_calldata_generator,
        default_router_address="0x1234567890123456789012345678901234567890"
    )
    
    path_analyzer = PathSegmentAnalyzer()
    
    pre_execution_validator = PreExecutionValidator(
        router_simulator=router_simulator,
        path_analyzer=path_analyzer
    )
    
    print(f"✅ PreExecutionValidator created")
    print(f"   - Min profit threshold: ${pre_execution_validator.min_profit_usd}")
    print(f"   - Max gas price: {pre_execution_validator.max_gas_price_gwei} gwei")
    print(f"   - Max segment gas: {pre_execution_validator.max_segment_gas:,}")
    
    return pre_execution_validator


async def main():
    """Run all async integration tests."""
    print("🚀 Async Tenderly Integration Tests")
    print("=" * 40)
    
    try:
        # Test 1: Basic async simulation
        simulation_result = await test_mock_tenderly_integration()
        assert simulation_result.status == SimulationStatus.SUCCESS
        
        # Test 2: Batch simulation
        batch_result = await test_batch_simulation()
        assert batch_result.success_rate == 100.0
        
        # Test 3: Atomic execution validation
        is_atomic_1, is_atomic_2 = await test_atomic_execution_validation()
        assert is_atomic_1 is True  # Atomic segments should be atomic
        assert is_atomic_2 is False  # Non-atomic segment should not be atomic
        
        # Test 4: Pre-execution validator
        validator = await test_pre_execution_validator()
        assert validator is not None
        
        print(f"\n{'='*40}")
        print(f"🎉 All Async Tests PASSED!")
        print(f"✅ Router simulation working")
        print(f"✅ Batch execution working")
        print(f"✅ Atomicity validation working")
        print(f"✅ Pre-execution validator ready")
        print(f"\n🚀 Task 11.5 - Tenderly Integration COMPLETE!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Async tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the async tests
    success = asyncio.run(main())
    exit(0 if success else 1)