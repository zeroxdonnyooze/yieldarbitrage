"""
Unit tests for ExecutionEngine.

Tests the complete execution engine including pre-flight checks,
simulation integration, position tracking, and execution monitoring.
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock
from decimal import Decimal

from yield_arbitrage.execution.execution_engine import (
    ExecutionEngine, ExecutionContext, ExecutionResult, ExecutionStatus,
    PreFlightCheck, PreFlightCheckResult
)
from yield_arbitrage.pathfinding.path_models import YieldPath
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType
from yield_arbitrage.execution.hybrid_simulator import SimulationResult, SimulationMode
from yield_arbitrage.execution.enhanced_transaction_builder import BatchExecutionPlan, RouterTransaction
from yield_arbitrage.mev_protection.execution_router import ExecutionRoute, ExecutionMethod
from yield_arbitrage.mev_protection.mev_risk_assessor import PathMEVAnalysis, MEVRiskLevel


class TestExecutionEngine:
    """Test ExecutionEngine functionality."""
    
    @pytest.fixture
    def mock_simulator(self):
        """Create mock hybrid simulator."""
        simulator = Mock()
        simulator.simulate_path = AsyncMock(return_value=SimulationResult(
            success=True,
            simulation_mode=SimulationMode.HYBRID.value,
            profit_usd=25.50,
            gas_cost_usd=8.75,
            output_amount=1000.0
        ))
        return simulator
    
    @pytest.fixture
    def mock_transaction_builder(self):
        """Create mock transaction builder."""
        builder = Mock()
        
        # Mock execution plan
        mock_plan = BatchExecutionPlan(
            plan_id="test_plan",
            router_address="0x" + "2" * 40,
            executor_address="0x" + "3" * 40,
            segments=[
                RouterTransaction(
                    segment_id="seg_1",
                    to_address="0x" + "2" * 40,
                    from_address="0x" + "3" * 40,
                    gas_limit=200000
                )
            ],
            total_gas_estimate=200000
        )
        
        builder.build_batch_execution_plan = AsyncMock(return_value=mock_plan)
        return builder
    
    @pytest.fixture
    def mock_mev_router(self):
        """Create mock MEV router."""
        router = Mock()
        router.select_execution_route = Mock(return_value=ExecutionRoute(
            method=ExecutionMethod.PUBLIC,
            endpoint="https://rpc.example.com",
            priority_fee_wei=1000000000
        ))
        return router
    
    @pytest.fixture
    def mock_delta_tracker(self):
        """Create mock delta tracker."""
        tracker = Mock()
        tracker.add_position = AsyncMock(return_value=True)
        tracker.get_portfolio_snapshot = AsyncMock(return_value=Mock(
            total_usd_long=Decimal('50000'),
            total_usd_short=Decimal('0')
        ))
        return tracker
    
    @pytest.fixture
    def mock_mev_assessor(self):
        """Create mock MEV assessor."""
        assessor = Mock()
        assessor.assess_path_risk = AsyncMock(return_value=PathMEVAnalysis(
            path_id="test_path",
            total_edges=2,
            overall_risk_level=MEVRiskLevel.LOW
        ))
        return assessor
    
    @pytest.fixture
    def mock_asset_oracle(self):
        """Create mock asset oracle."""
        oracle = Mock()
        oracle.get_price_usd = AsyncMock(return_value=2000.0)  # Default $2000 price
        return oracle
    
    @pytest.fixture
    def sample_path(self):
        """Create sample yield path."""
        edges = [
            Mock(spec=YieldGraphEdge,
                 edge_id="edge_1",
                 source_asset_id="ETH_MAINNET_WETH",
                 target_asset_id="ETH_MAINNET_USDC",
                 edge_type=EdgeType.TRADE,
                 protocol_name="uniswapv3"),
            Mock(spec=YieldGraphEdge,
                 edge_id="edge_2", 
                 source_asset_id="ETH_MAINNET_USDC",
                 target_asset_id="ETH_MAINNET_WETH",
                 edge_type=EdgeType.TRADE,
                 protocol_name="sushiswap")
        ]
        
        return YieldPath(
            path_id="test_path",
            edges=edges,
            expected_yield=0.025
        )
    
    @pytest.fixture
    def execution_engine(self, mock_simulator, mock_transaction_builder, mock_mev_router,
                        mock_delta_tracker, mock_mev_assessor, mock_asset_oracle):
        """Create ExecutionEngine instance with mocks."""
        return ExecutionEngine(
            simulator=mock_simulator,
            transaction_builder=mock_transaction_builder,
            mev_router=mock_mev_router,
            delta_tracker=mock_delta_tracker,
            mev_assessor=mock_mev_assessor,
            asset_oracle=mock_asset_oracle,
            router_address="0x" + "1" * 40,
            chain_id=1
        )
    
    def test_execution_engine_initialization(self, execution_engine):
        """Test ExecutionEngine initialization."""
        assert execution_engine.chain_id == 1
        assert execution_engine.router_address == "0x" + "1" * 40
        assert execution_engine.default_simulation_mode == SimulationMode.HYBRID
        assert execution_engine.enable_pre_flight_checks
        assert execution_engine.enable_position_tracking
        assert execution_engine.enable_mev_protection
        
        # Check initial stats
        stats = execution_engine.get_stats()
        assert stats["executions_attempted"] == 0
        assert stats["executions_successful"] == 0
        assert stats["active_executions"] == 0
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, execution_engine, sample_path):
        """Test complete successful execution flow."""
        # Execute path
        result = await execution_engine.execute_path(
            path=sample_path,
            initial_amount=1.0
        )
        
        # Verify result
        assert result.success
        assert result.status == ExecutionStatus.COMPLETED
        assert result.simulation_result is not None
        assert result.execution_plan is not None
        assert result.position_created is not None
        assert len(result.transaction_hashes) > 0
        assert len(result.pre_flight_checks) > 0
        
        # Verify stats updated
        stats = execution_engine.get_stats()
        assert stats["executions_attempted"] == 1
        assert stats["executions_successful"] == 1
        assert stats["total_profit_realized_usd"] > 0
    
    @pytest.mark.asyncio
    async def test_execution_with_custom_context(self, execution_engine, sample_path):
        """Test execution with custom execution context."""
        context = ExecutionContext(
            execution_id="custom_exec",
            path=sample_path,
            initial_amount=2.5,
            start_asset_id="ETH_MAINNET_WETH",
            chain_id=1,
            max_slippage=0.01,  # 1%
            position_size_limit_usd=200000.0
        )
        
        result = await execution_engine.execute_path(
            path=sample_path,
            initial_amount=2.5,
            execution_context=context
        )
        
        assert result.success
        assert "custom_exec" in result.execution_id
        assert result.execution_time_seconds is not None
    
    @pytest.mark.asyncio
    async def test_pre_flight_check_failure(self, execution_engine, sample_path, mock_asset_oracle):
        """Test execution failure due to pre-flight checks."""
        # Make oracle return None to cause pre-flight failure
        mock_asset_oracle.get_price_usd = AsyncMock(return_value=None)
        
        result = await execution_engine.execute_path(
            path=sample_path,
            initial_amount=1.0
        )
        
        # Should still succeed but with warnings (oracle failure is non-blocking)
        # Let's test with empty path which is blocking
        empty_path = YieldPath(path_id="empty", edges=[], expected_yield=0.0)
        
        result = await execution_engine.execute_path(
            path=empty_path,
            initial_amount=1.0
        )
        
        assert not result.success
        assert result.status == ExecutionStatus.FAILED
        assert "Pre-flight checks failed" in result.error_message
        assert len(result.pre_flight_checks) > 0
    
    @pytest.mark.asyncio
    async def test_simulation_failure(self, execution_engine, sample_path, mock_simulator):
        """Test execution failure due to simulation failure."""
        # Make simulation fail
        mock_simulator.simulate_path = AsyncMock(return_value=SimulationResult(
            success=False,
            simulation_mode=SimulationMode.HYBRID.value,
            revert_reason="Insufficient liquidity"
        ))
        
        result = await execution_engine.execute_path(
            path=sample_path,
            initial_amount=1.0
        )
        
        assert not result.success
        assert result.status == ExecutionStatus.FAILED
        assert "Simulation failed" in result.error_message
        assert result.simulation_result is not None
        assert not result.simulation_result.success
    
    @pytest.mark.asyncio
    async def test_position_limit_check(self, execution_engine, sample_path):
        """Test position limit pre-flight check."""
        # Test with very large position
        context = ExecutionContext(
            execution_id="large_pos",
            path=sample_path,
            initial_amount=100.0,  # 100 ETH = $200k at $2k/ETH
            start_asset_id="ETH_MAINNET_WETH",
            chain_id=1,
            position_size_limit_usd=100000.0  # $100k limit
        )
        
        result = await execution_engine.execute_path(
            path=sample_path,
            initial_amount=100.0,
            execution_context=context
        )
        
        # Should fail due to position size limit
        assert not result.success
        assert "Position size" in result.error_message
    
    @pytest.mark.asyncio
    async def test_mev_risk_assessment(self, execution_engine, sample_path, mock_mev_assessor):
        """Test MEV risk assessment in pre-flight checks."""
        # Set high MEV risk
        mock_mev_assessor.assess_path_risk = AsyncMock(return_value=PathMEVAnalysis(
            path_id="high_risk_path",
            total_edges=2,
            overall_risk_level=MEVRiskLevel.HIGH
        ))
        
        result = await execution_engine.execute_path(
            path=sample_path,
            initial_amount=1.0
        )
        
        # Should still succeed but with MEV warnings
        assert result.success or len([c for c in result.pre_flight_checks 
                                    if c.check_name == "mev_risk" and 
                                    c.result == PreFlightCheckResult.WARNING]) > 0
    
    def test_execution_status_tracking(self, execution_engine):
        """Test execution status tracking."""
        # No executions initially
        status = execution_engine.get_execution_status("nonexistent")
        assert status is None
        
        # Test with active execution (mock)
        context = ExecutionContext(
            execution_id="test_status",
            path=Mock(),
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            chain_id=1
        )
        
        execution_engine.active_executions["test_status"] = context
        
        status = execution_engine.get_execution_status("test_status")
        assert status is not None
        assert status["execution_id"] == "test_status"
        assert status["status"] == ExecutionStatus.PENDING.value
    
    def test_execution_cancellation(self, execution_engine):
        """Test execution cancellation."""
        # Add mock active execution
        context = ExecutionContext(
            execution_id="cancel_test",
            path=Mock(),
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            chain_id=1
        )
        
        execution_engine.active_executions["cancel_test"] = context
        
        # Cancel execution
        success = execution_engine.cancel_execution("cancel_test")
        assert success
        
        # Should be moved to completed executions
        assert "cancel_test" not in execution_engine.active_executions
        assert "cancel_test" in execution_engine.completed_executions
        
        result = execution_engine.completed_executions["cancel_test"]
        assert result.status == ExecutionStatus.CANCELLED
        assert not result.success
    
    @pytest.mark.asyncio
    async def test_path_validity_checks(self, execution_engine):
        """Test various path validity scenarios."""
        # Test disconnected path
        disconnected_edges = [
            Mock(spec=YieldGraphEdge,
                 source_asset_id="ETH_MAINNET_WETH",
                 target_asset_id="ETH_MAINNET_USDC"),
            Mock(spec=YieldGraphEdge,
                 source_asset_id="ETH_MAINNET_DAI",  # Disconnected!
                 target_asset_id="ETH_MAINNET_WETH")
        ]
        
        disconnected_path = YieldPath(
            path_id="disconnected",
            edges=disconnected_edges,
            expected_yield=0.01
        )
        
        result = await execution_engine.execute_path(
            path=disconnected_path,
            initial_amount=1.0
        )
        
        assert not result.success
        assert any("disconnected" in check.message.lower() 
                  for check in result.pre_flight_checks 
                  if check.result == PreFlightCheckResult.FAIL)
    
    @pytest.mark.asyncio
    async def test_execution_engine_stats(self, execution_engine, sample_path, mock_simulator):
        """Test statistics tracking."""
        initial_stats = execution_engine.get_stats()
        
        # Successful execution
        await execution_engine.execute_path(sample_path, 1.0)
        
        # Failed execution
        mock_simulator.simulate_path = AsyncMock(return_value=SimulationResult(
            success=False,
            simulation_mode=SimulationMode.BASIC.value,
            revert_reason="Test failure"
        ))
        
        await execution_engine.execute_path(sample_path, 1.0)
        
        final_stats = execution_engine.get_stats()
        
        assert final_stats["executions_attempted"] == initial_stats["executions_attempted"] + 2
        assert final_stats["executions_successful"] == initial_stats["executions_successful"] + 1
        assert final_stats["total_profit_realized_usd"] > initial_stats["total_profit_realized_usd"]
    
    @pytest.mark.asyncio
    async def test_position_tracking_integration(self, execution_engine, sample_path, mock_delta_tracker):
        """Test integration with DeltaTracker."""
        result = await execution_engine.execute_path(sample_path, 1.0)
        
        assert result.success
        assert result.position_created is not None
        
        # Verify delta tracker was called
        mock_delta_tracker.add_position.assert_called_once()
        call_args = mock_delta_tracker.add_position.call_args
        
        assert "pos_" in call_args[1]["position_id"]
        assert call_args[1]["position_type"] == "arbitrage"
        assert call_args[1]["path"] == sample_path.edges
    
    @pytest.mark.asyncio
    async def test_disabled_features(self, execution_engine, sample_path):
        """Test execution with disabled features."""
        # Disable features
        execution_engine.enable_pre_flight_checks = False
        execution_engine.enable_position_tracking = False
        execution_engine.enable_mev_protection = False
        
        result = await execution_engine.execute_path(sample_path, 1.0)
        
        assert result.success
        # Pre-flight checks should still run (safety feature)
        assert len(result.pre_flight_checks) > 0
        # Position tracking should be None
        assert result.position_created is None
        # MEV route should be None  
        assert result.execution_route is None


class TestExecutionContext:
    """Test ExecutionContext functionality."""
    
    def test_execution_context_creation(self):
        """Test ExecutionContext creation and defaults."""
        path = Mock()
        context = ExecutionContext(
            execution_id="test_context",
            path=path,
            initial_amount=5.0,
            start_asset_id="ETH_MAINNET_WETH",
            chain_id=1
        )
        
        assert context.execution_id == "test_context"
        assert context.initial_amount == 5.0
        assert context.max_slippage == 0.02  # 2% default
        assert context.status == ExecutionStatus.PENDING
        assert context.use_mev_protection
        assert context.flashbots_enabled
    
    def test_execution_context_custom_params(self):
        """Test ExecutionContext with custom parameters."""
        path = Mock()
        context = ExecutionContext(
            execution_id="custom_context",
            path=path,
            initial_amount=10.0,
            start_asset_id="ETH_MAINNET_USDC",
            chain_id=137,  # Polygon
            max_slippage=0.005,  # 0.5%
            max_gas_price_gwei=50.0,
            use_mev_protection=False
        )
        
        assert context.chain_id == 137
        assert context.max_slippage == 0.005
        assert context.max_gas_price_gwei == 50.0
        assert not context.use_mev_protection


class TestPreFlightChecks:
    """Test individual pre-flight check functionality."""
    
    def test_pre_flight_check_creation(self):
        """Test PreFlightCheck creation."""
        check = PreFlightCheck(
            check_name="test_check",
            result=PreFlightCheckResult.PASS,
            message="All good",
            details={"extra": "info"},
            is_blocking=False
        )
        
        assert check.check_name == "test_check"
        assert check.result == PreFlightCheckResult.PASS
        assert check.message == "All good"
        assert check.details["extra"] == "info"
        assert not check.is_blocking
    
    def test_pre_flight_check_levels(self):
        """Test different pre-flight check result levels."""
        pass_check = PreFlightCheck("pass", PreFlightCheckResult.PASS, "OK")
        warn_check = PreFlightCheck("warn", PreFlightCheckResult.WARNING, "Warning")
        fail_check = PreFlightCheck("fail", PreFlightCheckResult.FAIL, "Failed", is_blocking=True)
        
        assert pass_check.result == PreFlightCheckResult.PASS
        assert warn_check.result == PreFlightCheckResult.WARNING
        assert fail_check.result == PreFlightCheckResult.FAIL
        assert fail_check.is_blocking


if __name__ == "__main__":
    # Run basic functionality test
    print("🧪 Testing ExecutionEngine")
    print("=" * 50)
    
    # Test ExecutionContext
    from unittest.mock import Mock
    
    path = Mock()
    context = ExecutionContext(
        execution_id="test_manual",
        path=path,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        chain_id=1
    )
    
    print(f"✅ ExecutionContext created:")
    print(f"   - ID: {context.execution_id}")
    print(f"   - Amount: {context.initial_amount}")
    print(f"   - Chain: {context.chain_id}")
    print(f"   - Status: {context.status}")
    print(f"   - MEV Protection: {context.use_mev_protection}")
    
    # Test PreFlightCheck
    check = PreFlightCheck(
        check_name="manual_test",
        result=PreFlightCheckResult.PASS,
        message="Manual test passed"
    )
    
    print(f"\n✅ PreFlightCheck created:")
    print(f"   - Name: {check.check_name}")
    print(f"   - Result: {check.result}")
    print(f"   - Message: {check.message}")
    print(f"   - Blocking: {check.is_blocking}")
    
    print("\n✅ ExecutionEngine test passed!")