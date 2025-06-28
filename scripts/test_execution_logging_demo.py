#!/usr/bin/env python3
"""
Demonstration script for Execution Logging functionality.

This script demonstrates the integration between ExecutionEngine and PostgreSQL logging
without requiring a real database connection.
"""
import sys
import os
import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yield_arbitrage.execution.execution_engine import (
    ExecutionContext, ExecutionStatus, PreFlightCheck, PreFlightCheckResult
)
from yield_arbitrage.execution.hybrid_simulator import SimulationResult, SimulationMode
from yield_arbitrage.database.execution_logger import ExecutionLogger
from yield_arbitrage.pathfinding.path_models import YieldPath
from yield_arbitrage.graph_engine.models import EdgeType


def create_sample_execution_context():
    """Create a sample execution context for testing."""
    # Mock edges
    edges = [
        Mock(
            edge_id="ETH_MAINNET_UNISWAPV3_TRADE_WETH_USDC",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3"
        ),
        Mock(
            edge_id="ETH_MAINNET_SUSHISWAP_TRADE_USDC_WETH", 
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap"
        )
    ]
    
    # Create path
    path = YieldPath(
        path_id="demo_arbitrage_path_001",
        edges=edges,
        expected_yield=0.025  # 2.5% expected yield
    )
    
    # Create execution context
    context = ExecutionContext(
        execution_id="demo_execution_" + str(int(time.time())),
        path=path,
        initial_amount=1.0,  # 1 ETH
        start_asset_id="ETH_MAINNET_WETH",
        chain_id=1,
        max_slippage=0.01,  # 1%
        use_mev_protection=True
    )
    
    return context


def create_sample_pre_flight_checks():
    """Create sample pre-flight check results."""
    return [
        PreFlightCheck(
            check_name="path_validity",
            result=PreFlightCheckResult.PASS,
            message="Path connectivity verified: 2 edges properly connected",
            details={"edge_count": 2, "protocols": ["uniswapv3", "sushiswap"]}
        ),
        PreFlightCheck(
            check_name="asset_oracle_health",
            result=PreFlightCheckResult.PASS,
            message="All asset prices available",
            details={"assets_checked": ["ETH_MAINNET_WETH", "ETH_MAINNET_USDC"]}
        ),
        PreFlightCheck(
            check_name="position_limits",
            result=PreFlightCheckResult.PASS,
            message="Position size $2,000 within limits",
            details={"position_size_usd": 2000, "limit_usd": 100000}
        ),
        PreFlightCheck(
            check_name="market_conditions",
            result=PreFlightCheckResult.WARNING,
            message="Trading during low liquidity hours",
            details={"hour": 3, "warning_type": "low_liquidity_period"}
        ),
        PreFlightCheck(
            check_name="gas_conditions",
            result=PreFlightCheckResult.PASS,
            message="Gas cost estimate: $8.50",
            details={"estimated_gas": 400000, "gas_price_gwei": 25, "cost_usd": 8.50}
        ),
        PreFlightCheck(
            check_name="mev_risk",
            result=PreFlightCheckResult.PASS,
            message="MEV risk: LOW",
            details={"risk_level": "LOW", "flashbots_recommended": False}
        )
    ]


def create_sample_simulation_result():
    """Create sample simulation result."""
    return SimulationResult(
        success=True,
        simulation_mode=SimulationMode.HYBRID.value,
        profit_usd=25.50,
        profit_amount_start_asset=0.01275,  # 0.01275 ETH profit
        profit_percentage=1.275,  # 1.275%
        gas_used=387420,
        gas_cost_usd=9.75,
        output_amount=1.01275,  # 1.01275 ETH back (profit included)
        slippage_estimate=0.003,  # 0.3% total slippage
        warnings=[
            "High gas usage on Uniswap V3 step: 250,000 gas",
            "Sushiswap liquidity slightly lower than optimal"
        ],
        simulation_time_ms=1750.0,
        path_details=[
            {
                "step": 1,
                "edge_id": "ETH_MAINNET_UNISWAPV3_TRADE_WETH_USDC",
                "input_amount": 1.0,
                "output_amount": 2000.0,
                "gas_cost_usd": 6.25,
                "slippage_impact": 0.0015
            },
            {
                "step": 2,
                "edge_id": "ETH_MAINNET_SUSHISWAP_TRADE_USDC_WETH",
                "input_amount": 2000.0,
                "output_amount": 1.01275,
                "gas_cost_usd": 3.50,
                "slippage_impact": 0.0015
            }
        ]
    )


async def mock_database_session():
    """Mock database session for demonstration."""
    class MockSession:
        def __init__(self):
            self.added_records = []
            self.executed_statements = []
        
        def add(self, record):
            self.added_records.append(record)
            print(f"📝 Database ADD: {type(record).__name__}")
        
        async def commit(self):
            print("💾 Database COMMIT")
        
        async def execute(self, statement):
            self.executed_statements.append(statement)
            print(f"🔄 Database EXECUTE: UPDATE statement")
            return Mock(scalar_one_or_none=Mock(return_value=None))
        
        async def rollback(self):
            print("↩️  Database ROLLBACK")
        
        async def close(self):
            print("🔒 Database CLOSE")
    
    return MockSession()


async def demonstrate_execution_logging():
    """Demonstrate the complete execution logging flow."""
    print("🚀 Execution Logging Demonstration")
    print("=" * 60)
    
    # Create ExecutionLogger with mocked database
    logger = ExecutionLogger()
    
    # Patch the get_session function to use our mock
    original_get_session = None
    try:
        from yield_arbitrage.database.execution_logger import get_session
        original_get_session = get_session
    except ImportError:
        pass
    
    async def mock_get_session():
        mock_session = await mock_database_session()
        
        class MockContextManager:
            async def __aenter__(self):
                return mock_session
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await mock_session.close()
        
        return MockContextManager()
    
    # Replace get_session with our mock
    import yield_arbitrage.database.execution_logger as logger_module
    logger_module.get_session = mock_get_session
    
    try:
        # Step 1: Create execution context
        print("\n📋 Step 1: Creating Execution Context")
        context = create_sample_execution_context()
        print(f"   - Execution ID: {context.execution_id}")
        print(f"   - Path ID: {context.path.path_id}")
        print(f"   - Chain: {context.chain_id} (Ethereum)")
        print(f"   - Initial Amount: {context.initial_amount} ETH")
        print(f"   - Edges: {len(context.path.edges)}")
        
        # Step 2: Log execution start
        print("\n🎬 Step 2: Logging Execution Start")
        success = await logger.log_execution_start(
            context=context,
            session_id="demo_session_001",
            user_id="demo_user_123",
            api_key_id="api_key_456",
            request_source="demo_script"
        )
        print(f"   - Log Success: {success}")
        print(f"   - Records Created: {logger.stats['records_created']}")
        
        # Step 3: Log pre-flight checks
        print("\n🔍 Step 3: Logging Pre-Flight Checks")
        pre_flight_checks = create_sample_pre_flight_checks()
        
        for check in pre_flight_checks:
            status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[check.result.value]
            print(f"   {status_icon} {check.check_name}: {check.message}")
        
        success = await logger.log_pre_flight_results(
            execution_id=context.execution_id,
            pre_flight_checks=pre_flight_checks,
            pre_flight_time_ms=275
        )
        print(f"   - Log Success: {success}")
        print(f"   - Records Updated: {logger.stats['records_updated']}")
        
        # Step 4: Log simulation results
        print("\n🎮 Step 4: Logging Simulation Results")
        simulation_result = create_sample_simulation_result()
        
        print(f"   - Simulation Mode: {simulation_result.simulation_mode}")
        print(f"   - Success: {simulation_result.success}")
        print(f"   - Predicted Profit: ${simulation_result.profit_usd:.2f}")
        print(f"   - Gas Cost: ${simulation_result.gas_cost_usd:.2f}")
        print(f"   - Simulation Time: {simulation_result.simulation_time_ms:.0f}ms")
        
        market_context = {
            "eth_price_usd": 2000.0,
            "gas_price_gwei": 25.0,
            "block_number": 18500000
        }
        
        success = await logger.log_simulation_results(
            execution_id=context.execution_id,
            simulation_result=simulation_result,
            market_context=market_context
        )
        print(f"   - Log Success: {success}")
        
        # Step 5: Log execution completion (mock)
        print("\n🏁 Step 5: Logging Execution Completion")
        
        # Create mock execution result
        from yield_arbitrage.execution.execution_engine import ExecutionResult
        
        execution_result = ExecutionResult(
            execution_id=context.execution_id,
            success=True,
            status=ExecutionStatus.COMPLETED,
            simulation_result=simulation_result,
            transaction_hashes=["0xabc123...", "0xdef456..."],
            actual_profit_usd=24.75,  # Slightly less than predicted
            gas_used=392850,
            gas_cost_usd=9.95,  # Slightly more than estimated
            execution_time_seconds=12.5,
            position_created="pos_" + context.execution_id,
            warnings=["Actual slippage 0.35% vs 0.30% predicted"]
        )
        
        success = await logger.log_execution_completion(
            execution_result=execution_result,
            delta_exposure={
                "ETH_MAINNET_WETH": 0.01275,  # Net ETH gained
                "ETH_MAINNET_USDC": 0.0       # No residual USDC
            }
        )
        print(f"   - Execution Success: {execution_result.success}")
        print(f"   - Actual Profit: ${execution_result.actual_profit_usd:.2f}")
        print(f"   - Total Time: {execution_result.execution_time_seconds:.1f}s")
        print(f"   - Log Success: {success}")
        
        # Step 6: Display final statistics
        print("\n📊 Step 6: Final Logging Statistics")
        stats = logger.get_stats()
        
        print(f"   - Records Created: {stats['records_created']}")
        print(f"   - Records Updated: {stats['records_updated']}")
        print(f"   - Write Errors: {stats['write_errors']}")
        print(f"   - Last Write: {stats['last_write_time']}")
        
        # Step 7: Demonstrate analytics (mock)
        print("\n📈 Step 7: Execution Analytics (Mock)")
        print("   Note: In production, this would query the database")
        
        # Mock analytics data
        analytics = {
            "time_period_hours": 24,
            "total_executions": 47,
            "successful_executions": 42,
            "failed_executions": 5,
            "success_rate": 0.894,
            "profitable_executions": 38,
            "total_predicted_profit_usd": 1247.50,
            "avg_predicted_profit_usd": 32.83,
            "simulation_modes": {
                "hybrid": 35,
                "tenderly": 8,
                "basic": 4
            },
            "avg_simulation_time_ms": 1650.2,
            "pre_flight_failures": 3,
            "avg_pre_flight_warnings": 1.2,
            "most_common_protocols": {
                "uniswapv3": 28,
                "sushiswap": 19,
                "curve": 12,
                "aave": 8
            },
            "chain_distribution": {
                "ethereum": 47
            }
        }
        
        print(f"   - Success Rate: {analytics['success_rate']:.1%}")
        print(f"   - Avg Profit: ${analytics['avg_predicted_profit_usd']:.2f}")
        print(f"   - Top Protocol: {max(analytics['most_common_protocols'], key=analytics['most_common_protocols'].get)}")
        print(f"   - Avg Sim Time: {analytics['avg_simulation_time_ms']:.0f}ms")
        
        print("\n✅ Execution Logging Demonstration Complete!")
        print("\nKey Benefits Demonstrated:")
        print("• Comprehensive execution tracking from start to finish")
        print("• Detailed pre-flight check logging for debugging")
        print("• Simulation result storage for performance analysis")
        print("• Market context capture for historical analysis")
        print("• Position tracking integration with DeltaTracker")
        print("• Analytics capability for optimization insights")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original function if it existed
        if original_get_session:
            logger_module.get_session = original_get_session


if __name__ == "__main__":
    asyncio.run(demonstrate_execution_logging())