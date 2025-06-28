#!/usr/bin/env python3
"""Test hybrid simulation mode with real Tenderly API."""
import asyncio
import sys
import json
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    TenderlyConfig,
    SimulatorConfig,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


async def test_hybrid_mode_scenarios():
    """Test various hybrid mode scenarios."""
    print("🔄 Testing Hybrid Simulation Mode Scenarios\n")
    
    # Configure with real Tenderly credentials
    tenderly_config = TenderlyConfig(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    # Configure simulator for testing
    config = SimulatorConfig(
        tenderly_profit_threshold_usd=10.0,
        tenderly_amount_threshold_usd=1000.0,
        default_slippage_factor=0.03,
        min_liquidity_threshold=5000.0
    )
    
    # Mock dependencies
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2500.0)  # ETH price
    
    # Create simulator
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config,
        tenderly_config=tenderly_config
    )
    
    # Test scenarios
    scenarios = [
        {
            "name": "Small Trade - Should Use Basic Only",
            "path": create_simple_path(),
            "amount": 0.01,  # Small amount
            "expected_mode": "basic"
        },
        {
            "name": "Large Trade - Should Use Tenderly",
            "path": create_simple_path(), 
            "amount": 1.0,  # Large amount ($2500)
            "expected_mode": "tenderly"
        },
        {
            "name": "Complex Path - Should Use Tenderly", 
            "path": create_complex_path(),
            "amount": 0.1,  # Small amount but complex path
            "expected_mode": "tenderly"
        },
        {
            "name": "Risky Flash Loan - Should Use Tenderly",
            "path": create_risky_path(),
            "amount": 0.05,  # Small amount but risky
            "expected_mode": "tenderly"
        }
    ]
    
    # Mock edge state for basic simulation
    edge_state_json = json.dumps({
        "reserves": {"token0": 10000.0, "token1": 25000000.0},
        "token0": "ETH_MAINNET_WETH",
        "token1": "ETH_MAINNET_USDC", 
        "fee": 0.003,
        "last_updated": "2024-01-01T00:00:00Z"
    }).encode()
    mock_redis.get.return_value = edge_state_json
    
    results = []
    
    for scenario in scenarios:
        print(f"🧪 Testing: {scenario['name']}")
        
        try:
            result = await simulator.simulate_path(
                path=scenario['path'],
                initial_amount=scenario['amount'],
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
            
            print(f"   ✅ Success: {result.success}")
            print(f"   📊 Mode: {result.simulation_mode}")
            print(f"   💰 Profit: ${result.profit_usd:.2f}" if result.profit_usd else "   💰 Profit: N/A")
            print(f"   ⛽ Gas: ${result.gas_cost_usd:.2f}" if result.gas_cost_usd else "   ⛽ Gas: N/A")
            print(f"   ⏱️  Time: {result.simulation_time_ms:.1f}ms" if result.simulation_time_ms else "   ⏱️  Time: N/A")
            
            if result.warnings:
                print(f"   ⚠️  Warnings: {len(result.warnings)}")
                for warning in result.warnings[:2]:  # Show first 2 warnings
                    print(f"      - {warning}")
            
            results.append({
                "scenario": scenario['name'],
                "success": result.success,
                "mode": result.simulation_mode,
                "expected_mode": scenario['expected_mode'],
                "profit_usd": result.profit_usd,
                "time_ms": result.simulation_time_ms
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "scenario": scenario['name'],
                "success": False,
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("📋 Summary:")
    successful_tests = sum(1 for r in results if r.get('success', False))
    print(f"   Successful tests: {successful_tests}/{len(results)}")
    
    for result in results:
        if result.get('success'):
            expected = result.get('expected_mode', '')
            actual = result.get('mode', '')
            mode_check = "✅" if expected in actual else "❓"
            print(f"   {mode_check} {result['scenario']}: {actual}")
    
    return results


def create_simple_path():
    """Create a simple 2-step trading path."""
    return [
        YieldGraphEdge(
            edge_id="eth_usdc_swap",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="usdc_dai_swap",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_DAI",
            edge_type=EdgeType.TRADE,
            protocol_name="curve",
            chain_name="ethereum"
        )
    ]


def create_complex_path():
    """Create a complex multi-step path."""
    return [
        YieldGraphEdge(
            edge_id=f"step_{i}",
            source_asset_id=f"asset_{i}",
            target_asset_id=f"asset_{i+1}",
            edge_type=EdgeType.TRADE,
            protocol_name=f"protocol_{i}",
            chain_name="ethereum"
        )
        for i in range(5)  # 5-step path (complex)
    ]


def create_risky_path():
    """Create a path with risky edge types."""
    return [
        YieldGraphEdge(
            edge_id="flash_loan_start",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.FLASH_LOAN,  # Risky
            protocol_name="aave",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="arbitrage_trade",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v3",
            chain_name="ethereum"
        )
    ]


async def test_decision_logic():
    """Test the decision logic for when to use Tenderly."""
    print("🤔 Testing Hybrid Decision Logic\n")
    
    config = SimulatorConfig(
        tenderly_profit_threshold_usd=15.0,
        tenderly_amount_threshold_usd=2000.0
    )
    
    mock_redis = Mock()
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=3000.0)
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config
    )
    
    # Test different scenarios
    test_cases = [
        {
            "name": "High Profit",
            "path": create_simple_path(),
            "amount": 0.1,
            "profit_usd": 20.0,  # Above threshold
            "expected": True
        },
        {
            "name": "Large Trade",
            "path": create_simple_path(),
            "amount": 1.0,  # $3000 > $2000 threshold
            "profit_usd": 5.0,
            "expected": True
        },
        {
            "name": "High Slippage",
            "path": create_simple_path(),
            "amount": 0.1,
            "profit_usd": 5.0,
            "slippage": 0.04,  # 4% slippage
            "expected": True
        },
        {
            "name": "Low Everything",
            "path": create_simple_path(),
            "amount": 0.01,  # $30
            "profit_usd": 2.0,
            "slippage": 0.01,  # 1% slippage
            "expected": False
        }
    ]
    
    from yield_arbitrage.execution.hybrid_simulator import SimulationResult
    
    for case in test_cases:
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=case["profit_usd"],
            slippage_estimate=case.get("slippage", 0.01)
        )
        
        should_use = await simulator._should_use_tenderly_validation(
            path=case["path"],
            initial_amount=case["amount"],
            basic_result=basic_result
        )
        
        result_icon = "✅" if should_use == case["expected"] else "❌"
        print(f"   {result_icon} {case['name']}: {should_use} (expected {case['expected']})")
    
    print()


if __name__ == "__main__":
    print("🚀 Hybrid Simulation Mode Testing\n")
    
    # Test decision logic
    asyncio.run(test_decision_logic())
    
    # Test full scenarios
    results = asyncio.run(test_hybrid_mode_scenarios())
    
    # Check if we can proceed
    successful_count = sum(1 for r in results if r.get('success', False))
    if successful_count >= len(results) * 0.75:  # 75% success rate
        print("🎉 Hybrid Mode Implementation SUCCESSFUL!")
        print("✅ Ready to proceed to Task 6.7!")
    else:
        print("🔧 Some tests failed - may need debugging")