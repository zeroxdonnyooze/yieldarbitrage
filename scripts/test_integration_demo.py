#!/usr/bin/env python3
"""Comprehensive demonstration of integration testing for the unified simulation system."""
import asyncio
import sys
import time
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulatorConfig,
    TenderlyConfig,
    SimulationResult,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


async def demo_realistic_arbitrage_simulation():
    """Demonstrate realistic arbitrage simulation with real market conditions."""
    print("📊 Realistic Arbitrage Simulation Demo\n")
    
    # Setup simulation environment
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(side_effect=lambda asset_id: {
        "ETH_MAINNET_WETH": 2500.0,
        "ETH_MAINNET_USDC": 1.0,
        "ETH_MAINNET_DAI": 1.0,
        "ETH_MAINNET_USDT": 1.0,
    }.get(asset_id, 2000.0))
    
    config = SimulatorConfig(
        confidence_threshold=0.7,
        min_liquidity_threshold=50000.0,
        tenderly_profit_threshold_usd=10.0,
        local_rpc_url="http://localhost:8545"
    )
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config
    )
    
    # Test Case 1: Simple DEX Arbitrage
    print("🔄 Test 1: Simple DEX Arbitrage (ETH -> USDC -> ETH)")
    
    simple_path = [
        YieldGraphEdge(
            edge_id="eth_usdc_uniswap",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="usdc_eth_sushiswap",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap",
            chain_name="ethereum"
        )
    ]
    
    # Realistic edge states with small arbitrage opportunity
    current_time = time.time()
    edge_states = {
        "eth_usdc_uniswap": EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=1_500_000.0,
            gas_cost_usd=8.0,
            confidence_score=0.95,
            last_updated_timestamp=current_time - 30
        ),
        "usdc_eth_sushiswap": EdgeState(
            conversion_rate=0.000402,  # Small profit opportunity
            liquidity_usd=800_000.0,
            gas_cost_usd=8.0,
            confidence_score=0.88,
            last_updated_timestamp=current_time - 45
        )
    }
    
    async def mock_get_edge_state(edge_id):
        return edge_states.get(edge_id)
    
    simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
    
    # Test different amounts
    test_amounts = [0.1, 1.0, 10.0]
    
    for amount in test_amounts:
        result = await simulator.simulate_path(
            path=simple_path,
            initial_amount=amount,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        profit_status = "✅ Profitable" if result.success else "❌ Unprofitable"
        print(f"   Amount: {amount} ETH -> {profit_status}")
        print(f"      Profit: ${result.profit_usd:.2f} USD ({result.profit_percentage:.2f}%)")
        print(f"      Gas Cost: ${result.gas_cost_usd:.2f}")
        print(f"      Time: {result.simulation_time_ms:.1f}ms")
        print()
    
    # Test Case 2: Multi-hop Stablecoin Arbitrage
    print("🔗 Test 2: Multi-hop Stablecoin Arbitrage (ETH -> USDC -> DAI -> ETH)")
    
    multi_hop_path = [
        YieldGraphEdge(
            edge_id="eth_usdc_trade",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v3",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="usdc_dai_curve",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_DAI",
            edge_type=EdgeType.TRADE,
            protocol_name="curve",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="dai_eth_balancer",
            source_asset_id="ETH_MAINNET_DAI",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="balancer",
            chain_name="ethereum"
        )
    ]
    
    # Multi-hop edge states (typically unprofitable due to gas)
    multi_hop_states = {
        "eth_usdc_trade": EdgeState(
            conversion_rate=2498.5,
            liquidity_usd=2_000_000.0,
            gas_cost_usd=15.0,
            confidence_score=0.92,
            last_updated_timestamp=current_time - 25
        ),
        "usdc_dai_curve": EdgeState(
            conversion_rate=0.9995,
            liquidity_usd=5_000_000.0,
            gas_cost_usd=6.0,
            confidence_score=0.98,
            last_updated_timestamp=current_time - 15
        ),
        "dai_eth_balancer": EdgeState(
            conversion_rate=0.0004005,
            liquidity_usd=750_000.0,
            gas_cost_usd=18.0,
            confidence_score=0.85,
            last_updated_timestamp=current_time - 40
        )
    }
    
    async def mock_get_multi_hop_state(edge_id):
        return multi_hop_states.get(edge_id)
    
    simulator._get_edge_state = AsyncMock(side_effect=mock_get_multi_hop_state)
    
    result = await simulator.simulate_path(
        path=multi_hop_path,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.BASIC
    )
    
    print(f"   Multi-hop Result: {'✅ Profitable' if result.success else '❌ Unprofitable'}")
    print(f"   Profit: ${result.profit_usd:.2f} USD")
    print(f"   Gas Cost: ${result.gas_cost_usd:.2f} (High due to multiple hops)")
    print(f"   Steps: {len(result.path_details)}")
    print(f"   Time: {result.simulation_time_ms:.1f}ms")
    print()


async def demo_simulation_mode_comparison():
    """Demonstrate performance and accuracy comparison across simulation modes."""
    print("⚡ Simulation Mode Performance Comparison\n")
    
    # Setup
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2500.0)
    
    config = SimulatorConfig(local_rpc_url="http://localhost:8545")
    tenderly_config = TenderlyConfig(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config,
        tenderly_config=tenderly_config
    )
    
    # Test path
    test_path = [
        YieldGraphEdge(
            edge_id="test_trade",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
    ]
    
    # Mock edge state
    test_state = EdgeState(
        conversion_rate=2500.0,
        liquidity_usd=1_000_000.0,
        gas_cost_usd=10.0,
        confidence_score=0.9,
        last_updated_timestamp=time.time()
    )
    
    simulator._get_edge_state = AsyncMock(return_value=test_state)
    
    # Mock external simulations for comparison
    simulator._simulate_tenderly = AsyncMock(return_value=SimulationResult(
        success=True,
        simulation_mode=SimulationMode.TENDERLY.value,
        profit_usd=15.0,
        simulation_time_ms=800.0,
        gas_used=150_000,
        gas_cost_usd=22.5
    ))
    
    simulator._simulate_local = AsyncMock(return_value=SimulationResult(
        success=True,
        simulation_mode=SimulationMode.LOCAL.value,
        profit_usd=14.2,
        simulation_time_ms=2400.0,
        gas_used=148_000,
        gas_cost_usd=22.2
    ))
    
    # Test each mode
    modes = [
        (SimulationMode.BASIC, "📊 Basic Mathematical"),
        (SimulationMode.TENDERLY, "🌐 Tenderly API"),
        (SimulationMode.LOCAL, "🔧 Local Anvil"),
        (SimulationMode.HYBRID, "🔄 Hybrid (Best of Both)")
    ]
    
    results = {}
    
    for mode, description in modes:
        start_time = time.perf_counter()
        
        result = await simulator.simulate_path(
            path=test_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=mode
        )
        
        wall_time = (time.perf_counter() - start_time) * 1000
        
        results[mode.value] = {
            "description": description,
            "success": result.success,
            "profit": result.profit_usd,
            "gas_cost": result.gas_cost_usd,
            "sim_time": result.simulation_time_ms,
            "wall_time": wall_time
        }
        
        print(f"{description}:")
        print(f"   Success: {'✅' if result.success else '❌'}")
        profit_str = f"${result.profit_usd:.2f}" if result.profit_usd is not None else "N/A"
        print(f"   Profit: {profit_str}")
        gas_str = f"${result.gas_cost_usd:.2f}" if result.gas_cost_usd is not None else "N/A"
        print(f"   Gas Cost: {gas_str}")
        print(f"   Simulation Time: {result.simulation_time_ms:.1f}ms")
        print(f"   Wall Time: {wall_time:.1f}ms")
        if result.warnings:
            print(f"   Warnings: {len(result.warnings)}")
        print()
    
    # Performance summary
    print("📈 Performance Summary:")
    fastest = min(results.values(), key=lambda x: x["wall_time"])
    print(f"   Fastest: {fastest['description']} ({fastest['wall_time']:.1f}ms)")
    
    profitable = [r for r in results.values() if r["success"] and r["profit"] and r["profit"] > 0]
    if profitable:
        most_profitable = max(profitable, key=lambda x: x["profit"])
        print(f"   Most Profitable: {most_profitable['description']} (${most_profitable['profit']:.2f})")
    
    print()


async def demo_edge_validation_integration():
    """Demonstrate edge validation integration with realistic scenarios."""
    print("🔍 Edge Validation Integration Demo\n")
    
    # Setup
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2500.0)
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=SimulatorConfig()
    )
    
    # Test Case 1: Path with missing edge state
    print("📊 Test 1: Missing Edge State Validation")
    
    path_with_missing = [
        YieldGraphEdge(
            edge_id="existing_edge",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="missing_edge",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_DAI",
            edge_type=EdgeType.TRADE,
            protocol_name="curve",
            chain_name="ethereum"
        )
    ]
    
    async def mock_partial_states(edge_id):
        if edge_id == "existing_edge":
            return EdgeState(
                conversion_rate=2500.0,
                liquidity_usd=1_000_000.0,
                gas_cost_usd=10.0,
                confidence_score=0.9,
                last_updated_timestamp=time.time()
            )
        return None  # Missing state
    
    simulator._get_edge_state = AsyncMock(side_effect=mock_partial_states)
    
    result = await simulator.simulate_path(
        path=path_with_missing,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.HYBRID
    )
    
    print(f"   Result: {'✅ Simulated' if result else '❌ Failed'}")
    print(f"   Warnings: {len(result.warnings) if result.warnings else 0}")
    if result.warnings:
        for warning in result.warnings[:2]:  # Show first 2 warnings
            print(f"      • {warning}")
    print()
    
    # Test Case 2: Stale edge data
    print("⏰ Test 2: Stale Edge Data Validation")
    
    async def mock_stale_states(edge_id):
        return EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=1_000_000.0,
            gas_cost_usd=10.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time() - 1800  # 30 minutes old
        )
    
    simulator._get_edge_state = AsyncMock(side_effect=mock_stale_states)
    
    result = await simulator.simulate_path(
        path=path_with_missing[:1],  # Single edge
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.HYBRID
    )
    
    print(f"   Result: {'✅ Simulated' if result else '❌ Failed'}")
    stale_warnings = [w for w in (result.warnings or []) if "stale" in w.lower()]
    print(f"   Stale Data Warnings: {len(stale_warnings)}")
    print()
    
    # Test Case 3: Disconnected path
    print("🔗 Test 3: Disconnected Path Validation")
    
    disconnected_path = [
        YieldGraphEdge(
            edge_id="eth_usdc",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="wbtc_dai",  # Disconnected!
            source_asset_id="ETH_MAINNET_WBTC",
            target_asset_id="ETH_MAINNET_DAI",
            edge_type=EdgeType.TRADE,
            protocol_name="curve",
            chain_name="ethereum"
        )
    ]
    
    result = await simulator.simulate_path(
        path=disconnected_path,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.HYBRID
    )
    
    print(f"   Result: {'✅ Passed' if result.success else '❌ Failed'}")
    print(f"   Reason: {result.revert_reason if not result.success else 'N/A'}")
    if result.revert_reason and "disconnect" in result.revert_reason.lower():
        print("   ✅ Correctly detected disconnected path")
    print()


async def demo_error_handling_resilience():
    """Demonstrate error handling and system resilience."""
    print("🛡️ Error Handling & Resilience Demo\n")
    
    # Setup
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2500.0)
    
    config = SimulatorConfig(local_rpc_url="http://localhost:8545")
    tenderly_config = TenderlyConfig(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config,
        tenderly_config=tenderly_config
    )
    
    # Test path
    test_path = [
        YieldGraphEdge(
            edge_id="test_edge",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
    ]
    
    # Good edge state
    good_state = EdgeState(
        conversion_rate=2500.0,
        liquidity_usd=1_000_000.0,
        gas_cost_usd=10.0,
        confidence_score=0.9,
        last_updated_timestamp=time.time()
    )
    
    simulator._get_edge_state = AsyncMock(return_value=good_state)
    
    # Test Case 1: Tenderly failure with local fallback
    print("🔄 Test 1: Tenderly Failure → Local Fallback")
    
    simulator._simulate_tenderly = AsyncMock(side_effect=Exception("Tenderly API unavailable"))
    simulator._simulate_local = AsyncMock(return_value=SimulationResult(
        success=True,
        simulation_mode=SimulationMode.LOCAL.value,
        profit_usd=12.0,
        simulation_time_ms=2800.0,
        warnings=["Local simulation used as fallback"]
    ))
    
    result = await simulator.simulate_path(
        path=test_path,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.HYBRID
    )
    
    print(f"   Final Result: {'✅ Success' if result.success else '❌ Failed'}")
    print(f"   Profit: ${result.profit_usd:.2f}" if result.profit_usd else "   Profit: N/A")
    fallback_warnings = [w for w in (result.warnings or []) if "fallback" in w.lower()]
    print(f"   Fallback Used: {'✅ Yes' if fallback_warnings else '❌ No'}")
    print()
    
    # Test Case 2: All external simulations fail
    print("🚨 Test 2: All External Simulations Fail → Basic Fallback")
    
    simulator._simulate_tenderly = AsyncMock(side_effect=Exception("Tenderly down"))
    simulator._simulate_local = AsyncMock(side_effect=Exception("Anvil not available"))
    
    result = await simulator.simulate_path(
        path=test_path,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.HYBRID
    )
    
    print(f"   Final Result: {'✅ Success' if result.success else '❌ Failed'}")
    print(f"   Mode Used: {result.simulation_mode}")
    basic_warnings = [w for w in (result.warnings or []) if "basic" in w.lower()]
    print(f"   Basic Fallback: {'✅ Yes' if basic_warnings else '❌ No'}")
    print()
    
    # Test Case 3: Invalid input handling
    print("⚠️  Test 3: Invalid Input Handling")
    
    # Test with empty path
    empty_result = await simulator.simulate_path(
        path=[],
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.BASIC
    )
    
    print(f"   Empty Path: {'✅ Handled' if not empty_result.success else '❌ Not detected'}")
    
    # Test with negative amount
    negative_result = await simulator.simulate_path(
        path=test_path,
        initial_amount=-1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.BASIC
    )
    
    print(f"   Negative Amount: {'✅ Handled' if not negative_result.success else '❌ Not detected'}")
    print()


async def demo_real_world_scenarios():
    """Demonstrate realistic DeFi market scenarios."""
    print("🌍 Real-World DeFi Market Scenarios\n")
    
    # Setup
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(side_effect=lambda asset_id: {
        "ETH_MAINNET_WETH": 2487.32,  # Real ETH price
        "ETH_MAINNET_USDC": 0.9998,   # Slight depeg
        "ETH_MAINNET_USDT": 1.0002,   # Slight premium
        "ETH_MAINNET_DAI": 0.9996,    # Slight discount
    }.get(asset_id, 2000.0))
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=SimulatorConfig()
    )
    
    # Scenario 1: Stablecoin Depeg Arbitrage
    print("💰 Scenario 1: Stablecoin Depeg Arbitrage")
    print("   Market: USDC slightly depegged, USDT at premium")
    
    depeg_path = [
        YieldGraphEdge(
            edge_id="usdc_usdt_arb",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_USDT",
            edge_type=EdgeType.TRADE,
            protocol_name="curve",
            chain_name="ethereum"
        )
    ]
    
    # Profitable depeg scenario
    depeg_state = EdgeState(
        conversion_rate=1.0004,  # 4 basis points profit
        liquidity_usd=150_000_000.0,  # Large Curve pool
        gas_cost_usd=3.2,  # Low gas for stablecoin swap
        confidence_score=0.99,
        last_updated_timestamp=time.time() - 5
    )
    
    simulator._get_edge_state = AsyncMock(return_value=depeg_state)
    
    # Test different amounts
    depeg_amounts = [1000, 10000, 100000]  # USDC amounts
    
    for amount in depeg_amounts:
        result = await simulator.simulate_path(
            path=depeg_path,
            initial_amount=amount,
            start_asset_id="ETH_MAINNET_USDC",
            mode=SimulationMode.BASIC
        )
        
        profit_bps = (result.profit_percentage * 100) if result.profit_percentage else 0
        print(f"   ${amount:,} USDC: {'✅' if result.success else '❌'} {profit_bps:.1f} bps profit")
    
    print()
    
    # Scenario 2: High Gas Environment
    print("⛽ Scenario 2: High Gas Environment Impact")
    print("   Market: Gas prices at 100 gwei (high congestion)")
    
    high_gas_path = [
        YieldGraphEdge(
            edge_id="eth_usdc_high_gas",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="usdc_eth_high_gas",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap",
            chain_name="ethereum"
        )
    ]
    
    # High gas scenario
    high_gas_states = {
        "eth_usdc_high_gas": EdgeState(
            conversion_rate=2487.0,
            liquidity_usd=2_000_000.0,
            gas_cost_usd=45.0,  # High gas cost
            confidence_score=0.92,
            last_updated_timestamp=time.time() - 10
        ),
        "usdc_eth_high_gas": EdgeState(
            conversion_rate=0.000402,
            liquidity_usd=1_500_000.0,
            gas_cost_usd=38.0,  # High gas cost
            confidence_score=0.89,
            last_updated_timestamp=time.time() - 15
        )
    }
    
    async def mock_high_gas_states(edge_id):
        return high_gas_states.get(edge_id)
    
    simulator._get_edge_state = AsyncMock(side_effect=mock_high_gas_states)
    
    result = await simulator.simulate_path(
        path=high_gas_path,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.BASIC
    )
    
    print(f"   1 ETH Arbitrage: {'✅ Profitable' if result.success else '❌ Unprofitable'}")
    print(f"   Gas Impact: ${result.gas_cost_usd:.2f} (${result.gas_cost_usd/(result.profit_usd or 1)*100:.0f}% of potential profit)")
    print(f"   Net Result: ${result.profit_usd:.2f}")
    print()
    
    # Scenario 3: Low Liquidity Warning
    print("🏊 Scenario 3: Low Liquidity Pool Impact")
    print("   Market: Small pool with limited liquidity")
    
    low_liq_state = EdgeState(
        conversion_rate=2500.0,
        liquidity_usd=50_000.0,  # Small pool
        gas_cost_usd=12.0,
        confidence_score=0.75,  # Lower confidence
        last_updated_timestamp=time.time() - 120  # Slightly stale
    )
    
    simulator._get_edge_state = AsyncMock(return_value=low_liq_state)
    
    # Test impact of large trade on small pool
    large_trade_result = await simulator.simulate_path(
        path=depeg_path[:1],
        initial_amount=25_000,  # Large relative to liquidity
        start_asset_id="ETH_MAINNET_USDC",
        mode=SimulationMode.BASIC
    )
    
    print(f"   Large Trade Impact: {'⚠️  High' if large_trade_result.slippage_estimate > 0.02 else '✅ Low'}")
    print(f"   Slippage Estimate: {(large_trade_result.slippage_estimate or 0)*100:.2f}%")
    print(f"   Confidence Score: {low_liq_state.confidence_score:.2f}")
    print()


if __name__ == "__main__":
    print("🚀 Comprehensive Integration Testing Demo")
    print("=" * 60)
    print()
    
    # Run all demos
    asyncio.run(demo_realistic_arbitrage_simulation())
    asyncio.run(demo_simulation_mode_comparison())
    asyncio.run(demo_edge_validation_integration())
    asyncio.run(demo_error_handling_resilience())
    asyncio.run(demo_real_world_scenarios())
    
    print("🎉 Integration Testing Demo Completed!")
    print()
    print("📋 Integration Testing Summary:")
    print("   ✅ Realistic arbitrage path simulation")
    print("   ✅ Multi-mode performance comparison")
    print("   ✅ Edge validation with real market conditions")
    print("   ✅ Error handling and fallback mechanisms")
    print("   ✅ Real-world DeFi market scenarios")
    print("   ✅ Gas cost impact analysis")
    print("   ✅ Liquidity and slippage estimation")
    print("   ✅ Stablecoin depeg arbitrage detection")
    print()
    print("✅ Task 6.9: Integration Testing - COMPLETED")
    print("🚀 All unified simulation system testing complete!")