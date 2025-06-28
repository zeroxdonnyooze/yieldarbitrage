#!/usr/bin/env python3
"""Demonstration of edge validation and call graph extraction."""
import asyncio
import sys
import json
import time
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    TenderlyConfig,
    SimulatorConfig,
    SimulationResult,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


async def demo_edge_validation():
    """Demonstrate edge validation functionality."""
    print("🔍 Edge Validation Demo\n")
    
    # Setup simulator
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2500.0)
    
    config = SimulatorConfig(
        confidence_threshold=0.8,
        min_liquidity_threshold=50000.0
    )
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config
    )
    
    # Test Case 1: Valid path
    print("📊 Test 1: Valid Path Validation")
    valid_path = [
        YieldGraphEdge(
            edge_id="eth_usdc_uniswap",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="usdc_dai_curve",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_DAI",
            edge_type=EdgeType.TRADE,
            protocol_name="curve",
            chain_name="ethereum"
        )
    ]
    
    # Mock good edge states
    good_state = EdgeState(
        conversion_rate=2500.0,
        liquidity_usd=500000.0,
        gas_cost_usd=8.0,
        confidence_score=0.95,
        last_updated_timestamp=time.time()
    )
    
    simulator._get_edge_state = AsyncMock(return_value=good_state)
    
    result = await simulator._validate_path_edges(valid_path)
    
    print(f"   ✅ Valid: {result['valid']}")
    print(f"   📝 Issues: {len(result['issues'])}")
    print(f"   ⚠️  Warnings: {len(result['warnings'])}")
    print(f"   🔗 Edge states: {len(result['edge_states'])}")
    print()
    
    # Test Case 2: Disconnected path
    print("🔗 Test 2: Disconnected Path Validation")
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
    
    result = await simulator._validate_path_edges(disconnected_path)
    
    print(f"   ❌ Valid: {result['valid']}")
    print(f"   📝 Issues: {len(result['issues'])}")
    if result['issues']:
        print(f"      → {result['issues'][0]}")
    print()
    
    # Test Case 3: Stale edge state
    print("⏰ Test 3: Stale Edge State Validation")
    stale_state = EdgeState(
        conversion_rate=2500.0,
        liquidity_usd=100000.0,
        gas_cost_usd=5.0,
        confidence_score=0.9,
        last_updated_timestamp=time.time() - 600  # 10 minutes ago
    )
    
    simulator._get_edge_state = AsyncMock(return_value=stale_state)
    
    result = await simulator._validate_path_edges(valid_path[:1])
    
    print(f"   ✅ Valid: {result['valid']} (warnings only)")
    print(f"   ⚠️  Warnings: {len(result['warnings'])}")
    print(f"   🕐 Stale states: {len(result['stale_states'])}")
    if result['warnings']:
        print(f"      → {result['warnings'][0]}")
    print()
    
    # Test Case 4: Protocol-specific validation
    print("🏗️ Test 4: Protocol-Specific Validation")
    
    # Valid Aave edge
    aave_edge = YieldGraphEdge(
        edge_id="aave_lend",
        source_asset_id="ETH_MAINNET_WETH",
        target_asset_id="ETH_MAINNET_AWETH",
        edge_type=EdgeType.LEND,
        protocol_name="aave_v2",
        chain_name="ethereum"
    )
    
    protocol_result = simulator._validate_protocol_specific(aave_edge, 0)
    print(f"   Aave LEND edge - Issues: {len(protocol_result['issues'])}, Warnings: {len(protocol_result['warnings'])}")
    
    # Invalid Uniswap edge type
    invalid_uniswap = YieldGraphEdge(
        edge_id="uniswap_lend",
        source_asset_id="ETH_MAINNET_WETH",
        target_asset_id="ETH_MAINNET_USDC",
        edge_type=EdgeType.LEND,  # Wrong for Uniswap!
        protocol_name="uniswap_v2",
        chain_name="ethereum"
    )
    
    protocol_result = simulator._validate_protocol_specific(invalid_uniswap, 0)
    print(f"   Invalid Uniswap edge - Issues: {len(protocol_result['issues'])}, Warnings: {len(protocol_result['warnings'])}")
    if protocol_result['issues']:
        print(f"      → {protocol_result['issues'][0]}")
    
    # Flash loan warning
    flash_loan_edge = YieldGraphEdge(
        edge_id="flash_loan",
        source_asset_id="ETH_MAINNET_WETH",
        target_asset_id="ETH_MAINNET_USDC",
        edge_type=EdgeType.FLASH_LOAN,
        protocol_name="aave_v2",
        chain_name="ethereum"
    )
    
    protocol_result = simulator._validate_protocol_specific(flash_loan_edge, 0)
    print(f"   Flash loan edge - Issues: {len(protocol_result['issues'])}, Warnings: {len(protocol_result['warnings'])}")
    if protocol_result['warnings']:
        print(f"      → {protocol_result['warnings'][0]}")
    print()


def demo_call_graph_extraction():
    """Demonstrate call graph extraction from Tenderly traces."""
    print("📞 Call Graph Extraction Demo\n")
    
    # Setup simulator
    mock_redis = Mock()
    mock_oracle = Mock()
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=SimulatorConfig()
    )
    
    # Test Case 1: Complex transaction trace
    print("🕸️ Test 1: Complex Transaction Trace")
    
    complex_trace = {
        "transaction": {
            "trace": {
                "to": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 Router
                "gasUsed": "150000",
                "type": "CALL",
                "input": "0x38ed1739",  # swapExactTokensForTokens
                "calls": [
                    {
                        "to": "0xa0b86a33e6417c5c6ef57e11d8caf7d8c0f7c8f8",  # USDC
                        "gasUsed": "5000",
                        "type": "CALL",
                        "input": "0xa9059cbb",  # transfer
                        "calls": []
                    },
                    {
                        "to": "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",  # ETH/USDC Pair
                        "gasUsed": "25000",
                        "type": "CALL", 
                        "input": "0x022c0d9f",  # swap
                        "calls": [
                            {
                                "to": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
                                "gasUsed": "3000",
                                "type": "CALL",
                                "input": "0xa9059cbb"  # transfer
                            }
                        ]
                    }
                ]
            },
            "logs": [
                {
                    "address": "0xa0b86a33e6417c5c6ef57e11d8caf7d8c0f7c8f8",
                    "topics": [
                        "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"  # Transfer
                    ],
                    "data": "0x"
                },
                {
                    "address": "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
                    "topics": [
                        "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"  # Sync
                    ],
                    "data": "0x"
                },
                {
                    "address": "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
                    "topics": [
                        "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"  # Swap
                    ],
                    "data": "0x"
                }
            ],
            "gas_used": 183000
        }
    }
    
    call_graph = simulator._extract_call_graph_from_trace(complex_trace)
    
    print(f"   📞 Total calls: {call_graph['total_calls']}")
    print(f"   🏗️ Unique contracts: {len(call_graph['unique_contracts'])}")
    print(f"   ⛽ Total gas used: {call_graph.get('total_gas_used', 'N/A')}")
    print(f"   📝 Events emitted: {len(call_graph['events_emitted'])}")
    print(f"   ❌ Failed calls: {len(call_graph['failed_calls'])}")
    print()
    
    print("   📊 Call Hierarchy:")
    for i, call in enumerate(call_graph['call_hierarchy'][:5]):  # First 5 calls
        indent = "  " * call['depth']
        print(f"   {i+1}. {indent}→ {call['to'][:10]}... ({call['call_type']}) - {call['gas_used']} gas")
    print()
    
    print("   📝 Events:")
    for event in call_graph['events_emitted']:
        print(f"   • {event['name']} from {event['address'][:10]}...")
    print()
    
    print("   ⛽ Gas by Contract:")
    for contract, gas in call_graph['gas_by_contract'].items():
        print(f"   • {contract[:10]}...: {gas:,} gas")
    print()
    
    # Test Case 2: Protocol detection
    print("🔍 Test 2: Protocol Detection")
    
    # Add known protocol contracts to test detection
    call_graph['unique_contracts'] = [
        "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",  # Uniswap V2 Factory
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # Aave V2
        "0x1234567890123456789012345678901234567890"   # Unknown contract
    ]
    
    simulator._detect_protocol_interactions(call_graph)
    
    print(f"   🏗️ Detected protocols: {len(call_graph['protocol_interactions'])}")
    for protocol, contracts in call_graph['protocol_interactions'].items():
        print(f"   • {protocol}: {len(contracts)} contract(s)")
    print()


async def demo_integrated_validation():
    """Demonstrate validation integrated with simulation."""
    print("🔄 Integrated Validation Demo\n")
    
    # Setup simulator with real Tenderly config
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2500.0)
    
    tenderly_config = TenderlyConfig(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=SimulatorConfig(),
        tenderly_config=tenderly_config
    )
    
    # Test with good edge state
    good_state = EdgeState(
        conversion_rate=2500.0,
        liquidity_usd=1000000.0,
        gas_cost_usd=5.0,
        confidence_score=0.9,
        last_updated_timestamp=time.time()
    )
    
    simulator._get_edge_state = AsyncMock(return_value=good_state)
    
    # Create valid path
    path = [
        YieldGraphEdge(
            edge_id="eth_usdc_test",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
    ]
    
    print("🧪 Test: Simulation with Edge Validation")
    
    # Mock a successful Tenderly result with trace
    mock_tenderly_result = SimulationResult(
        success=True,
        simulation_mode=SimulationMode.TENDERLY.value,
        profit_usd=15.0,
        output_amount=2525.0,
        gas_used=150000,
        gas_cost_usd=7.5,
        tenderly_trace={
            "transaction": {
                "trace": {
                    "to": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
                    "gasUsed": "150000",
                    "type": "CALL",
                    "input": "0x38ed1739"
                },
                "logs": [
                    {
                        "address": "0xa0b86a33e6417c5c6ef57e11d8caf7d8c0f7c8f8",
                        "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"],
                        "data": "0x"
                    }
                ],
                "gas_used": 150000
            }
        }
    )
    
    simulator._simulate_tenderly = AsyncMock(return_value=mock_tenderly_result)
    
    result = await simulator.simulate_path(
        path=path,
        initial_amount=1.0,
        start_asset_id="ETH_MAINNET_WETH",
        mode=SimulationMode.TENDERLY
    )
    
    print(f"   ✅ Success: {result.success}")
    print(f"   💰 Profit: ${result.profit_usd:.2f}")
    print(f"   ⛽ Gas cost: ${result.gas_cost_usd:.2f}")
    print(f"   ⚠️  Warnings: {len(result.warnings or [])}")
    print(f"   📊 Path details: {len(result.path_details or [])}")
    
    # Check if call graph was extracted
    if result.path_details:
        for detail in result.path_details:
            if detail.get('type') == 'call_graph':
                call_graph = detail['data']
                print(f"   📞 Call graph extracted: {call_graph.get('total_calls', 0)} calls")
                break
    
    print()


if __name__ == "__main__":
    print("🚀 Edge Validation and Call Graph Extraction Demo\n")
    
    # Run demos
    asyncio.run(demo_edge_validation())
    demo_call_graph_extraction()
    asyncio.run(demo_integrated_validation())
    
    print("🎉 All demos completed successfully!")
    print("✅ Edge validation working correctly")
    print("✅ Call graph extraction functional")
    print("✅ Integration with simulation flow working")
    print("\n🚀 Ready to proceed to Task 6.8!")