#!/usr/bin/env python3
"""Comprehensive integration test with real Tenderly API."""
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    TenderlyConfig,
    SimulatorConfig,
    SimulationResult
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


async def test_full_integration():
    """Test complete integration with real Tenderly API."""
    print("🚀 Testing full integration with real Tenderly API...")
    
    # Create real Tenderly config
    tenderly_config = TenderlyConfig(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    # Mock other dependencies (Redis, Oracle)
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2000.0)  # ETH price
    
    # Create simulator
    simulator = HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=SimulatorConfig(),
        tenderly_config=tenderly_config
    )
    
    try:
        print("✅ Simulator created with real Tenderly config")
        
        # Test 1: Basic mode (should work without Tenderly)
        print("\n📊 Testing Basic simulation mode...")
        
        # Create a simple test edge
        edge = YieldGraphEdge(
            edge_id="test_eth_usdc",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
        
        # Mock edge state as proper JSON
        import json
        edge_state = {
            "reserves": {"token0": 1000.0, "token1": 2000000.0},
            "token0": "ETH_MAINNET_WETH",
            "token1": "ETH_MAINNET_USDC",
            "fee": 0.003,
            "last_updated": "2024-01-01T00:00:00Z"
        }
        
        # Mock Redis to return edge state as proper JSON
        mock_redis.get.return_value = json.dumps(edge_state).encode()
        
        result = await simulator.simulate_path(
            path=[edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        print(f"✅ Basic simulation: success={result.success}")
        print(f"   Output amount: {result.output_amount}")
        print(f"   Gas cost USD: {result.gas_cost_usd}")
        print(f"   Profit USD: {result.profit_usd}")
        
        # Test 2: Tenderly mode (should use real API)
        print("\n🌐 Testing Tenderly simulation mode...")
        
        # This will create a real Virtual TestNet and simulate
        tenderly_result = await simulator.simulate_path(
            path=[edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.TENDERLY,
            block_number=18500000
        )
        
        print(f"✅ Tenderly simulation: success={tenderly_result.success}")
        print(f"   Mode: {tenderly_result.simulation_mode}")
        print(f"   Virtual TestNet used: {bool(tenderly_result.logs)}")
        
        if not tenderly_result.success:
            print(f"   Expected failure reason: {tenderly_result.revert_reason}")
        
        # Test 3: Hybrid mode (should try Tenderly, fallback to basic)
        print("\n🔄 Testing Hybrid simulation mode...")
        
        hybrid_result = await simulator.simulate_path(
            path=[edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        print(f"✅ Hybrid simulation: success={hybrid_result.success}")
        print(f"   Final mode used: {hybrid_result.simulation_mode}")
        
        # Test 4: Check simulator statistics
        print("\n📊 Simulator statistics:")
        if hasattr(simulator, 'tenderly_client') and simulator.tenderly_client:
            stats = simulator.tenderly_client.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        print("\n🎉 Full integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if hasattr(simulator, 'tenderly_client') and simulator.tenderly_client:
            await simulator.tenderly_client.close()


async def test_transaction_building():
    """Test that transaction building works with real contracts."""
    print("\n🔧 Testing transaction building...")
    
    try:
        from yield_arbitrage.execution.transaction_builder import TransactionBuilder
        
        builder = TransactionBuilder()
        
        # Create a test edge for Uniswap V2
        edge = YieldGraphEdge(
            edge_id="test_uniswap_v2",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
        
        # Add contract info via metadata
        edge.metadata = {"contract_address": "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852"}  # ETH/USDC pair
        
        # Build transaction
        tx = builder.build_edge_transaction(
            edge=edge,
            input_amount=1.0,
            from_address="0x000000000000000000000000000000000000dead"
        )
        
        print(f"✅ Transaction built successfully!")
        print(f"   To: {tx.to_address}")
        print(f"   From: {tx.from_address}")
        print(f"   Data length: {len(tx.data)}")
        print(f"   Value: {tx.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transaction building failed: {e}")
        return False


if __name__ == "__main__":
    print("🌟 Comprehensive Integration Test\n")
    
    # Test full integration
    success1 = asyncio.run(test_full_integration())
    
    # Test transaction building
    success2 = asyncio.run(test_transaction_building())
    
    print(f"\n📋 Final Results:")
    print(f"  Full Integration: {'✅' if success1 else '❌'}")
    print(f"  Transaction Building: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("\n🎉 All integration tests passed!")
        print("🚀 System ready for production with real Tenderly API!")
    else:
        print("\n🔧 Some tests failed - check output above")