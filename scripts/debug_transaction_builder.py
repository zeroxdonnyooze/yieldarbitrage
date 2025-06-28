#!/usr/bin/env python3
"""Debug script for transaction builder issues."""
import asyncio
import sys
import logging
from decimal import Decimal
from unittest.mock import AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.execution.real_transaction_builder import RealTransactionBuilder
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def debug_transaction_builder():
    """Debug the transaction builder."""
    print("🔍 Debugging Transaction Builder...")
    
    # Initialize blockchain provider
    blockchain_provider = BlockchainProvider()
    await blockchain_provider.initialize()
    
    try:
        # Mock Redis client
        redis_client = AsyncMock()
        redis_client.ping.return_value = True
        redis_client.get.return_value = None
        
        # Initialize oracle
        oracle = OnChainPriceOracle(blockchain_provider, redis_client)
        await oracle.initialize()
        
        # Initialize transaction builder
        builder = RealTransactionBuilder(blockchain_provider, oracle)
        await builder.initialize()
        
        # Create test arbitrage path
        print("\n🔍 Creating test arbitrage path...")
        
        edge1 = YieldGraphEdge(
            edge_id="test_uniswap_v3_usdc_weth",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v3",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=0.00035,
                liquidity_usd=1000000,
                gas_cost_usd=15.0,
                confidence_score=0.95
            )
        )
        
        edge2 = YieldGraphEdge(
            edge_id="test_uniswap_v3_weth_usdc",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v3",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=2860.0,
                liquidity_usd=500000,
                gas_cost_usd=20.0,
                confidence_score=0.90
            )
        )
        
        test_path = [edge1, edge2]
        test_recipient = "0x742d35Cc6634C0532925a3b8D39C39c0fa6d5C4d"
        
        print(f"   Edge 1: {edge1.source_asset_id} -> {edge1.target_asset_id} ({edge1.protocol_name})")
        print(f"   Edge 2: {edge2.source_asset_id} -> {edge2.target_asset_id} ({edge2.protocol_name})")
        
        # Test transaction building
        print("\n🔍 Testing transaction building...")
        
        transaction = await builder.build_simple_arbitrage(
            test_path,
            Decimal("1000"),  # $1000 test amount
            recipient_address=test_recipient
        )
        
        if transaction:
            print(f"   ✅ Transaction built: {transaction.transaction_id}")
            print(f"   📊 Expected profit: ${transaction.expected_profit:.2f}")
            print(f"   ⛽ Gas estimate: {transaction.max_gas_limit:,}")
            print(f"   👥 Steps: {len(transaction.steps)}")
            
            for i, step in enumerate(transaction.steps):
                print(f"      Step {i+1}: {step.edge.protocol_name} - {step.function_name}")
                print(f"                Input: {step.input_amount}, Expected: {step.expected_output}")
            
            # Test simulation
            print("\n🔍 Testing transaction simulation...")
            simulation_passed = await builder.simulate_transaction(transaction)
            print(f"   🔍 Simulation: {'✅ PASSED' if simulation_passed else '❌ FAILED'}")
            
            if transaction.simulation_result:
                print(f"   📊 Simulation result: {transaction.simulation_result}")
        else:
            print("   ❌ Transaction building failed")
        
    finally:
        await blockchain_provider.close()


if __name__ == "__main__":
    asyncio.run(debug_transaction_builder())