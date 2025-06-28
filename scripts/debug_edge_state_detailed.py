#!/usr/bin/env python3
"""Detailed debug script for edge state collection issues."""
import asyncio
import sys
import logging
import traceback
from unittest.mock import AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.data.real_edge_pipeline import RealEdgeStatePipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def debug_edge_state_detailed():
    """Debug the edge state collection with detailed error tracing."""
    print("🔍 Debugging Edge State Collection (Detailed)...")
    
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
        
        # Initialize edge pipeline
        pipeline = RealEdgeStatePipeline(blockchain_provider, oracle, redis_client)
        await pipeline.initialize()
        
        print(f"\n🔍 Pipeline initialized with {len(pipeline.active_edges)} edges")
        
        # Test edge discovery
        edges = await pipeline.discover_edges()
        print(f"   Discovered {len(edges)} edges")
        
        if edges:
            # Test edge state update on first edge
            edge = edges[0]
            print(f"\n🔍 Testing edge state update for: {edge.edge_id}")
            print(f"   Protocol: {edge.protocol_name}")
            print(f"   Source: {edge.source_asset_id}")
            print(f"   Target: {edge.target_asset_id}")
            
            # Get the adapter directly to test it
            adapter = pipeline.protocol_adapters.get("uniswap_v3")
            if adapter:
                print(f"   📡 Adapter found: {type(adapter).__name__}")
                print(f"   🔗 Web3 initialized: {adapter.web3 is not None}")
                print(f"   📋 Quoter contract: {adapter.quoter_contract is not None}")
                
                # Get edge config
                config = pipeline.edge_configs.get(edge.edge_id)
                if config:
                    print(f"   📊 Config: {config.pool_metadata}")
                    
                    # Get token info
                    pool_address = config.pool_metadata.get("pool_address")
                    token_info = pipeline._get_pool_token_info(pool_address)
                    print(f"   🪙 Token info: {token_info}")
                    
                    if token_info:
                        # Test the adapter directly
                        metadata_with_tokens = {
                            **config.pool_metadata,
                            **token_info
                        }
                        print(f"   📋 Combined metadata: {metadata_with_tokens}")
                        
                        try:
                            print("   🔧 Calling adapter.update_edge_state directly...")
                            updated_state = await adapter.update_edge_state(edge, metadata_with_tokens)
                            if updated_state:
                                print(f"   ✅ Direct adapter call succeeded")
                                print(f"   📊 New state: rate={updated_state.conversion_rate}, liq=${updated_state.liquidity_usd}")
                            else:
                                print(f"   ❌ Direct adapter call returned None")
                        except Exception as e:
                            print(f"   ❌ Direct adapter call failed: {e}")
                            print(f"   🔍 Traceback: {traceback.format_exc()}")
                    else:
                        print("   ❌ No token info found")
                else:
                    print("   ❌ No config found for edge")
            else:
                print("   ❌ No adapter found")
        
    finally:
        await blockchain_provider.close()


if __name__ == "__main__":
    asyncio.run(debug_edge_state_detailed())