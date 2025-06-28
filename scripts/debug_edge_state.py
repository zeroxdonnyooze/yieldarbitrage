#!/usr/bin/env python3
"""Debug script for edge state collection issues."""
import asyncio
import sys
import logging
from unittest.mock import AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.data.real_edge_pipeline import RealEdgeStatePipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def debug_edge_state():
    """Debug the edge state collection."""
    print("🔍 Debugging Edge State Collection...")
    
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
            
            try:
                updated_edge = await pipeline.update_edge_state(edge)
                if updated_edge:
                    print(f"   ✅ Edge state updated successfully")
                    print(f"   📊 New state: {updated_edge.state}")
                else:
                    print(f"   ❌ Edge state update returned None")
            except Exception as e:
                print(f"   ❌ Edge state update failed: {e}")
                import traceback
                print(f"   🔍 Traceback: {traceback.format_exc()}")
        
    finally:
        await blockchain_provider.close()


if __name__ == "__main__":
    asyncio.run(debug_edge_state())