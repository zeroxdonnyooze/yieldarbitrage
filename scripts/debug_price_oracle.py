#!/usr/bin/env python3
"""Debug script for price oracle issues."""
import asyncio
import sys
import logging
from unittest.mock import AsyncMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def debug_price_oracle():
    """Debug the price oracle."""
    print("🔍 Debugging Price Oracle...")
    
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
        
        # Test price fetching
        test_assets = ["ETH_MAINNET_WETH", "ETH_MAINNET_USDC", "ETH_MAINNET_USDT", "ETH_MAINNET_DAI"]
        
        for asset in test_assets:
            try:
                print(f"\n🔍 Testing {asset}...")
                price = await oracle.get_price_usd(asset)
                print(f"   Price: {price} (type: {type(price)})")
                
                if price is not None:
                    print(f"   ✅ Success: ${price:.2f}")
                else:
                    print(f"   ❌ Failed: None returned")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
    finally:
        await blockchain_provider.close()


if __name__ == "__main__":
    asyncio.run(debug_price_oracle())