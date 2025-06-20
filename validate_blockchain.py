#!/usr/bin/env python3
"""Validation script for blockchain provider with real RPC connections."""
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.blockchain_connector import BlockchainProvider


async def test_blockchain_provider():
    """Test blockchain provider with real RPC connections."""
    print("🔗 Testing blockchain provider...")
    
    provider = BlockchainProvider()
    
    try:
        # Initialize provider
        print("🔄 Initializing blockchain provider...")
        await provider.initialize()
        
        # Get supported chains
        chains = await provider.get_supported_chains()
        print(f"✅ Supported chains: {chains}")
        
        if not chains:
            print("⚠️ No chains configured - check your RPC URLs in .env")
            return False
        
        # Test each chain
        for chain_name in chains:
            print(f"\n🔍 Testing {chain_name}...")
            
            # Test connection
            connected = await provider.is_connected(chain_name)
            print(f"  Connected: {connected}")
            
            if connected:
                # Test basic operations
                block_number = await provider.get_block_number(chain_name)
                print(f"  Block number: {block_number}")
                
                gas_price = await provider.get_gas_price(chain_name)
                if gas_price:
                    gas_price_gwei = gas_price / 1e9
                    print(f"  Gas price: {gas_price_gwei:.2f} Gwei")
                else:
                    print(f"  Gas price: None")
                
                # Test health check
                health = await provider.get_chain_health(chain_name)
                print(f"  Health: {health['status']}")
        
        # Test overall health
        print(f"\n📊 Overall chain health:")
        all_health = await provider.get_all_chain_health()
        for chain, health in all_health.items():
            status = health.get('status', 'unknown')
            block = health.get('block_number', 'N/A')
            print(f"  {chain}: {status} (block: {block})")
        
        print("\n✅ All blockchain provider tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Blockchain provider test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        await provider.close()


async def test_specific_operations():
    """Test specific blockchain operations."""
    print("\n🧪 Testing specific blockchain operations...")
    
    provider = BlockchainProvider()
    
    try:
        await provider.initialize()
        chains = await provider.get_supported_chains()
        
        if not chains:
            print("⚠️ No chains available for testing")
            return False
        
        # Test with first available chain
        chain_name = chains[0]
        print(f"Using {chain_name} for detailed testing...")
        
        # Test getting Web3 instance
        w3 = await provider.get_web3(chain_name)
        if w3:
            print(f"✅ Web3 instance retrieved for {chain_name}")
            
            # Test some Web3 operations directly
            is_connected = await w3.is_connected()
            print(f"  Direct is_connected(): {is_connected}")
            
            if is_connected:
                chain_id = await w3.eth.chain_id
                print(f"  Chain ID: {chain_id}")
                
                # Test with a known address (Vitalik's address)
                vitalik_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
                balance = await provider.get_balance(chain_name, vitalik_address)
                if balance is not None:
                    balance_eth = balance / 1e18
                    print(f"  Vitalik's balance: {balance_eth:.4f} ETH")
                else:
                    print(f"  Could not fetch balance")
        else:
            print(f"❌ Failed to get Web3 instance for {chain_name}")
            return False
        
        print("✅ Specific operations test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Specific operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await provider.close()


if __name__ == "__main__":
    async def main():
        success1 = await test_blockchain_provider()
        success2 = await test_specific_operations()
        
        if success1 and success2:
            print("\n🎉 All blockchain tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some blockchain tests failed!")
            sys.exit(1)
    
    asyncio.run(main())