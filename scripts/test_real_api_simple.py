#!/usr/bin/env python3
"""Simple test to verify real Tenderly API integration works."""
import asyncio
import sys
import os

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient,
    TenderlyNetworkId,
    TenderlyTransaction
)


async def test_basic_api_functionality():
    """Test that our Tenderly integration works with real API."""
    print("🧪 Testing basic Tenderly API functionality...")
    
    # Test 1: Client initialization
    client = TenderlyClient(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    try:
        await client.initialize()
        print("✅ Client initialized successfully")
        
        # Test 2: Simple transaction simulation (no Virtual TestNet needed)
        print("\n🔄 Testing direct transaction simulation...")
        
        tx = TenderlyTransaction(
            from_address="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # Vitalik's address
            to_address="0x000000000000000000000000000000000000beef",
            value="1000000000000000000",  # 1 ETH
            data="0x"
        )
        
        result = await client.simulate_transaction(
            transaction=tx,
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000
        )
        
        print(f"✅ Direct simulation completed")
        print(f"   Success: {result.success}")
        print(f"   Gas used: {result.gas_used}")
        print(f"   Time: {result.simulation_time_ms:.2f}ms")
        
        if not result.success:
            print(f"   Expected failure reason: {result.revert_reason}")
        
        # Test 3: Virtual TestNet creation and usage
        print("\n🌐 Testing Virtual TestNet workflow...")
        
        testnet = await client.create_virtual_testnet(
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000,
            slug="api-test",
            display_name="API Test"
        )
        
        print(f"✅ Virtual TestNet created: {testnet.testnet_id}")
        
        # Simulate on the Virtual TestNet
        testnet_result = await client.simulate_transaction(
            transaction=tx,
            network_id=TenderlyNetworkId.ETHEREUM,
            testnet_id=testnet.testnet_id
        )
        
        print(f"✅ Virtual TestNet simulation completed")
        print(f"   Success: {testnet_result.success}")
        print(f"   Gas used: {testnet_result.gas_used}")
        
        # Cleanup
        await client.delete_virtual_testnet(testnet.testnet_id)
        print("✅ Virtual TestNet cleaned up")
        
        # Test 4: Statistics
        print("\n📊 Client statistics:")
        stats = client.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await client.close()


async def test_transaction_bundle():
    """Test transaction bundle simulation."""
    print("\n📦 Testing transaction bundle simulation...")
    
    client = TenderlyClient(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    try:
        await client.initialize()
        
        # Create bundle
        tx1 = TenderlyTransaction(
            from_address="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            to_address="0x000000000000000000000000000000000000beef",
            value="500000000000000000",  # 0.5 ETH
        )
        
        tx2 = TenderlyTransaction(
            from_address="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            to_address="0x000000000000000000000000000000000000cafe",
            value="300000000000000000",  # 0.3 ETH
        )
        
        results = await client.simulate_transaction_bundle(
            transactions=[tx1, tx2],
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000
        )
        
        print(f"✅ Bundle simulation: {len(results)} transactions processed")
        total_gas = sum(r.gas_used or 0 for r in results)
        print(f"   Total gas: {total_gas}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bundle test failed: {e}")
        return False
        
    finally:
        await client.close()


if __name__ == "__main__":
    print("🎯 Real Tenderly API Integration Test\n")
    
    # Test basic functionality
    success1 = asyncio.run(test_basic_api_functionality())
    
    # Test bundles
    success2 = asyncio.run(test_transaction_bundle())
    
    print(f"\n📋 Test Results:")
    print(f"  Basic API: {'✅' if success1 else '❌'}")
    print(f"  Bundles: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("\n🎉 All real API tests passed!")
        print("✅ Tenderly integration is working correctly")
        print("🚀 Ready to proceed with Task 6.6!")
    else:
        print("\n🔧 Some tests failed")