#!/usr/bin/env python3
"""Test Tenderly Virtual TestNets with updated client."""
import asyncio
import sys
import os

# Add src to path so we can import our modules
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient, 
    TenderlyNetworkId, 
    TenderlyTransaction
)


async def test_virtual_testnets():
    """Test Virtual TestNets with real Tenderly API."""
    print("🚀 Testing Tenderly Virtual TestNets...")
    
    client = TenderlyClient(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd", 
        project_slug="project"
    )
    
    try:
        # Initialize client
        await client.initialize()
        print("✅ TenderlyClient initialized successfully!")
        
        # Test Virtual TestNet creation (using new method)
        print("\n🌐 Creating Virtual TestNet...")
        testnet = await client.create_virtual_testnet(
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000,
            slug="yield-arb-test",
            display_name="Yield Arbitrage Test"
        )
        print(f"✅ Virtual TestNet created!")
        print(f"  Testnet ID: {testnet.testnet_id}")
        print(f"  Chain ID: {testnet.chain_id}")
        print(f"  Admin RPC: {testnet.admin_rpc_url}")
        print(f"  Public RPC: {testnet.public_rpc_url}")
        
        # Test transaction simulation on Virtual TestNet
        print("\n🧪 Testing transaction simulation...")
        tx = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0x000000000000000000000000000000000000beef",
            value="1000000000000000000",  # 1 ETH in wei
            data="0x"
        )
        
        sim_result = await client.simulate_transaction(
            transaction=tx,
            network_id=TenderlyNetworkId.ETHEREUM,
            testnet_id=testnet.testnet_id
        )
        
        print(f"✅ Transaction simulated successfully!")
        print(f"  Success: {sim_result.success}")
        print(f"  Gas used: {sim_result.gas_used}")
        print(f"  Simulation time: {sim_result.simulation_time_ms}ms")
        
        if not sim_result.success:
            print(f"  Error: {sim_result.error_message}")
            print(f"  Revert reason: {sim_result.revert_reason}")
        
        # Test legacy fork interface (should work with Virtual TestNets)
        print("\n🔄 Testing legacy fork interface...")
        legacy_fork = await client.create_fork(
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000,
            alias="legacy-test"
        )
        print(f"✅ Legacy fork created (actually Virtual TestNet): {legacy_fork.fork_id}")
        
        # Clean up both testnets
        print("\n🗑️ Cleaning up...")
        await client.delete_virtual_testnet(testnet.testnet_id)
        print("✅ Virtual TestNet deleted")
        
        await client.delete_fork(legacy_fork.fork_id)
        print("✅ Legacy fork deleted")
        
        # Test statistics
        print("\n📊 Client statistics:")
        stats = client.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await client.close()


async def test_transaction_bundle():
    """Test transaction bundle simulation on Virtual TestNet."""
    print("\n🎯 Testing transaction bundle simulation...")
    
    client = TenderlyClient(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd", 
        project_slug="project"
    )
    
    try:
        await client.initialize()
        
        # Create Virtual TestNet
        testnet = await client.create_virtual_testnet(
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000,
            slug="bundle-test",
            display_name="Bundle Test"
        )
        
        # Create bundle of transactions
        tx1 = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0x000000000000000000000000000000000000beef",
            value="500000000000000000",  # 0.5 ETH
            data="0x"
        )
        
        tx2 = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0x000000000000000000000000000000000000cafe",
            value="300000000000000000",  # 0.3 ETH
            data="0x"
        )
        
        # Simulate bundle
        results = await client.simulate_transaction_bundle(
            transactions=[tx1, tx2],
            network_id=TenderlyNetworkId.ETHEREUM,
            testnet_id=testnet.testnet_id
        )
        
        print(f"✅ Bundle simulation complete! {len(results)} transactions processed")
        for i, result in enumerate(results):
            print(f"  TX {i+1}: success={result.success}, gas={result.gas_used}")
        
        # Clean up
        await client.delete_virtual_testnet(testnet.testnet_id)
        print("✅ Bundle test cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Bundle test error: {e}")
        return False
        
    finally:
        await client.close()


if __name__ == "__main__":
    print("🌐 Testing Tenderly Virtual TestNets Integration\n")
    
    # Test basic Virtual TestNet operations
    success1 = asyncio.run(test_virtual_testnets())
    
    # Test transaction bundles
    success2 = asyncio.run(test_transaction_bundle())
    
    print(f"\n📋 Test Results:")
    print(f"  Virtual TestNets: {'✅' if success1 else '❌'}")
    print(f"  Transaction Bundles: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("\n🎉 All Virtual TestNet tests passed!")
        print("🚀 Ready to use real Tenderly API with Virtual TestNets!")
    else:
        print("\n🔧 Some tests failed - check output above")