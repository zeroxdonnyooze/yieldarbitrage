#!/usr/bin/env python3
"""Final verification that everything works correctly."""
import asyncio
import sys

sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient,
    TenderlyNetworkId,
    TenderlyTransaction
)


async def main():
    """Final verification test."""
    print("🎯 Final Verification Test\n")
    
    client = TenderlyClient(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )
    
    try:
        await client.initialize()
        print("✅ Tenderly client initialized")
        
        # Create Virtual TestNet
        testnet = await client.create_virtual_testnet(
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000,
            slug="final-test",
            display_name="Final Test"
        )
        print(f"✅ Virtual TestNet created: {testnet.testnet_id}")
        
        # Test with smaller amount that should work
        tx = TenderlyTransaction(
            from_address="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            to_address="0x000000000000000000000000000000000000beef",
            value="100000000000000000",  # 0.1 ETH (should be available)
            data="0x"
        )
        
        result = await client.simulate_transaction(
            transaction=tx,
            network_id=TenderlyNetworkId.ETHEREUM,
            testnet_id=testnet.testnet_id
        )
        
        print(f"✅ Transaction simulation on Virtual TestNet")
        print(f"   Success: {result.success}")
        print(f"   Gas used: {result.gas_used}")
        
        # Cleanup
        await client.delete_virtual_testnet(testnet.testnet_id)
        print("✅ Virtual TestNet cleaned up")
        
        # Final stats
        stats = client.get_stats()
        print(f"\n📊 Final statistics:")
        print(f"   Simulations run: {stats['simulations_run']}")
        print(f"   Virtual TestNets created: {stats['testnets_created']}")
        print(f"   API errors: {stats['api_errors']}")
        
        success = result.success and stats['api_errors'] == 0
        print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}: Real Tenderly API integration is {'working perfectly' if success else 'having issues'}!")
        
        return success
        
    except Exception as e:
        print(f"❌ Final test failed: {e}")
        return False
        
    finally:
        await client.close()


if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        print("\n🚀 READY TO PROCEED WITH TASK 6.6!")
        print("✅ All Tenderly integration components working correctly")
        print("✅ Virtual TestNets operational")
        print("✅ Transaction simulation functional")
        print("✅ API credentials configured properly")
    else:
        print("\n🔧 Need to resolve issues before proceeding")