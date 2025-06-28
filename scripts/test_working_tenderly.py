#!/usr/bin/env python3
"""Test Tenderly API with correct credentials."""
import asyncio
import aiohttp
import json


async def test_real_tenderly_simulation():
    """Test real Tenderly simulation with correct credentials."""
    api_key = "E5tSD537G0z2r9xur64acExE2DNjRFWP"
    username = "bomanyd"
    project_slug = "project"  # Found from discovery
    base_url = "https://api.tenderly.co/api/v1"
    
    headers = {
        "X-Access-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Simple ETH transfer for testing
    simulation_data = {
        "network_id": "1",  # Ethereum mainnet
        "from": "0x000000000000000000000000000000000000dead",
        "to": "0x000000000000000000000000000000000000beef",
        "value": "1000000000000000000",  # 1 ETH in wei
        "gas": 21000,
        "gas_price": "20000000000",  # 20 gwei
        "save": True,
        "simulation_type": "full"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        print("🧪 Testing transaction simulation with correct credentials...")
        
        url = f"{base_url}/account/{username}/project/{project_slug}/simulate"
        print(f"URL: {url}")
        
        try:
            async with session.post(url, json=simulation_data) as response:
                print(f"Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Simulation successful!")
                    
                    # Extract key information
                    transaction = data.get("transaction", {})
                    print(f"Transaction successful: {transaction.get('status', 'unknown')}")
                    print(f"Gas used: {transaction.get('gas_used', 'unknown')}")
                    print(f"Transaction hash: {transaction.get('hash', 'none')}")
                    
                    if transaction.get("error_message"):
                        print(f"Error message: {transaction['error_message']}")
                    
                    return True
                    
                else:
                    text = await response.text()
                    print(f"❌ Simulation failed: {response.status}")
                    print(f"Error: {text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error during simulation: {e}")
            return False


async def test_fork_creation():
    """Test creating and deleting a fork."""
    api_key = "E5tSD537G0z2r9xur64acExE2DNjRFWP"
    username = "bomanyd"
    project_slug = "project"
    base_url = "https://api.tenderly.co/api/v1"
    
    headers = {
        "X-Access-Key": api_key,
        "Content-Type": "application/json"
    }
    
    fork_data = {
        "network_id": "1",
        "block_number": 18500000,  # Recent mainnet block
        "alias": "test-fork-integration",
        "description": "Test fork for yield arbitrage integration"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        print("\n🍴 Testing fork creation...")
        
        # Create fork
        create_url = f"{base_url}/account/{username}/project/{project_slug}/fork"
        print(f"Create URL: {create_url}")
        
        try:
            async with session.post(create_url, json=fork_data) as response:
                print(f"Create fork response: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Fork created successfully!")
                    
                    # Extract fork ID
                    root_tx = data.get("root_transaction", {})
                    fork_id = root_tx.get("fork_id")
                    
                    if fork_id:
                        print(f"Fork ID: {fork_id}")
                        print(f"Block number: {root_tx.get('block_number')}")
                        
                        # Now delete the fork
                        print("\n🗑️ Cleaning up fork...")
                        delete_url = f"{base_url}/account/{username}/project/{project_slug}/fork/{fork_id}"
                        
                        async with session.delete(delete_url) as delete_response:
                            print(f"Delete fork response: {delete_response.status}")
                            if delete_response.status in [200, 204]:
                                print("✅ Fork deleted successfully!")
                                return True
                            else:
                                delete_text = await delete_response.text()
                                print(f"❌ Delete failed: {delete_text}")
                    
                    return True
                    
                else:
                    text = await response.text()
                    print(f"❌ Fork creation failed: {response.status}")
                    print(f"Error: {text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error during fork operations: {e}")
            return False


async def test_tenderly_client_integration():
    """Test our TenderlyClient with real credentials."""
    import sys
    import os
    
    # Add src to path so we can import our modules
    sys.path.append('/home/david/projects/yieldarbitrage/src')
    
    from yield_arbitrage.execution.tenderly_client import TenderlyClient, TenderlyNetworkId, TenderlyTransaction
    
    print("\n🔧 Testing our TenderlyClient class...")
    
    client = TenderlyClient(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd", 
        project_slug="project"
    )
    
    try:
        # Initialize client
        await client.initialize()
        print("✅ TenderlyClient initialized successfully!")
        
        # Test fork creation
        fork = await client.create_fork(
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000,
            description="TenderlyClient integration test"
        )
        print(f"✅ Fork created via TenderlyClient: {fork.fork_id}")
        
        # Test transaction simulation
        tx = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0x000000000000000000000000000000000000beef",
            value="1000000000000000000",
            data="0x"
        )
        
        sim_result = await client.simulate_transaction(
            transaction=tx,
            network_id=TenderlyNetworkId.ETHEREUM,
            fork_id=fork.fork_id
        )
        
        print(f"✅ Transaction simulated: success={sim_result.success}, gas={sim_result.gas_used}")
        
        # Clean up
        await client.delete_fork(fork.fork_id)
        print("✅ Fork cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ TenderlyClient error: {e}")
        return False
        
    finally:
        await client.close()


if __name__ == "__main__":
    print("🚀 Testing Tenderly API with real credentials...")
    
    # Test simulation
    sim_success = asyncio.run(test_real_tenderly_simulation())
    
    # Test fork operations  
    fork_success = asyncio.run(test_fork_creation())
    
    # Test our client class
    client_success = asyncio.run(test_tenderly_client_integration())
    
    print(f"\n📋 Test Results:")
    print(f"  Simulation: {'✅' if sim_success else '❌'}")
    print(f"  Fork ops:   {'✅' if fork_success else '❌'}")
    print(f"  Client:     {'✅' if client_success else '❌'}")
    
    if all([sim_success, fork_success, client_success]):
        print("\n🎉 All Tenderly integration tests passed!")
        print("Ready to update configurations with real credentials!")
    else:
        print("\n🔧 Some tests failed - need to debug further")