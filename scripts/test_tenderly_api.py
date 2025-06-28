#!/usr/bin/env python3
"""Quick script to test Tenderly API access and discover account details."""
import asyncio
import aiohttp
import json


async def test_tenderly_api():
    """Test Tenderly API access and try to discover account details."""
    api_key = "E5tSD537G0z2r9xur64acExE2DNjRFWP"
    base_url = "https://api.tenderly.co/api/v1"
    
    headers = {
        "X-Access-Key": api_key,
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        print("🔑 Testing Tenderly API key...")
        
        # Test 1: Try to get account info
        try:
            async with session.get(f"{base_url}/account") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ API key is valid!")
                    print(f"Account data: {json.dumps(data, indent=2)}")
                    return data
                else:
                    print(f"❌ API key test failed: {response.status}")
                    text = await response.text()
                    print(f"Error: {text}")
        except Exception as e:
            print(f"❌ Error testing API key: {e}")
        
        # Test 2: Try to list projects (this might give us usernames/projects)
        try:
            print("\n🔍 Trying to discover projects...")
            async with session.get(f"{base_url}/account/projects") as response:
                if response.status == 200:
                    data = await response.json()
                    print("📋 Available projects:")
                    print(json.dumps(data, indent=2))
                    return data
                else:
                    print(f"❌ Projects list failed: {response.status}")
                    text = await response.text()
                    print(f"Error: {text}")
        except Exception as e:
            print(f"❌ Error listing projects: {e}")
        
        # Test 3: Try some common endpoints to see what works
        endpoints_to_try = [
            "/user",
            "/account/me", 
            "/projects",
            "/account/settings"
        ]
        
        print("\n🧪 Testing various endpoints...")
        for endpoint in endpoints_to_try:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    print(f"{endpoint}: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"  Data: {json.dumps(data, indent=2)[:200]}...")
            except Exception as e:
                print(f"{endpoint}: Error - {e}")
        
        return None


async def test_simple_simulation():
    """Test a simple transaction simulation."""
    api_key = "E5tSD537G0z2r9xur64acExE2DNjRFWP"
    base_url = "https://api.tenderly.co/api/v1"
    
    headers = {
        "X-Access-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Simple ETH transfer transaction
    simulation_data = {
        "network_id": "1",  # Ethereum mainnet
        "from": "0x000000000000000000000000000000000000dead",
        "to": "0x000000000000000000000000000000000000beef", 
        "value": "1000000000000000000",  # 1 ETH
        "gas": 21000,
        "gas_price": "20000000000",  # 20 gwei
        "save": True,
        "simulation_type": "quick"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        print("\n🧪 Testing transaction simulation...")
        
        try:
            # Note: We need username and project_slug for the URL
            # Let's try a generic simulation endpoint first
            async with session.post(f"{base_url}/simulate", json=simulation_data) as response:
                print(f"Simulation response: {response.status}")
                text = await response.text()
                print(f"Response: {text[:500]}...")
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Simulation successful!")
                    return data
                
        except Exception as e:
            print(f"❌ Simulation error: {e}")
    
    return None


if __name__ == "__main__":
    print("🚀 Testing Tenderly API integration...")
    
    # Test API access
    account_data = asyncio.run(test_tenderly_api())
    
    # Test simulation
    sim_data = asyncio.run(test_simple_simulation())
    
    print("\n📋 Summary:")
    if account_data:
        print("✅ Account access working")
    else:
        print("❌ Need to determine correct username/project")
        
    if sim_data:
        print("✅ Simulation working") 
    else:
        print("❌ Simulation needs correct endpoint format")