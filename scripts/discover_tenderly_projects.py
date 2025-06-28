#!/usr/bin/env python3
"""Discover Tenderly projects for the user."""
import asyncio
import aiohttp
import json


async def discover_projects():
    """Discover available Tenderly projects."""
    api_key = "E5tSD537G0z2r9xur64acExE2DNjRFWP"
    username = "bomanyd"  # Found from previous test
    base_url = "https://api.tenderly.co/api/v1"
    
    headers = {
        "X-Access-Key": api_key,
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        print(f"🔍 Discovering projects for user: {username}")
        
        # Try various project-related endpoints
        endpoints_to_try = [
            f"/account/{username}",
            f"/account/{username}/projects",
            f"/account/{username}/project",
            "/user/projects",
            "/user/accounts",
            "/accounts",
        ]
        
        for endpoint in endpoints_to_try:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    print(f"\n{endpoint}: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Success! Data:")
                        print(json.dumps(data, indent=2))
                        
                        # Look for project information
                        if isinstance(data, dict):
                            if "projects" in data:
                                print("📋 Found projects!")
                                return data["projects"]
                            elif "project" in data:
                                print("📋 Found project!")
                                return [data["project"]]
                        elif isinstance(data, list):
                            print("📋 Found list of items!")
                            return data
                            
                    else:
                        text = await response.text()
                        print(f"❌ {response.status}: {text[:100]}...")
            except Exception as e:
                print(f"❌ Error with {endpoint}: {e}")
        
        # If no projects found, let's try creating one for testing
        print(f"\n🏗️ No existing projects found. Let's try creating a test project...")
        
        create_project_data = {
            "name": "yield-arbitrage-test",
            "description": "Test project for yield arbitrage system",
            "settings": {
                "public": False
            }
        }
        
        try:
            async with session.post(f"{base_url}/account/{username}/projects", json=create_project_data) as response:
                print(f"Create project response: {response.status}")
                if response.status in [200, 201]:
                    data = await response.json()
                    print("✅ Project created successfully!")
                    print(json.dumps(data, indent=2))
                    return [data]
                else:
                    text = await response.text()
                    print(f"❌ Create failed: {text}")
        except Exception as e:
            print(f"❌ Error creating project: {e}")
        
        return None


async def test_simulation_with_user_project():
    """Test simulation with discovered user/project."""
    api_key = "E5tSD537G0z2r9xur64acExE2DNjRFWP" 
    username = "bomanyd"
    project_slug = "yield-arbitrage-test"  # We'll try this
    base_url = "https://api.tenderly.co/api/v1"
    
    headers = {
        "X-Access-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Simple ETH transfer for testing
    simulation_data = {
        "network_id": "1",
        "from": "0x000000000000000000000000000000000000dead",
        "to": "0x000000000000000000000000000000000000beef",
        "value": "1000000000000000000",  # 1 ETH
        "gas": 21000,
        "gas_price": "20000000000",
        "save": True
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        print(f"\n🧪 Testing simulation with user={username}, project={project_slug}")
        
        # Try the correct endpoint format
        url = f"{base_url}/account/{username}/project/{project_slug}/simulate"
        
        try:
            async with session.post(url, json=simulation_data) as response:
                print(f"Simulation URL: {url}")
                print(f"Response: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Simulation successful!")
                    print(json.dumps(data, indent=2)[:1000] + "...")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Failed: {text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


if __name__ == "__main__":
    print("🚀 Discovering Tenderly projects...")
    
    # Discover projects
    projects = asyncio.run(discover_projects())
    
    if projects:
        print(f"\n✅ Found {len(projects)} project(s)!")
        for proj in projects:
            print(f"  - {proj}")
    else:
        print("\n❌ No projects found")
    
    # Test simulation
    success = asyncio.run(test_simulation_with_user_project())
    
    if success:
        print("\n🎉 Ready to use Tenderly API!")
    else:
        print("\n🔧 Still need to configure projects correctly")