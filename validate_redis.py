#!/usr/bin/env python3
"""Quick validation script for Redis connection."""
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.cache import get_redis, close_redis

async def test_redis():
    """Test Redis connection and basic operations."""
    print("🔍 Testing Redis connection...")
    
    try:
        # Test Redis connection
        redis_client = await get_redis()
        print("✅ Redis connection successful")
        
        # Test basic operations
        test_key = "test:validation"
        test_value = "Hello Redis!"
        
        # Set a value
        await redis_client.set(test_key, test_value, ex=60)  # Expire in 60 seconds
        print("✅ Redis SET operation successful")
        
        # Get the value
        retrieved_value = await redis_client.get(test_key)
        if retrieved_value == test_value:
            print("✅ Redis GET operation successful")
        else:
            print(f"❌ Redis GET failed - expected '{test_value}', got '{retrieved_value}'")
            return False
        
        # Clean up
        await redis_client.delete(test_key)
        print("✅ Redis DELETE operation successful")
        
        # Test ping
        pong = await redis_client.ping()
        if pong:
            print("✅ Redis PING successful")
        else:
            print("❌ Redis PING failed")
            return False
        
        print("✅ All Redis tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False
    finally:
        await close_redis()

if __name__ == "__main__":
    success = asyncio.run(test_redis())
    if not success:
        sys.exit(1)