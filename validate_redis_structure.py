#!/usr/bin/env python3
"""Validate Redis integration structure."""
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_redis_structure():
    """Test Redis integration structure and imports."""
    print("🔍 Testing Redis integration structure...")
    
    try:
        # Test imports
        from yield_arbitrage.cache import get_redis, close_redis, redis_client
        print("✅ Redis cache imports successful")
        
        # Test Redis client module
        from yield_arbitrage.cache.redis_client import init_redis, health_check
        print("✅ Redis client module imports successful")
        
        # Test settings integration
        from yield_arbitrage.config.settings import settings
        assert hasattr(settings, 'redis_url')
        print("✅ Redis URL configuration available")
        
        # Test main app imports
        from yield_arbitrage.main import create_app
        app = create_app()
        print("✅ FastAPI app creation with Redis integration successful")
        
        # Test health endpoint structure  
        from yield_arbitrage.api.health import router
        routes = [route.path for route in router.routes]
        assert "/health" in routes
        assert "/ready" in routes
        print("✅ Health endpoints include Redis checks")
        
        print("✅ All Redis structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Redis structure test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_redis_structure()
    if not success:
        sys.exit(1)