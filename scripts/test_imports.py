#!/usr/bin/env python3
"""Test script to catch import errors before deployment."""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all critical imports."""
    errors = []
    
    # Test main app import
    try:
        from yield_arbitrage.main import app
        print("✅ Main app import successful")
    except Exception as e:
        errors.append(f"Main app: {e}")
        print(f"❌ Main app import failed: {e}")
    
    # Test health router
    try:
        from yield_arbitrage.api.health import router
        print("✅ Health router import successful")
    except Exception as e:
        errors.append(f"Health router: {e}")
        print(f"❌ Health router import failed: {e}")
    
    # Test database components
    try:
        from yield_arbitrage.database import startup_database, shutdown_database
        print("✅ Database components import successful")
    except Exception as e:
        errors.append(f"Database: {e}")
        print(f"❌ Database import failed: {e}")
    
    # Test pathfinding models
    try:
        from yield_arbitrage.pathfinding.path_models import YieldPath
        print("✅ YieldPath import successful")
    except Exception as e:
        errors.append(f"YieldPath: {e}")
        print(f"❌ YieldPath import failed: {e}")
    
    return errors

if __name__ == "__main__":
    print("🔍 Testing imports...")
    errors = test_imports()
    
    if not errors:
        print("\n🎉 All imports successful! Ready for deployment.")
        sys.exit(0)
    else:
        print(f"\n❌ Found {len(errors)} import errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)