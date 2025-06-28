#!/usr/bin/env python3
"""Test script to validate Alembic configuration."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alembic import command
from alembic.config import Config
from yield_arbitrage.database.connection import Base
from yield_arbitrage.database.models import ExecutedPath, TokenMetadata


def test_alembic_configuration():
    """Test that Alembic configuration is properly set up."""
    print("🔧 Testing Alembic configuration...")
    
    try:
        # Get the alembic.ini path
        project_root = Path(__file__).parent.parent
        alembic_ini_path = project_root / "alembic.ini"
        
        if not alembic_ini_path.exists():
            print(f"❌ alembic.ini not found at {alembic_ini_path}")
            return False
        
        print(f"✅ Found alembic.ini at {alembic_ini_path}")
        
        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini_path))
        
        # Test that we can access the script location
        script_location = alembic_cfg.get_main_option("script_location")
        script_path = project_root / script_location
        
        if not script_path.exists():
            print(f"❌ Script location not found: {script_path}")
            return False
        
        print(f"✅ Script location found: {script_path}")
        
        # Test that env.py exists and is configured
        env_py_path = script_path / "env.py"
        if not env_py_path.exists():
            print(f"❌ env.py not found at {env_py_path}")
            return False
        
        print(f"✅ env.py found at {env_py_path}")
        
        # Test metadata import
        print("📋 Testing metadata and model imports...")
        print(f"   Base metadata tables: {list(Base.metadata.tables.keys())}")
        print(f"   ExecutedPath table: {'executed_paths' in Base.metadata.tables}")
        print(f"   TokenMetadata table: {'token_metadata' in Base.metadata.tables}")
        
        if 'executed_paths' not in Base.metadata.tables:
            print("❌ ExecutedPath model not found in metadata")
            return False
        
        if 'token_metadata' not in Base.metadata.tables:
            print("❌ TokenMetadata model not found in metadata")
            return False
        
        print("✅ All required models found in metadata")
        
        # Test file template configuration
        file_template = alembic_cfg.get_main_option("file_template")
        if file_template:
            print(f"✅ File template configured: {file_template}")
        else:
            print("ℹ️  Using default file template")
        
        print("✅ Alembic configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Alembic configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alembic_env_imports():
    """Test that alembic env.py can import our models successfully."""
    print("\n🧪 Testing Alembic env.py imports...")
    
    try:
        # Test importing the env module
        project_root = Path(__file__).parent.parent
        alembic_path = project_root / "alembic"
        
        # Add alembic to path temporarily
        sys.path.insert(0, str(alembic_path))
        
        # Import env module
        import env
        
        # Check that target_metadata is set
        if not hasattr(env, 'target_metadata'):
            print("❌ target_metadata not found in env.py")
            return False
        
        if env.target_metadata is None:
            print("❌ target_metadata is None")
            return False
        
        print(f"✅ target_metadata found with {len(env.target_metadata.tables)} tables")
        print(f"   Tables: {list(env.target_metadata.tables.keys())}")
        
        # Check that config is set up
        if not hasattr(env, 'config'):
            print("❌ config not found in env.py")
            return False
        
        print("✅ Alembic config found")
        
        # Check database URL
        db_url = env.config.get_main_option("sqlalchemy.url")
        print(f"✅ Database URL configured: {db_url}")
        
        print("✅ Alembic env.py imports test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Alembic env.py imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Remove alembic from path
        if str(alembic_path) in sys.path:
            sys.path.remove(str(alembic_path))


def main():
    """Main test function."""
    print("🚀 Starting Alembic configuration tests...")
    print("=" * 60)
    
    test_results = []
    
    # Run configuration tests
    tests = [
        ("Alembic Configuration", test_alembic_configuration),
        ("Alembic Env Imports", test_alembic_env_imports),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running test: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            test_results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"📊 Test result: {status}")
            
        except Exception as e:
            print(f"💥 Test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Alembic is properly configured.")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)