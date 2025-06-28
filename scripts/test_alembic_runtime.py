#!/usr/bin/env python3
"""Test script to validate Alembic runtime configuration."""

import os
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_alembic_command(cmd_args: list[str], description: str) -> bool:
    """Test running an Alembic command."""
    print(f"🔧 Testing: {description}")
    try:
        # Run alembic command
        result = subprocess.run(
            ['python', '-m', 'alembic'] + cmd_args,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print(f"   Stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"   Stderr: {result.stderr.strip()}")
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def test_alembic_configuration():
    """Test basic Alembic configuration commands."""
    print("🚀 Testing Alembic configuration...")
    print("=" * 60)
    
    test_results = []
    
    # Test different Alembic commands
    tests = [
        (["--help"], "Help command"),
        (["current"], "Show current revision"),
        (["history"], "Show revision history"),
        (["heads"], "Show head revisions"),
    ]
    
    for cmd_args, description in tests:
        print(f"\n📋 Running: alembic {' '.join(cmd_args)}")
        print("-" * 40)
        
        result = test_alembic_command(cmd_args, description)
        test_results.append((description, result))
    
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
        print("🎉 All tests passed! Alembic runtime is working correctly.")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed. Please check the configuration.")
        return False


def test_metadata_detection():
    """Test that Alembic can detect our models in metadata."""
    print("\n🔍 Testing metadata detection...")
    
    try:
        # Import our models to ensure they're in metadata
        from yield_arbitrage.database.connection import Base
        from yield_arbitrage.database.models import ExecutedPath, TokenMetadata
        
        print(f"📋 Available tables in metadata:")
        for table_name in Base.metadata.tables:
            print(f"   - {table_name}")
        
        required_tables = ['executed_paths', 'token_metadata']
        missing_tables = [t for t in required_tables if t not in Base.metadata.tables]
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        print("✅ All required tables found in metadata")
        return True
        
    except Exception as e:
        print(f"❌ Metadata detection failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Alembic runtime tests...")
    
    # Test metadata detection first
    metadata_ok = test_metadata_detection()
    if not metadata_ok:
        print("💥 Metadata test failed, skipping runtime tests")
        return False
    
    # Test Alembic runtime
    runtime_ok = test_alembic_configuration()
    
    if metadata_ok and runtime_ok:
        print("\n🎉 All Alembic tests passed! Configuration is ready for migrations.")
        return True
    else:
        print("\n💥 Some tests failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)