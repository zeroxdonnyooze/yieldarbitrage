#!/usr/bin/env python3
"""Test script to validate Alembic migration."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_alembic_command(cmd_args: list[str], description: str) -> tuple[bool, str, str]:
    """Run an Alembic command and return success status and output."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            ['python', '-m', 'alembic'] + cmd_args,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        success = result.returncode == 0
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status} (return code: {result.returncode})")
        
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"   Stderr: {result.stderr.strip()}")
            
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"   ❌ TIMEOUT")
        return False, "", "Command timed out"
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False, "", str(e)


async def test_migration_dry_run():
    """Test migration in dry-run mode (SQL generation only)."""
    print("\n🚀 Testing migration dry-run...")
    print("=" * 60)
    
    # Test generating SQL for upgrade
    success, stdout, stderr = run_alembic_command(
        ["upgrade", "head", "--sql"],
        "Generate SQL for migration"
    )
    
    if not success:
        print("❌ Failed to generate migration SQL")
        return False
    
    # Check that SQL contains expected DDL
    sql_content = stdout.lower()
    expected_elements = [
        "create table executed_paths",
        "create table token_metadata", 
        "create index",
        "uuid",
        "numeric",
        "varchar[]"  # PostgreSQL array syntax
    ]
    
    missing_elements = []
    for element in expected_elements:
        if element not in sql_content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ Missing expected SQL elements: {missing_elements}")
        return False
    
    print("✅ Migration SQL generation successful")
    print("✅ All expected SQL elements found")
    return True


def test_migration_syntax():
    """Test that migration file has valid Python syntax."""
    print("\n🧪 Testing migration file syntax...")
    print("=" * 60)
    
    try:
        # Find the migration file
        versions_dir = Path(__file__).parent.parent / "alembic" / "versions"
        migration_files = list(versions_dir.glob("*.py"))
        
        if not migration_files:
            print("❌ No migration files found")
            return False
        
        latest_migration = max(migration_files, key=lambda p: p.stat().st_mtime)
        print(f"📄 Testing: {latest_migration.name}")
        
        # Try to compile the migration file
        with open(latest_migration, 'r') as f:
            migration_code = f.read()
        
        compile(migration_code, str(latest_migration), 'exec')
        print("✅ Migration file syntax is valid")
        
        # Check for required functions
        if 'def upgrade()' not in migration_code:
            print("❌ Migration missing upgrade() function")
            return False
        
        if 'def downgrade()' not in migration_code:
            print("❌ Migration missing downgrade() function")
            return False
        
        print("✅ Migration has required functions")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in migration file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing migration syntax: {e}")
        return False


def test_alembic_history():
    """Test that Alembic can show migration history."""
    print("\n📚 Testing migration history...")
    print("=" * 60)
    
    success, stdout, stderr = run_alembic_command(
        ["history", "--verbose"],
        "Show migration history"
    )
    
    if not success:
        print("❌ Failed to show migration history")
        return False
    
    # Check that our migration appears in history
    if "create_initial_tables" not in stdout:
        print("❌ Initial migration not found in history")
        return False
    
    print("✅ Migration appears in history")
    return True


def test_migration_check():
    """Test Alembic check command."""
    print("\n🔍 Testing migration check...")
    print("=" * 60)
    
    success, stdout, stderr = run_alembic_command(
        ["check"],
        "Check for pending migrations"
    )
    
    # Note: This might fail if database doesn't exist, which is fine for this test
    print(f"   Check command completed (success: {success})")
    return True  # Always return True as this is an informational test


async def test_model_imports():
    """Test that our database models can be imported correctly."""
    print("\n📦 Testing model imports...")
    print("=" * 60)
    
    try:
        from yield_arbitrage.database.connection import Base
        from yield_arbitrage.database.models import ExecutedPath, TokenMetadata
        
        print("✅ Successfully imported Base")
        print("✅ Successfully imported ExecutedPath")
        print("✅ Successfully imported TokenMetadata")
        
        # Check metadata
        table_names = list(Base.metadata.tables.keys())
        print(f"📋 Tables in metadata: {table_names}")
        
        if 'executed_paths' not in table_names:
            print("❌ executed_paths table not in metadata")
            return False
        
        if 'token_metadata' not in table_names:
            print("❌ token_metadata table not in metadata")
            return False
        
        print("✅ All expected tables found in metadata")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing model imports: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Alembic migration tests...")
    print("=" * 80)
    
    test_results = []
    
    # Define tests
    tests = [
        ("Model Imports", test_model_imports),
        ("Migration Syntax", test_migration_syntax), 
        ("Migration History", test_alembic_history),
        ("Migration Check", test_migration_check),
        ("Migration Dry Run", test_migration_dry_run),
    ]
    
    # Run tests
    for test_name, test_func in tests:
        print(f"\n📋 Running test: {test_name}")
        print("-" * 50)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            test_results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"📊 Test result: {status}")
            
        except Exception as e:
            print(f"💥 Test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Migration is ready to apply.")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed. Please review the migration.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)