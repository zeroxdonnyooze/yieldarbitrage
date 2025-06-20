#!/usr/bin/env python3
"""Quick validation script for database connection."""
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.database import get_db, create_tables
from sqlalchemy import text

async def test_database():
    """Test database connection and setup."""
    print("🔍 Testing database connection...")
    
    try:
        # Test database connection via dependency
        async for db in get_db():
            result = await db.execute(text("SELECT 1 as test"))
            row = result.scalar()
            if row == 1:
                print("✅ Database connection successful")
            else:
                print("❌ Database connection failed - unexpected result")
                return False
            break
        
        # Test table creation
        print("📊 Testing table creation...")
        await create_tables()
        print("✅ Table creation successful")
        
        print("✅ All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_database())
    if not success:
        sys.exit(1)