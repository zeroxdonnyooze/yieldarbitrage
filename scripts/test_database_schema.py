#!/usr/bin/env python3
"""Test script to validate database schema with real PostgreSQL."""

import asyncio
import os
import sys
from decimal import Decimal
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yield_arbitrage.database import (
    AsyncSessionLocal, Base, create_tables, close_db, engine,
    ExecutedPath, TokenMetadata
)


async def test_database_schema():
    """Test database schema creation and basic operations."""
    print("🔧 Testing database schema...")
    
    try:
        # Drop and recreate all tables for clean test
        print("📝 Creating database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Tables created successfully")
        
        # Test ExecutedPath creation
        print("🧪 Testing ExecutedPath model...")
        async with AsyncSessionLocal() as session:
            now = datetime.now(timezone.utc)
            
            # Create test ExecutedPath
            path = ExecutedPath(
                path_hash="testhash1234567890123456789012345678901234567890123456789012345",
                transaction_hash="0xtest1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                block_number=19000000,
                chain_name="ethereum",
                edge_ids=["ETH_UNISWAP_TRADE_WETH_USDC"],
                edge_types=["TRADE"],
                protocols=["UniswapV3"],
                input_amount=Decimal("1000000000000000000"),  # 1 ETH
                output_amount=Decimal("2000000000"),  # 2000 USDC
                input_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                output_token="0xA0b86a33E6441b9435B654f6D26Cc98b6e1D0a3A",
                profit_usd=Decimal("25.50"),
                profit_percentage=Decimal("0.0125"),  # 1.25%
                effective_rate=Decimal("2000.0"),
                gas_used=200000,
                gas_price_gwei=Decimal("25.0"),
                gas_cost_usd=Decimal("10.00"),
                execution_time_ms=1200,
                slippage_actual=Decimal("0.005"),  # 0.5%
                mev_protected=True,
                flash_loan_used=False,
                ml_confidence_score=Decimal("0.8500"),
                discovered_at=now,
                executed_at=now
            )
            
            session.add(path)
            await session.commit()
            await session.refresh(path)
            
            print(f"✅ ExecutedPath created: {path}")
            print(f"   ID: {path.id}")
            print(f"   Profit: ${path.profit_usd}")
            print(f"   Status: {path.status}")
        
        # Test TokenMetadata creation
        print("🪙 Testing TokenMetadata model...")
        async with AsyncSessionLocal() as session:
            now = datetime.now(timezone.utc)
            
            # Create test TokenMetadata
            token = TokenMetadata(
                asset_id="ETH_MAINNET_0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                chain_name="ethereum",
                contract_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                symbol="WETH",
                name="Wrapped Ether",
                decimals=18,
                is_wrapped=True,
                price_usd=Decimal("2500.75"),
                price_eth=Decimal("1.0"),
                market_cap_usd=Decimal("30000000000.00"),
                volume_24h_usd=Decimal("2000000000.00"),
                liquidity_usd=Decimal("800000000.00"),
                security_score=Decimal("0.9500"),
                liquidity_score=Decimal("0.9800"),
                reliability_score=Decimal("0.9900"),
                protocols=["Uniswap", "Aave", "Compound"],
                is_verified=True,
                risk_level="low",
                tags=["wrapped", "ethereum", "defi"],
                first_seen_at=now
            )
            
            session.add(token)
            await session.commit()
            await session.refresh(token)
            
            print(f"✅ TokenMetadata created: {token}")
            print(f"   Asset ID: {token.asset_id}")
            print(f"   Symbol: {token.symbol}")
            print(f"   Price: ${token.price_usd}")
            print(f"   Verified: {token.is_verified}")
        
        # Test querying
        print("🔍 Testing database queries...")
        async with AsyncSessionLocal() as session:
            # Query ExecutedPaths
            from sqlalchemy import select
            
            result = await session.execute(
                select(ExecutedPath).where(ExecutedPath.chain_name == "ethereum")
            )
            paths = result.scalars().all()
            print(f"✅ Found {len(paths)} executed paths on Ethereum")
            
            # Query TokenMetadata
            result = await session.execute(
                select(TokenMetadata).where(TokenMetadata.is_verified == True)
            )
            tokens = result.scalars().all()
            print(f"✅ Found {len(tokens)} verified tokens")
        
        print("🎉 Database schema test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        await close_db()


async def main():
    """Main test function."""
    print("🚀 Starting database schema validation...")
    
    # Check if we can connect to PostgreSQL
    try:
        # Try to connect to database
        from sqlalchemy import text
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL connection successful")
    except Exception as e:
        print(f"❌ Cannot connect to PostgreSQL: {e}")
        print("💡 Make sure PostgreSQL is running and DATABASE_URL is set correctly")
        return False
    
    # Run schema tests
    success = await test_database_schema()
    
    if success:
        print("✨ All database tests passed!")
        return True
    else:
        print("💥 Some database tests failed!")
        return False


if __name__ == "__main__":
    asyncio.run(main())