#!/usr/bin/env python3
"""Test script to validate enhanced database connection and session management."""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yield_arbitrage.database import (
    init_database, health_check, get_pool_status, get_session, transaction,
    execute_with_retry, startup_database, shutdown_database,
    ExecutedPath, TokenMetadata
)
from sqlalchemy import text, select


async def test_database_initialization():
    """Test database initialization and health check."""
    print("🔧 Testing database initialization...")
    
    try:
        await init_database()
        print("✅ Database initialized successfully")
        
        # Test health check
        healthy = await health_check()
        print(f"🏥 Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
        
        # Test pool status
        pool_status = await get_pool_status()
        print(f"🏊 Pool status: {pool_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


async def test_session_management():
    """Test session management with various scenarios."""
    print("\n🧪 Testing session management...")
    
    try:
        # Test basic session usage
        print("📝 Testing basic session...")
        async with get_session() as session:
            result = await session.execute(text("SELECT 1 as test_value"))
            value = result.scalar()
            assert value == 1
        print("✅ Basic session test passed")
        
        # Test transaction context manager
        print("📝 Testing transaction context manager...")
        async with transaction() as session:
            result = await session.execute(text("SELECT 'transaction_test' as test_value"))
            value = result.scalar()
            assert value == "transaction_test"
        print("✅ Transaction context manager test passed")
        
        # Test execute with retry
        print("📝 Testing execute with retry...")
        result = await execute_with_retry(text("SELECT 'retry_test' as test_value"))
        value = result.scalar()
        assert value == "retry_test"
        print("✅ Execute with retry test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        return False


async def test_orm_operations():
    """Test ORM operations with the enhanced connection."""
    print("\n💾 Testing ORM operations...")
    
    try:
        now = datetime.now(timezone.utc)
        
        # Test ExecutedPath creation
        print("📝 Testing ExecutedPath creation...")
        async with get_session() as session:
            path = ExecutedPath(
                path_hash="test_enhanced_connection_hash_123456789012345678901234567890",
                transaction_hash="0xenhanced123456789012345678901234567890abcdef1234567890abcdef",
                block_number=19500000,
                chain_name="ethereum",
                edge_ids=["ETH_ENHANCED_TEST"],
                edge_types=["TRADE"],
                protocols=["TestProtocol"],
                input_amount=Decimal("1000000000000000000"),
                output_amount=Decimal("2000000000"),
                input_token="0xA0b86a33E6441b9435B654f6D26Cc98b6e1D0a3A",
                output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                profit_usd=Decimal("100.00"),
                profit_percentage=Decimal("0.05"),
                effective_rate=Decimal("2000.0"),
                gas_used=300000,
                gas_price_gwei=Decimal("30.0"),
                gas_cost_usd=Decimal("15.00"),
                execution_time_ms=2000,
                slippage_actual=Decimal("0.01"),
                mev_protected=True,
                flash_loan_used=True,
                ml_confidence_score=Decimal("0.9500"),
                discovered_at=now,
                executed_at=now
            )
            
            session.add(path)
            await session.commit()
            await session.refresh(path)
            
            print(f"✅ ExecutedPath created: {path}")
            path_id = path.id
        
        # Test TokenMetadata creation
        print("📝 Testing TokenMetadata creation...")
        async with get_session() as session:
            token = TokenMetadata(
                asset_id="ETH_MAINNET_ENHANCED_TEST_TOKEN",
                chain_name="ethereum",
                contract_address="0x1111111111111111111111111111111111111111",
                symbol="ENHANCED",
                name="Enhanced Test Token",
                decimals=18,
                is_stable=False,
                is_wrapped=False,
                is_yield_bearing=True,
                price_usd=Decimal("50.00"),
                price_eth=Decimal("0.02"),
                market_cap_usd=Decimal("1000000000.00"),
                volume_24h_usd=Decimal("50000000.00"),
                liquidity_usd=Decimal("10000000.00"),
                security_score=Decimal("0.8500"),
                liquidity_score=Decimal("0.9000"),
                reliability_score=Decimal("0.8800"),
                base_apr=Decimal("0.0650"),
                reward_tokens=["ETH", "TOKEN"],
                protocols=["Aave", "Compound", "Uniswap"],
                is_verified=True,
                risk_level="low",
                tags=["test", "enhanced", "yield"],
                first_seen_at=now
            )
            
            session.add(token)
            await session.commit()
            await session.refresh(token)
            
            print(f"✅ TokenMetadata created: {token}")
            token_id = token.id
        
        # Test querying
        print("📝 Testing queries...")
        async with get_session() as session:
            # Query ExecutedPath
            result = await session.execute(
                select(ExecutedPath).where(ExecutedPath.id == path_id)
            )
            found_path = result.scalar_one_or_none()
            assert found_path is not None
            assert found_path.profit_usd == Decimal("100.00")
            
            # Query TokenMetadata
            result = await session.execute(
                select(TokenMetadata).where(TokenMetadata.id == token_id)
            )
            found_token = result.scalar_one_or_none()
            assert found_token is not None
            assert found_token.symbol == "ENHANCED"
            
        print("✅ All ORM operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ ORM operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_concurrent_connections():
    """Test concurrent database connections and pool utilization."""
    print("\n🚀 Testing concurrent connections...")
    
    async def worker_task(worker_id: int, duration: float = 1.0):
        """Worker task that holds a database connection."""
        try:
            async with get_session() as session:
                start_time = time.time()
                
                # Simulate some work
                await asyncio.sleep(duration)
                
                result = await session.execute(
                    text("SELECT :worker_id as worker_id, :duration as duration"),
                    {"worker_id": worker_id, "duration": duration}
                )
                data = result.fetchone()
                
                elapsed = time.time() - start_time
                print(f"  Worker {worker_id}: Completed in {elapsed:.2f}s")
                return data
                
        except Exception as e:
            print(f"  Worker {worker_id}: Failed - {e}")
            return None
    
    try:
        # Test pool status before concurrent operations
        pool_status_before = await get_pool_status()
        print(f"📊 Pool status before: {pool_status_before}")
        
        # Run multiple concurrent workers
        print("🏃 Running 10 concurrent workers...")
        tasks = [worker_task(i, 0.5) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_tasks = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        print(f"✅ {successful_tasks}/10 workers completed successfully")
        
        # Test pool status after concurrent operations
        pool_status_after = await get_pool_status()
        print(f"📊 Pool status after: {pool_status_after}")
        
        return successful_tasks >= 8  # Allow for some potential failures
        
    except Exception as e:
        print(f"❌ Concurrent connections test failed: {e}")
        return False


async def test_startup_shutdown_lifecycle():
    """Test full startup and shutdown lifecycle."""
    print("\n🔄 Testing startup/shutdown lifecycle...")
    
    try:
        # Test startup
        print("🚀 Testing startup_database...")
        await startup_database()
        print("✅ Startup completed successfully")
        
        # Verify functionality after startup
        healthy = await health_check()
        assert healthy, "Health check failed after startup"
        
        pool_status = await get_pool_status()
        assert pool_status.get("pool_size", 0) > 0, "Pool not properly initialized"
        
        # Test shutdown
        print("🛑 Testing shutdown_database...")
        await shutdown_database()
        print("✅ Shutdown completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Startup/shutdown lifecycle test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting enhanced database connection tests...")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Database Initialization", test_database_initialization),
        ("Session Management", test_session_management),
        ("ORM Operations", test_orm_operations),
        ("Concurrent Connections", test_concurrent_connections),
        ("Startup/Shutdown Lifecycle", test_startup_shutdown_lifecycle),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running test: {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
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
        print("🎉 All tests passed! Enhanced database connection is working correctly.")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())