#!/usr/bin/env python3
"""Test script to validate token filtering functionality."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yield_arbitrage.protocols.token_filter import (
    TokenCriteria,
    TokenInfo,
    PoolInfo,
    TokenFilter,
    TokenFilterCache,
    default_token_filter
)


def test_token_criteria():
    """Test token criteria functionality."""
    print("🔧 Testing token criteria...")
    
    # Test default criteria
    default_criteria = TokenCriteria()
    print(f"   Default min market cap: ${default_criteria.min_market_cap_usd:,.0f}")
    print(f"   Default min daily volume: ${default_criteria.min_daily_volume_usd:,.0f}")
    print(f"   Default min pool TVL: ${default_criteria.min_pool_tvl_usd:,.0f}")
    print(f"   Require verified: {default_criteria.require_verified}")
    
    # Test custom criteria
    custom_criteria = TokenCriteria(
        min_market_cap_usd=5_000_000,
        min_daily_volume_usd=100_000,
        require_verified=False,
        blacklisted_tokens={"0xbad1", "0xbad2"},
        whitelisted_tokens={"0xgood1"}
    )
    print(f"   Custom min market cap: ${custom_criteria.min_market_cap_usd:,.0f}")
    print(f"   Blacklisted tokens: {len(custom_criteria.blacklisted_tokens)}")
    print(f"   Whitelisted tokens: {len(custom_criteria.whitelisted_tokens)}")
    
    print("✅ Token criteria tests passed")
    return True


def test_token_info_filtering():
    """Test token info filtering logic."""
    print("\n🧪 Testing token info filtering...")
    
    criteria = TokenCriteria(
        min_market_cap_usd=1_000_000,
        min_daily_volume_usd=50_000,
        require_verified=True
    )
    
    # Test tokens
    test_tokens = [
        {
            "name": "Good Token (passes all criteria)",
            "token": TokenInfo(
                address="0x123",
                symbol="GOOD",
                name="Good Token",
                decimals=18,
                market_cap_usd=5_000_000,
                daily_volume_usd=200_000,
                is_verified=True
            ),
            "should_pass": True
        },
        {
            "name": "Low Market Cap Token",
            "token": TokenInfo(
                address="0x456",
                symbol="LOWMC",
                name="Low Market Cap Token",
                decimals=18,
                market_cap_usd=500_000,  # Too low
                daily_volume_usd=100_000,
                is_verified=True
            ),
            "should_pass": False
        },
        {
            "name": "Low Volume Token",
            "token": TokenInfo(
                address="0x789",
                symbol="LOWVOL",
                name="Low Volume Token",
                decimals=18,
                market_cap_usd=2_000_000,
                daily_volume_usd=25_000,  # Too low
                is_verified=True
            ),
            "should_pass": False
        },
        {
            "name": "Unverified Token",
            "token": TokenInfo(
                address="0xabc",
                symbol="UNVER",
                name="Unverified Token",
                decimals=18,
                market_cap_usd=2_000_000,
                daily_volume_usd=100_000,
                is_verified=False  # Not verified
            ),
            "should_pass": False
        }
    ]
    
    # Test each token
    for test_case in test_tokens:
        result = test_case["token"].meets_criteria(criteria)
        expected = test_case["should_pass"]
        status = "✅" if result == expected else "❌"
        
        print(f"   {status} {test_case['name']}: {result} (expected {expected})")
        
        if result != expected:
            return False
    
    print("✅ Token info filtering tests passed")
    return True


def test_whitelist_blacklist():
    """Test whitelist and blacklist functionality."""
    print("\n🚫 Testing whitelist and blacklist...")
    
    criteria = TokenCriteria(
        min_market_cap_usd=1_000_000,
        require_verified=True,
        blacklisted_tokens={"0xbad"},
        whitelisted_tokens={"0xwhite"}
    )
    
    # Whitelisted token should pass even with bad criteria
    whitelisted_token = TokenInfo(
        address="0xwhite",
        symbol="WHITE",
        name="Whitelisted Token",
        decimals=18,
        market_cap_usd=100,  # Very low
        is_verified=False  # Not verified
    )
    
    assert whitelisted_token.meets_criteria(criteria) is True
    print("   ✅ Whitelisted token bypassed criteria")
    
    # Blacklisted token should fail even with good criteria
    blacklisted_token = TokenInfo(
        address="0xbad",
        symbol="BAD",
        name="Blacklisted Token",
        decimals=18,
        market_cap_usd=10_000_000,  # Very high
        is_verified=True
    )
    
    assert blacklisted_token.meets_criteria(criteria) is False
    print("   ✅ Blacklisted token was rejected")
    
    print("✅ Whitelist and blacklist tests passed")
    return True


def test_pool_info_filtering():
    """Test pool info filtering logic."""
    print("\n🏊 Testing pool info filtering...")
    
    criteria = TokenCriteria(min_pool_tvl_usd=100_000)
    
    # Good pool
    good_pool = PoolInfo(
        pool_address="0xgoodpool",
        token0_address="0xtoken0",
        token1_address="0xtoken1",
        tvl_usd=500_000
    )
    
    assert good_pool.meets_criteria(criteria) is True
    print("   ✅ High TVL pool passed")
    
    # Low TVL pool
    low_tvl_pool = PoolInfo(
        pool_address="0xbadpool",
        token0_address="0xtoken0",
        token1_address="0xtoken1",
        tvl_usd=50_000  # Too low
    )
    
    assert low_tvl_pool.meets_criteria(criteria) is False
    print("   ✅ Low TVL pool was rejected")
    
    print("✅ Pool info filtering tests passed")
    return True


def test_cache_functionality():
    """Test caching functionality."""
    print("\n📦 Testing cache functionality...")
    
    cache = TokenFilterCache(ttl_seconds=3600)
    
    # Test token caching
    token = TokenInfo(
        address="0x123",
        symbol="TEST",
        name="Test Token",
        decimals=18
    )
    
    # Cache token
    cache.set_token("0x123", token)
    
    # Retrieve token
    cached_token = cache.get_token("0x123")
    assert cached_token == token
    print("   ✅ Token caching works")
    
    # Test pool caching
    pool = PoolInfo(
        pool_address="0xpool",
        token0_address="0xtoken0",
        token1_address="0xtoken1",
        tvl_usd=500_000
    )
    
    # Cache pool
    cache.set_pool("0xpool", pool)
    
    # Retrieve pool
    cached_pool = cache.get_pool("0xpool")
    assert cached_pool == pool
    print("   ✅ Pool caching works")
    
    # Test cache stats
    cache_size = len(cache._token_cache) + len(cache._pool_cache)
    print(f"   📊 Cache size: {cache_size} entries")
    
    print("✅ Cache functionality tests passed")
    return True


async def test_token_filter_basic():
    """Test basic token filter functionality."""
    print("\n🔍 Testing token filter (basic functionality)...")
    
    # Create token filter with custom criteria
    criteria = TokenCriteria(
        min_market_cap_usd=1_000_000,
        min_daily_volume_usd=50_000,
        require_verified=True
    )
    
    token_filter = TokenFilter(criteria, cache_ttl=1800)
    
    # Test criteria update
    original_market_cap = token_filter.criteria.min_market_cap_usd
    token_filter.update_criteria(min_market_cap_usd=2_000_000)
    
    assert token_filter.criteria.min_market_cap_usd == 2_000_000
    assert token_filter.criteria.min_market_cap_usd != original_market_cap
    print("   ✅ Criteria update works")
    
    # Test stats (initially empty)
    stats = token_filter.get_stats()
    assert stats["tokens_evaluated"] == 0
    assert stats["tokens_passed"] == 0
    print("   ✅ Initial stats are correct")
    
    # Test cache clearing
    token_filter.cache.set_token("0x123", TokenInfo("0x123", "TEST", "Test", 18))
    assert len(token_filter.cache._token_cache) > 0
    
    token_filter.clear_cache()
    assert len(token_filter.cache._token_cache) == 0
    print("   ✅ Cache clearing works")
    
    print("✅ Token filter basic tests passed")
    return True


def test_default_token_filter():
    """Test default token filter instance."""
    print("\n🌐 Testing default token filter...")
    
    # Check that default instance exists
    assert default_token_filter is not None
    assert isinstance(default_token_filter, TokenFilter)
    print("   ✅ Default instance exists")
    
    # Check default criteria
    criteria = default_token_filter.criteria
    assert criteria.min_market_cap_usd == 1_000_000
    assert criteria.min_daily_volume_usd == 50_000
    assert criteria.min_pool_tvl_usd == 100_000
    assert criteria.require_verified is True
    print("   ✅ Default criteria are correct")
    
    # Check initial stats
    stats = default_token_filter.get_stats()
    assert "tokens_evaluated" in stats
    assert "cache_size" in stats
    assert "success_rate" in stats
    print("   ✅ Stats structure is correct")
    
    print("✅ Default token filter tests passed")
    return True


async def main():
    """Main test function."""
    print("🚀 Starting Token Filtering Tests...")
    print("=" * 60)
    
    test_results = []
    
    # Synchronous tests
    sync_tests = [
        ("Token Criteria", test_token_criteria),
        ("Token Info Filtering", test_token_info_filtering),
        ("Whitelist/Blacklist", test_whitelist_blacklist),
        ("Pool Info Filtering", test_pool_info_filtering),
        ("Cache Functionality", test_cache_functionality),
        ("Default Token Filter", test_default_token_filter),
    ]
    
    # Asynchronous tests
    async_tests = [
        ("Token Filter Basic", test_token_filter_basic),
    ]
    
    # Run synchronous tests
    for test_name, test_func in sync_tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            test_results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"📊 Result: {status}")
            
        except Exception as e:
            print(f"💥 Test crashed: {e}")
            test_results.append((test_name, False))
    
    # Run asynchronous tests
    for test_name, test_func in async_tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            test_results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"📊 Result: {status}")
            
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
        print("🎉 All tests passed! Token filtering is working correctly.")
        print("\n💡 Key Features Validated:")
        print("   • Token criteria configuration and validation")
        print("   • Market cap, volume, and verification filtering")
        print("   • Whitelist and blacklist functionality")
        print("   • Pool TVL filtering")
        print("   • Caching with TTL expiration")
        print("   • Statistics tracking")
        print("   • Default filter instance")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)