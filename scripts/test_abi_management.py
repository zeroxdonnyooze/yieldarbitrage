#!/usr/bin/env python3
"""Test script to validate ABI management functionality."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yield_arbitrage.protocols.abi_manager import abi_manager
from yield_arbitrage.protocols.contracts import (
    get_uniswap_v3_contract,
    get_well_known_tokens,
    is_supported_chain,
    get_supported_chains,
    UNISWAP_V3_FEE_TIERS
)


def test_abi_availability():
    """Test that all required ABIs are available."""
    print("🔧 Testing ABI availability...")
    
    test_results = []
    
    # Test Uniswap V3 ABIs
    tests = [
        ("Uniswap V3 Factory ABI", lambda: abi_manager.get_uniswap_v3_factory_abi()),
        ("Uniswap V3 Quoter ABI", lambda: abi_manager.get_uniswap_v3_quoter_abi()),
        ("Uniswap V3 Pool ABI", lambda: abi_manager.get_uniswap_v3_pool_abi()),
        ("ERC20 ABI", lambda: abi_manager.get_erc20_abi()),
    ]
    
    for test_name, test_func in tests:
        try:
            abi = test_func()
            if abi and isinstance(abi, list) and len(abi) > 0:
                print(f"✅ {test_name}: {len(abi)} items")
                test_results.append(True)
            else:
                print(f"❌ {test_name}: Invalid or empty ABI")
                test_results.append(False)
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
            test_results.append(False)
    
    return all(test_results)


def test_abi_structure():
    """Test ABI structure and required functions."""
    print("\n🧪 Testing ABI structure...")
    
    test_results = []
    
    # Test Factory ABI functions
    factory_abi = abi_manager.get_uniswap_v3_factory_abi()
    factory_functions = [item["name"] for item in factory_abi if item.get("type") == "function"]
    factory_events = [item["name"] for item in factory_abi if item.get("type") == "event"]
    
    factory_test = "getPool" in factory_functions and "PoolCreated" in factory_events
    print(f"{'✅' if factory_test else '❌'} Factory ABI: getPool function and PoolCreated event")
    test_results.append(factory_test)
    
    # Test Quoter ABI functions
    quoter_abi = abi_manager.get_uniswap_v3_quoter_abi()
    quoter_functions = [item["name"] for item in quoter_abi if item.get("type") == "function"]
    
    quoter_test = "quoteExactInputSingle" in quoter_functions and "quoteExactOutputSingle" in quoter_functions
    print(f"{'✅' if quoter_test else '❌'} Quoter ABI: Quote functions available")
    test_results.append(quoter_test)
    
    # Test Pool ABI functions
    pool_abi = abi_manager.get_uniswap_v3_pool_abi()
    pool_functions = [item["name"] for item in pool_abi if item.get("type") == "function"]
    pool_events = [item["name"] for item in pool_abi if item.get("type") == "event"]
    
    required_pool_functions = ["token0", "token1", "fee", "slot0", "liquidity"]
    required_pool_events = ["Swap", "Mint"]
    
    pool_test = (all(func in pool_functions for func in required_pool_functions) and
                 all(event in pool_events for event in required_pool_events))
    print(f"{'✅' if pool_test else '❌'} Pool ABI: Required functions and events")
    test_results.append(pool_test)
    
    # Test ERC20 ABI
    erc20_abi = abi_manager.get_erc20_abi()
    erc20_functions = [item["name"] for item in erc20_abi if item.get("type") == "function"]
    
    required_erc20_functions = ["name", "symbol", "decimals", "totalSupply", "balanceOf"]
    erc20_test = all(func in erc20_functions for func in required_erc20_functions)
    print(f"{'✅' if erc20_test else '❌'} ERC20 ABI: Standard functions available")
    test_results.append(erc20_test)
    
    return all(test_results)


def test_contract_addresses():
    """Test contract address management."""
    print("\n📍 Testing contract addresses...")
    
    test_results = []
    
    # Test supported chains
    supported_chains = get_supported_chains()
    chains_test = len(supported_chains) >= 3 and "ethereum" in supported_chains
    print(f"{'✅' if chains_test else '❌'} Supported chains: {', '.join(supported_chains)}")
    test_results.append(chains_test)
    
    # Test contract addresses for each chain
    for chain in supported_chains[:3]:  # Test first 3 chains
        factory_addr = get_uniswap_v3_contract(chain, "factory")
        quoter_addr = get_uniswap_v3_contract(chain, "quoter")
        
        addr_test = (factory_addr and quoter_addr and 
                    factory_addr.startswith("0x") and quoter_addr.startswith("0x"))
        print(f"{'✅' if addr_test else '❌'} {chain.title()}: Factory & Quoter addresses")
        test_results.append(addr_test)
    
    # Test well-known tokens
    eth_tokens = get_well_known_tokens("ethereum")
    tokens_test = len(eth_tokens) >= 4 and all(addr.startswith("0x") for addr in eth_tokens.values())
    print(f"{'✅' if tokens_test else '❌'} Ethereum tokens: {list(eth_tokens.keys())}")
    test_results.append(tokens_test)
    
    return all(test_results)


def test_fee_tiers():
    """Test Uniswap V3 fee tiers."""
    print("\n💰 Testing fee tiers...")
    
    expected_fees = [100, 500, 3000, 10000]
    fee_test = UNISWAP_V3_FEE_TIERS == expected_fees
    
    print(f"{'✅' if fee_test else '❌'} Fee tiers: {UNISWAP_V3_FEE_TIERS}")
    
    # Test fee calculations
    for fee in UNISWAP_V3_FEE_TIERS:
        percentage = fee / 1000000  # Fee is in parts per million
        print(f"   {fee} = {percentage:.4%}")
    
    return fee_test


def test_abi_validation():
    """Test ABI validation functionality."""
    print("\n🔍 Testing ABI validation...")
    
    test_results = []
    
    # Test valid ABI
    valid_abi = [
        {
            "type": "function",
            "name": "testFunction",
            "inputs": [],
            "outputs": []
        }
    ]
    
    valid_test = abi_manager.validate_abi(valid_abi)
    print(f"{'✅' if valid_test else '❌'} Valid ABI validation")
    test_results.append(valid_test)
    
    # Test invalid ABI
    invalid_abi = [{"name": "missing_type"}]
    invalid_test = not abi_manager.validate_abi(invalid_abi)
    print(f"{'✅' if invalid_test else '❌'} Invalid ABI rejection")
    test_results.append(invalid_test)
    
    # Test real ABIs
    for protocol in ["uniswap_v3", "erc20"]:
        if protocol == "uniswap_v3":
            for contract_type in ["factory", "quoter", "pool"]:
                abi = abi_manager.get_abi(protocol, contract_type)
                real_test = abi_manager.validate_abi(abi)
                print(f"{'✅' if real_test else '❌'} {protocol}.{contract_type} ABI validation")
                test_results.append(real_test)
        else:
            abi = abi_manager.get_abi(protocol)
            real_test = abi_manager.validate_abi(abi)
            print(f"{'✅' if real_test else '❌'} {protocol} ABI validation")
            test_results.append(real_test)
    
    return all(test_results)


def main():
    """Main test function."""
    print("🚀 Starting ABI Management Tests...")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("ABI Availability", test_abi_availability),
        ("ABI Structure", test_abi_structure),
        ("Contract Addresses", test_contract_addresses),
        ("Fee Tiers", test_fee_tiers),
        ("ABI Validation", test_abi_validation),
    ]
    
    for test_name, test_func in tests:
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
        print("🎉 All tests passed! ABI management is working correctly.")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)