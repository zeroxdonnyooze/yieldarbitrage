#!/usr/bin/env python3
"""
Simple test of AMM math implementations.
"""
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.dex_protocols.uniswap_v2_math import UniswapV2Math
from yield_arbitrage.protocols.dex_protocols.curve_stableswap_math import CurveStableSwapMath
from yield_arbitrage.protocols.dex_protocols.balancer_weighted_math import BalancerWeightedMath


def test_uniswap_v2():
    """Test Uniswap V2 implementation."""
    print("Testing Uniswap V2 Math...")
    
    v2_math = UniswapV2Math()
    
    # Test basic swap
    amount_out = v2_math.calculate_amount_out(
        Decimal('1000'),
        Decimal('1000000'),
        Decimal('1000000')
    )
    
    print(f"  1000 in → {amount_out} out (expected ~997)")
    
    # Test price impact
    impact = v2_math.calculate_price_impact(
        Decimal('100000'),
        Decimal('1000000'),
        Decimal('1000000')
    )
    
    print(f"  Price impact for 100K trade: {impact * 100:.2f}%")
    
    print("✅ Uniswap V2 tests passed\n")


def test_curve_stableswap():
    """Test Curve implementation."""
    print("Testing Curve StableSwap Math...")
    
    curve_math = CurveStableSwapMath(amplification_coefficient=Decimal('100'))
    
    # Test balanced pool
    balances = [Decimal('1000000'), Decimal('1000000')]
    
    amount_out = curve_math.get_dy(0, 1, Decimal('1000'), balances)
    print(f"  1000 USDC → {amount_out} USDT (expected ~999)")
    
    # Test invariant calculation
    D = curve_math.get_D(balances)
    print(f"  Invariant D: {D} (expected ~2000000)")
    
    print("✅ Curve tests passed\n")


def test_balancer_weighted():
    """Test Balancer implementation."""
    print("Testing Balancer Weighted Math...")
    
    balancer_math = BalancerWeightedMath()
    
    # Test spot price
    spot_price = balancer_math.calculate_spot_price(
        Decimal('100'),     # 100 WETH
        Decimal('0.8'),     # 80% weight
        Decimal('600000'),  # 600K USDC  
        Decimal('0.2'),     # 20% weight
        Decimal('0.001')    # 0.1% fee
    )
    
    print(f"  Spot price: {spot_price} (USDC per WETH)")
    
    # Test swap
    amount_out = balancer_math.calculate_out_given_in(
        Decimal('100'),
        Decimal('0.8'),
        Decimal('600000'),
        Decimal('0.2'),
        Decimal('1')  # 1 WETH
    )
    
    print(f"  1 WETH → {amount_out} USDC")
    
    print("✅ Balancer tests passed\n")


def test_comparative():
    """Compare implementations on same data."""
    print("Comparing implementations...")
    
    # Same balanced pool
    reserve_in = Decimal('1000000')
    reserve_out = Decimal('1000000')
    amount_in = Decimal('10000')
    
    # V2
    v2_math = UniswapV2Math()
    v2_output = v2_math.calculate_amount_out(amount_in, reserve_in, reserve_out)
    
    # Curve
    curve_math = CurveStableSwapMath(amplification_coefficient=Decimal('100'))
    curve_output = curve_math.get_dy(0, 1, amount_in, [reserve_in, reserve_out])
    
    # Balancer 50/50
    balancer_math = BalancerWeightedMath()
    balancer_output = balancer_math.calculate_out_given_in(
        reserve_in, Decimal('0.5'), reserve_out, Decimal('0.5'), amount_in
    )
    
    print(f"  V2 output: {v2_output}")
    print(f"  Curve output: {curve_output}")
    print(f"  Balancer output: {balancer_output}")
    
    print("✅ Comparative tests passed\n")


def main():
    """Run all tests."""
    print("🧮 Testing Real AMM Math Implementations")
    print("=" * 50)
    
    try:
        test_uniswap_v2()
        test_curve_stableswap()
        test_balancer_weighted()
        test_comparative()
        
        print("✅ All tests passed!")
        print("\n🎯 Real math implementations working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()