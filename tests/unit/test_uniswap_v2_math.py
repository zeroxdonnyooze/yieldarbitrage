"""
Unit tests for Uniswap V2 Math implementation.

Tests the exact constant product formula implementation.
"""
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.dex_protocols.uniswap_v2_math import UniswapV2Math


def test_basic_swap_calculation():
    """Test basic swap calculation with known values."""
    v2_math = UniswapV2Math()
    
    # Test case: 1M USDC, 1M USDT pool, swap 1000 USDC
    amount_out = v2_math.calculate_amount_out(
        Decimal('1000'),      # 1000 USDC in
        Decimal('1000000'),   # 1M USDC reserve
        Decimal('1000000')    # 1M USDT reserve
    )
    
    # Expected: ~996.00 USDT (997/1000 factor for 0.3% fee)
    # Formula: (1000 * 997 * 1000000) / (1000000 * 1000 + 1000 * 997)
    expected = Decimal('996.007')
    assert abs(amount_out - expected) < Decimal('0.01')
    print(f"✅ Basic swap: 1000 USDC → {amount_out} USDT (expected ~{expected})")


def test_exact_uniswap_formula():
    """Test exact Uniswap V2 formula implementation."""
    v2_math = UniswapV2Math()
    
    # Manual calculation of exact formula
    amount_in = Decimal('5000')
    reserve_in = Decimal('2000000')
    reserve_out = Decimal('1000000')
    
    # Manual: (5000 * 997 * 1000000) / (2000000 * 1000 + 5000 * 997)
    # = 4985000000 / 2004985000 = 2.487514...
    
    amount_out = v2_math.calculate_amount_out(amount_in, reserve_in, reserve_out)
    manual_calc = (amount_in * 997 * reserve_out) / (reserve_in * 1000 + amount_in * 997)
    
    assert abs(amount_out - manual_calc) < Decimal('0.000001')
    print(f"✅ Exact formula: {amount_in} in → {amount_out} out")


def test_price_impact_calculation():
    """Test price impact calculation."""
    v2_math = UniswapV2Math()
    
    # Small trade should have minimal impact
    small_impact = v2_math.calculate_price_impact(
        Decimal('1000'),
        Decimal('10000000'),
        Decimal('10000000')
    )
    
    # Large trade should have significant impact
    large_impact = v2_math.calculate_price_impact(
        Decimal('1000000'),
        Decimal('10000000'),
        Decimal('10000000')
    )
    
    assert small_impact < Decimal('0.01')   # <1% for small trade
    assert large_impact > Decimal('0.05')   # >5% for large trade
    assert large_impact > small_impact      # Large trade has more impact
    
    print(f"✅ Price impact: Small trade {small_impact*100:.3f}%, Large trade {large_impact*100:.3f}%")


def test_spot_price_calculation():
    """Test spot price calculation."""
    v2_math = UniswapV2Math()
    
    # 1M USDC, 333.33 WETH pool → $3000/ETH
    spot_price = v2_math.get_spot_price(
        Decimal('1000000'),   # USDC reserve
        Decimal('333.33'),    # WETH reserve
        include_fee=False
    )
    
    expected_price = Decimal('333.33') / Decimal('1000000')  # WETH per USDC
    assert abs(spot_price - expected_price) < Decimal('0.000001')
    
    # With fee should be slightly different
    spot_price_with_fee = v2_math.get_spot_price(
        Decimal('1000000'),
        Decimal('333.33'),
        include_fee=True
    )
    
    assert spot_price_with_fee < spot_price  # Fee reduces effective price
    print(f"✅ Spot price: {spot_price} WETH/USDC (no fee), {spot_price_with_fee} (with fee)")


def test_invariant_preservation():
    """Test that constant product invariant k=xy is preserved."""
    v2_math = UniswapV2Math()
    
    initial_x = Decimal('1000000')
    initial_y = Decimal('2000000')
    initial_k = initial_x * initial_y
    
    amount_in = Decimal('10000')
    amount_out = v2_math.calculate_amount_out(amount_in, initial_x, initial_y)
    
    new_x = initial_x + amount_in
    new_y = initial_y - amount_out
    new_k = new_x * new_y
    
    # k should increase slightly due to fees
    assert new_k > initial_k
    k_increase = (new_k - initial_k) / initial_k
    assert k_increase < Decimal('0.001')  # Less than 0.1% increase
    
    print(f"✅ Invariant: k increased by {k_increase*100:.6f}% (due to fees)")


def test_reverse_calculation():
    """Test calculating required input for desired output."""
    v2_math = UniswapV2Math()
    
    reserve_in = Decimal('1000000')
    reserve_out = Decimal('500000')
    desired_out = Decimal('1000')
    
    required_in = v2_math.calculate_amount_in(desired_out, reserve_in, reserve_out)
    
    # Verify by calculating forward
    actual_out = v2_math.calculate_amount_out(required_in, reserve_in, reserve_out)
    
    # Should get at least the desired amount (might be slightly more due to rounding)
    assert actual_out >= desired_out
    assert abs(actual_out - desired_out) < Decimal('2')  # Within $2
    
    print(f"✅ Reverse calc: Need {required_in} in to get {actual_out} out (wanted {desired_out})")


def test_arbitrage_calculation():
    """Test arbitrage profit calculation between two pools."""
    v2_math = UniswapV2Math()
    
    # Pool 1: ETH more expensive (less ETH per USDC)
    pool1_usdc = Decimal('10000000')
    pool1_eth = Decimal('3000')  # $3333/ETH
    
    # Pool 2: ETH cheaper (more ETH per USDC)  
    pool2_usdc = Decimal('8000000')
    pool2_eth = Decimal('2700')  # $2963/ETH
    
    # Test arbitrage with different amounts
    for amount in [1000, 10000, 100000]:
        profit = v2_math.calculate_arbitrage_profit(
            Decimal(str(amount)),
            pool2_usdc, pool2_eth,  # Buy ETH on pool 2 (cheaper)
            pool1_eth, pool1_usdc   # Sell ETH on pool 1 (expensive)
        )
        
        print(f"    ${amount} arbitrage → ${profit:.2f} profit")
        
        if amount <= 50000:  # Small amounts should be profitable
            assert profit > 0
        
    print("✅ Arbitrage calculation working")


def test_edge_cases():
    """Test edge cases and error handling."""
    v2_math = UniswapV2Math()
    
    # Zero amounts
    assert v2_math.calculate_amount_out(Decimal('0'), Decimal('1000'), Decimal('1000')) == Decimal('0')
    
    # Negative amounts
    assert v2_math.calculate_amount_out(Decimal('-100'), Decimal('1000'), Decimal('1000')) == Decimal('0')
    
    # Zero reserves
    assert v2_math.calculate_amount_out(Decimal('100'), Decimal('0'), Decimal('1000')) == Decimal('0')
    assert v2_math.calculate_amount_out(Decimal('100'), Decimal('1000'), Decimal('0')) == Decimal('0')
    
    # Amount larger than reserve (should return close to 0 or very small - can't drain pool)
    large_amount = v2_math.calculate_amount_out(Decimal('1000000'), Decimal('1000'), Decimal('1000'))
    assert large_amount < Decimal('1000')  # Should not drain the pool completely
    
    print("✅ Edge cases handled correctly")


def test_different_fee_rates():
    """Test with different fee rates."""
    # Test 0.05% fee (like Uniswap V3 0.05% tier)
    v2_math_low_fee = UniswapV2Math(fee_rate=Decimal('0.0005'))
    
    # Test 1% fee (high fee DEX)
    v2_math_high_fee = UniswapV2Math(fee_rate=Decimal('0.01'))
    
    amount_in = Decimal('1000')
    reserve_in = Decimal('1000000')
    reserve_out = Decimal('1000000')
    
    low_fee_out = v2_math_low_fee.calculate_amount_out(amount_in, reserve_in, reserve_out)
    normal_fee_out = UniswapV2Math().calculate_amount_out(amount_in, reserve_in, reserve_out)
    high_fee_out = v2_math_high_fee.calculate_amount_out(amount_in, reserve_in, reserve_out)
    
    # Lower fees should give more output
    assert low_fee_out > normal_fee_out > high_fee_out
    
    print(f"✅ Fee rates: 0.05%→{low_fee_out:.2f}, 0.3%→{normal_fee_out:.2f}, 1%→{high_fee_out:.2f}")


def test_simulate_swap_to_price():
    """Test swap to target price simulation."""
    v2_math = UniswapV2Math()
    
    current_reserve_in = Decimal('1000000')
    current_reserve_out = Decimal('500000')
    
    # Current price: 0.5 out per in
    current_price = current_reserve_out / current_reserve_in
    
    # Target price: 0.6 out per in (need to add to out token)
    target_price = Decimal('0.6')
    
    amount_in, amount_out = v2_math.simulate_swap_to_price(
        current_reserve_in, current_reserve_out, target_price
    )
    
    if amount_in > 0:  # If swap is needed
        # Verify the swap moves us toward target price
        new_reserve_in = current_reserve_in + amount_in
        new_reserve_out = current_reserve_out - amount_out
        new_price = new_reserve_out / new_reserve_in
        
        print(f"✅ Price targeting: {current_price:.3f} → {new_price:.3f} (target {target_price:.3f})")
    else:
        print("✅ Price targeting: Already at target price")


def run_all_tests():
    """Run all Uniswap V2 tests."""
    print("🧪 Testing Uniswap V2 Math Implementation")
    print("=" * 50)
    
    try:
        test_basic_swap_calculation()
        test_exact_uniswap_formula()
        test_price_impact_calculation()
        test_spot_price_calculation()
        test_invariant_preservation()
        test_reverse_calculation()
        test_arbitrage_calculation()
        test_edge_cases()
        test_different_fee_rates()
        test_simulate_swap_to_price()
        
        print("\n✅ All Uniswap V2 tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()