"""
Unit tests for Balancer Weighted Pool Math implementation.

Tests the weighted pool invariant with power calculations.
"""
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.dex_protocols.balancer_weighted_math import BalancerWeightedMath


def test_spot_price_calculation():
    """Test spot price calculation for weighted pools."""
    balancer_math = BalancerWeightedMath()
    
    # 80/20 WETH/USDC pool
    # 100 WETH ($300k), 600k USDC
    spot_price = balancer_math.calculate_spot_price(
        Decimal('100'),    # WETH balance
        Decimal('0.8'),    # WETH weight
        Decimal('600000'), # USDC balance  
        Decimal('0.2'),    # USDC weight
        Decimal('0.001')   # 0.1% fee
    )
    
    # Price should reflect weights: (100/0.8) / (600k/0.2) = 125 / 3M = ~0.00004167
    expected = Decimal('0.0000416')
    assert abs(spot_price - expected) < expected * Decimal('0.1')  # Within 10%
    print(f"✅ Spot price: {spot_price} (expected ~{expected})")


def test_weighted_swap_80_20():
    """Test swap in 80/20 weighted pool."""
    balancer_math = BalancerWeightedMath()
    
    amount_out = balancer_math.calculate_out_given_in(
        Decimal('100'),    # WETH balance
        Decimal('0.8'),    # WETH weight
        Decimal('600000'), # USDC balance
        Decimal('0.2'),    # USDC weight
        Decimal('1')       # 1 WETH in
    )
    
    # Should get substantial USDC (pool is 80% WETH weighted)
    assert amount_out > Decimal('1000')   # >$1000
    assert amount_out < Decimal('30000')  # <$30000
    print(f"✅ 80/20 pool: 1 WETH → {amount_out} USDC")


def test_weighted_swap_50_50():
    """Test swap in 50/50 balanced pool (should behave like Uniswap V2)."""
    balancer_math = BalancerWeightedMath()
    
    # 50/50 USDC/USDT pool (1M each)
    amount_out = balancer_math.calculate_out_given_in(
        Decimal('1000000'), # USDC balance
        Decimal('0.5'),     # USDC weight
        Decimal('1000000'), # USDT balance
        Decimal('0.5'),     # USDT weight
        Decimal('1000')     # 1000 USDC in
    )
    
    # Should get close to 1000 USDT (minus fees)
    assert amount_out > Decimal('995')   # More than 99.5%
    assert amount_out < Decimal('1000')  # Less than 100%
    print(f"✅ 50/50 pool: 1000 USDC → {amount_out} USDT")


def test_invariant_preservation():
    """Test that weighted invariant V = ∏(B_i^W_i) is preserved."""
    balancer_math = BalancerWeightedMath()
    
    # Initial balances and weights
    balance_in = Decimal('1000000')
    weight_in = Decimal('0.6')
    balance_out = Decimal('500000')
    weight_out = Decimal('0.4')
    
    # Calculate initial invariant
    initial_invariant = (balance_in ** weight_in) * (balance_out ** weight_out)
    
    # Perform swap
    amount_in = Decimal('10000')
    amount_out = balancer_math.calculate_out_given_in(
        balance_in, weight_in, balance_out, weight_out, amount_in
    )
    
    # Calculate new balances
    new_balance_in = balance_in + amount_in
    new_balance_out = balance_out - amount_out
    
    # Calculate new invariant (should be preserved minus fees)
    new_invariant = (new_balance_in ** weight_in) * (new_balance_out ** weight_out)
    
    # Invariant should be approximately preserved (fees cause slight increase)
    invariant_change = (new_invariant - initial_invariant) / initial_invariant
    assert abs(invariant_change) < Decimal('0.01')  # Within 1%
    
    print(f"✅ Invariant change: {invariant_change*100:.4f}%")


def test_reverse_calculation():
    """Test calculating required input for desired output."""
    balancer_math = BalancerWeightedMath()
    
    balance_in = Decimal('1000000')
    weight_in = Decimal('0.5')
    balance_out = Decimal('500000')
    weight_out = Decimal('0.5')
    desired_out = Decimal('1000')
    
    required_in = balancer_math.calculate_in_given_out(
        balance_in, weight_in, balance_out, weight_out, desired_out
    )
    
    # Verify by calculating forward
    actual_out = balancer_math.calculate_out_given_in(
        balance_in, weight_in, balance_out, weight_out, required_in
    )
    
    # Should get at least the desired amount (within rounding tolerance)
    assert actual_out >= desired_out - Decimal('0.01')  # Allow small rounding error
    assert abs(actual_out - desired_out) < Decimal('2')  # Within $2
    
    print(f"✅ Reverse calc: Need {required_in} in to get {actual_out} out (wanted {desired_out})")


def test_different_weight_ratios():
    """Test various weight ratios."""
    balancer_math = BalancerWeightedMath()
    
    # Same balances, different weights
    balance_in = Decimal('1000000')
    balance_out = Decimal('1000000')
    amount_in = Decimal('10000')
    
    weight_ratios = [
        (Decimal('0.5'), Decimal('0.5')),   # 50/50
        (Decimal('0.8'), Decimal('0.2')),   # 80/20  
        (Decimal('0.9'), Decimal('0.1')),   # 90/10
    ]
    
    amounts_out = []
    for weight_in, weight_out in weight_ratios:
        amount_out = balancer_math.calculate_out_given_in(
            balance_in, weight_in, balance_out, weight_out, amount_in
        )
        amounts_out.append(amount_out)
        ratio_str = f"{int(weight_in*100)}/{int(weight_out*100)}"
        print(f"    {ratio_str}: {amount_out:.2f} out")
    
    # Higher weight on input token should give more output
    assert amounts_out[1] > amounts_out[0]  # 80/20 > 50/50
    assert amounts_out[2] > amounts_out[1]  # 90/10 > 80/20
    
    print("✅ Higher input weight gives more output")


def test_price_impact_calculation():
    """Test price impact for different trade sizes."""
    balancer_math = BalancerWeightedMath()
    
    balance_in = Decimal('1000000')
    weight_in = Decimal('0.8')
    balance_out = Decimal('200000')
    weight_out = Decimal('0.2')
    
    # Test different trade sizes
    trade_sizes = [1000, 10000, 100000]
    
    for amount in trade_sizes:
        amount_decimal = Decimal(str(amount))
        
        # Calculate price impact
        spot_price = balancer_math.calculate_spot_price(
            balance_in, weight_in, balance_out, weight_out, Decimal('0')
        )
        
        amount_out = balancer_math.calculate_out_given_in(
            balance_in, weight_in, balance_out, weight_out, amount_decimal
        )
        
        execution_price = amount_out / amount_decimal
        price_impact = (spot_price - execution_price) / spot_price
        
        print(f"    ${amount} trade: {price_impact*100:.3f}% price impact")
        
        # Larger trades should have more impact
        if amount >= 50000:
            assert price_impact > Decimal('0.01')  # >1% for large trades


def test_multi_asset_pool():
    """Test 3-asset pool calculations."""
    balancer_math = BalancerWeightedMath()
    
    # 50/30/20 WETH/WBTC/USDC pool
    balances = [Decimal('1000'), Decimal('50'), Decimal('3000000')]  # WETH, WBTC, USDC
    weights = [Decimal('0.5'), Decimal('0.3'), Decimal('0.2')]
    
    # Calculate total invariant
    total_invariant = balancer_math.calculate_invariant(balances, weights)
    
    # Should be positive and finite
    assert total_invariant > 0
    assert total_invariant.is_finite()
    
    print(f"✅ 3-asset pool invariant: {total_invariant:.2e}")


def test_LP_token_calculations():
    """Test LP token minting and burning."""
    balancer_math = BalancerWeightedMath()
    
    # Current pool state
    balances = [Decimal('1000000'), Decimal('500000')]
    weights = [Decimal('0.8'), Decimal('0.2')]
    total_supply = Decimal('1000000')  # Current LP tokens
    
    # Add liquidity (proportional)
    amounts_in = [Decimal('100000'), Decimal('50000')]  # 10% increase
    
    lp_tokens_out = balancer_math.calculate_lp_tokens_out(
        balances, weights, amounts_in, total_supply
    )
    
    # Should get approximately 10% more LP tokens
    expected_lp = total_supply * Decimal('0.1')
    assert abs(lp_tokens_out - expected_lp) < expected_lp * Decimal('0.05')  # Within 5%
    
    print(f"✅ LP tokens: {amounts_in} → {lp_tokens_out:.0f} LP tokens")


def test_edge_cases():
    """Test edge cases and error handling."""
    balancer_math = BalancerWeightedMath()
    
    # Zero amounts
    assert balancer_math.calculate_out_given_in(
        Decimal('1000'), Decimal('0.5'), Decimal('1000'), Decimal('0.5'), Decimal('0')
    ) == Decimal('0')
    
    # Negative amounts
    assert balancer_math.calculate_out_given_in(
        Decimal('1000'), Decimal('0.5'), Decimal('1000'), Decimal('0.5'), Decimal('-100')
    ) == Decimal('0')
    
    # Zero balances
    assert balancer_math.calculate_out_given_in(
        Decimal('0'), Decimal('0.5'), Decimal('1000'), Decimal('0.5'), Decimal('100')
    ) == Decimal('0')
    
    # Invalid weights (don't sum to 1)
    try:
        balancer_math.calculate_out_given_in(
            Decimal('1000'), Decimal('0.7'), Decimal('1000'), Decimal('0.5'), Decimal('100')
        )
        # Should still work (weights are used as ratios)
    except:
        pass  # Some implementations might validate weight sums
    
    # Weights sum to zero
    assert balancer_math.calculate_out_given_in(
        Decimal('1000'), Decimal('0'), Decimal('1000'), Decimal('0'), Decimal('100')
    ) == Decimal('0')
    
    print("✅ Edge cases handled correctly")


def test_fee_impact():
    """Test impact of different fee levels."""
    balancer_math = BalancerWeightedMath()
    
    balance_in = Decimal('1000000')
    weight_in = Decimal('0.5')
    balance_out = Decimal('1000000')
    weight_out = Decimal('0.5')
    amount_in = Decimal('10000')
    
    # Test different fee levels
    fees = [Decimal('0'), Decimal('0.001'), Decimal('0.003'), Decimal('0.01')]
    amounts_out = []
    
    for fee in fees:
        amount_out = balancer_math.calculate_out_given_in_with_fee(
            balance_in, weight_in, balance_out, weight_out, amount_in, fee
        )
        amounts_out.append(amount_out)
        print(f"    {fee*100:.1f}% fee: {amount_out:.2f} out")
    
    # Higher fees should give less output
    for i in range(1, len(amounts_out)):
        assert amounts_out[i] < amounts_out[i-1]
    
    print("✅ Higher fees reduce output")


def run_all_tests():
    """Run all Balancer Weighted Pool tests."""
    print("⚖️  Testing Balancer Weighted Pool Math Implementation")
    print("=" * 60)
    
    try:
        test_spot_price_calculation()
        test_weighted_swap_80_20()
        test_weighted_swap_50_50()
        test_invariant_preservation()
        test_reverse_calculation()
        test_different_weight_ratios()
        test_price_impact_calculation()
        test_multi_asset_pool()
        test_LP_token_calculations()
        test_edge_cases()
        test_fee_impact()
        
        print("\n✅ All Balancer Weighted Pool tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()