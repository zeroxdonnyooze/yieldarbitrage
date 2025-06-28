"""
Unit tests for real AMM math implementations.

Tests that our protocol implementations produce accurate results
compared to known values and edge cases.
"""
import pytest
from decimal import Decimal

from yield_arbitrage.protocols.dex_protocols.uniswap_v2_math import UniswapV2Math
from yield_arbitrage.protocols.dex_protocols.uniswap_v3_math import UniswapV3Math
from yield_arbitrage.protocols.dex_protocols.curve_stableswap_math import CurveStableSwapMath
from yield_arbitrage.protocols.dex_protocols.balancer_weighted_math import BalancerWeightedMath


class TestUniswapV2Math:
    """Test Uniswap V2 math implementation."""
    
    def test_basic_swap_calculation(self):
        """Test basic swap calculation matches expected values."""
        v2_math = UniswapV2Math()
        
        # Known case: 10M USDC, 3333 WETH pool
        # Swap 1000 USDC for WETH
        amount_out = v2_math.calculate_amount_out(
            Decimal('1000'),      # 1000 USDC in
            Decimal('10000000'),  # 10M USDC reserve
            Decimal('3333')       # 3333 WETH reserve
        )
        
        # Should get approximately 0.3332 WETH (minus fees)
        expected = Decimal('0.3332')
        assert abs(amount_out - expected) < Decimal('0.001')
    
    def test_price_impact_calculation(self):
        """Test price impact is calculated correctly."""
        v2_math = UniswapV2Math()
        
        price_impact = v2_math.calculate_price_impact(
            Decimal('1000000'),   # 1M USDC (large trade)
            Decimal('10000000'),  # 10M USDC reserve
            Decimal('3333')       # 3333 WETH reserve
        )
        
        # Large trade should have significant impact
        assert price_impact > Decimal('0.05')  # >5% impact
        assert price_impact < Decimal('0.15')  # <15% impact
    
    def test_invariant_preserved(self):
        """Test that k=xy invariant is preserved."""
        v2_math = UniswapV2Math()
        
        initial_reserve_in = Decimal('1000000')
        initial_reserve_out = Decimal('2000000')
        initial_k = initial_reserve_in * initial_reserve_out
        
        # Small trade
        amount_in = Decimal('1000')
        amount_out = v2_math.calculate_amount_out(
            amount_in, initial_reserve_in, initial_reserve_out
        )
        
        # Calculate new reserves
        new_reserve_in = initial_reserve_in + amount_in
        new_reserve_out = initial_reserve_out - amount_out
        new_k = new_reserve_in * new_reserve_out
        
        # k should be preserved (minus fees)
        # With 0.3% fee, k should actually increase slightly
        assert new_k >= initial_k
        assert new_k < initial_k * Decimal('1.01')  # Not too much increase


class TestCurveStableSwapMath:
    """Test Curve StableSwap math implementation."""
    
    def test_balanced_pool_low_slippage(self):
        """Test that balanced pools have low slippage."""
        curve_math = CurveStableSwapMath(amplification_coefficient=Decimal('100'))
        
        # Balanced 2-pool
        balances = [Decimal('1000000'), Decimal('1000000')]
        
        # Small swap should have minimal impact
        amount_out = curve_math.get_dy(0, 1, Decimal('1000'), balances)
        
        # Should get close to 1000 (minus small fee)
        assert amount_out > Decimal('990')   # More than 99%
        assert amount_out < Decimal('1000')  # Less than 100%
    
    def test_imbalanced_pool_higher_slippage(self):
        """Test that imbalanced pools have higher slippage."""
        curve_math = CurveStableSwapMath(amplification_coefficient=Decimal('100'))
        
        # Imbalanced pool (one token scarce)
        balances = [Decimal('2000000'), Decimal('500000')]
        
        # Try to get the scarce token
        amount_out = curve_math.get_dy(0, 1, Decimal('10000'), balances)
        
        # Should be significantly less than proportional
        proportional = Decimal('5000')  # 10k * (500k/2000k) = 2.5k expected
        assert amount_out < proportional
    
    def test_invariant_calculation(self):
        """Test that D calculation converges."""
        curve_math = CurveStableSwapMath(amplification_coefficient=Decimal('100'))
        
        balances = [Decimal('1000000'), Decimal('1000000')]
        D = curve_math.get_D(balances)
        
        # D should be approximately sum of balances for balanced pool
        expected_D = sum(balances)
        assert abs(D - expected_D) < expected_D * Decimal('0.01')  # Within 1%


class TestBalancerWeightedMath:
    """Test Balancer weighted pool math implementation."""
    
    def test_spot_price_calculation(self):
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
        
        # Price should reflect weights: (100/0.8) / (600k/0.2) = 125 / 3M = ~0.00004
        expected = Decimal('0.00004')
        assert abs(spot_price - expected) < expected * Decimal('0.1')  # Within 10%
    
    def test_weighted_swap(self):
        """Test swap in weighted pool."""
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
        assert amount_out < Decimal('10000')  # <$10000


class TestUniswapV3Math:
    """Test Uniswap V3 math implementation."""
    
    def test_tick_to_price_conversion(self):
        """Test tick to price conversion."""
        v3_math = UniswapV3Math()
        
        # Tick 0 should give price = 1
        sqrt_price_x96 = v3_math.tick_to_sqrt_price_x96(0)
        price = v3_math.sqrt_price_x96_to_price(sqrt_price_x96)
        
        assert abs(price - Decimal('1')) < Decimal('0.0001')
    
    def test_amount_calculations(self):
        """Test amount delta calculations."""
        v3_math = UniswapV3Math()
        
        # Test amount0 delta calculation
        sqrt_price_a = v3_math.tick_to_sqrt_price_x96(-100)
        sqrt_price_b = v3_math.tick_to_sqrt_price_x96(100)
        liquidity = Decimal('1000000')
        
        amount0 = v3_math.get_amount0_delta(sqrt_price_a, sqrt_price_b, liquidity)
        amount1 = v3_math.get_amount1_delta(sqrt_price_a, sqrt_price_b, liquidity)
        
        # Both amounts should be positive
        assert amount0 > 0
        assert amount1 > 0


def test_comparative_analysis():
    """Test that different implementations give reasonable comparative results."""
    # Same pool values
    reserve_in = Decimal('1000000')
    reserve_out = Decimal('1000000')
    amount_in = Decimal('10000')
    
    # V2 calculation
    v2_math = UniswapV2Math()
    v2_output = v2_math.calculate_amount_out(amount_in, reserve_in, reserve_out)
    
    # Curve calculation (should be very similar for balanced stablecoin pool)
    curve_math = CurveStableSwapMath(amplification_coefficient=Decimal('100'))
    curve_output = curve_math.get_dy(0, 1, amount_in, [reserve_in, reserve_out])
    
    # Balancer 50/50 calculation (should be very similar to V2)
    balancer_math = BalancerWeightedMath()
    balancer_output = balancer_math.calculate_out_given_in(
        reserve_in, Decimal('0.5'), reserve_out, Decimal('0.5'), amount_in
    )
    
    # All should give similar results for balanced pools
    assert abs(v2_output - curve_output) < v2_output * Decimal('0.1')  # Within 10%
    assert abs(v2_output - balancer_output) < v2_output * Decimal('0.1')  # Within 10%


if __name__ == "__main__":
    pytest.main([__file__])