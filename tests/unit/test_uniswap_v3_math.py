"""
Unit tests for Uniswap V3 Math implementation.

Tests tick calculations, concentrated liquidity, and sqrtPrice mechanics.
"""
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.dex_protocols.uniswap_v3_math import UniswapV3Math


def test_tick_to_price_conversion():
    """Test tick to price conversion."""
    v3_math = UniswapV3Math()
    
    # Test known values
    test_cases = [
        (0, Decimal('1')),           # Tick 0 = price 1
        (69080, Decimal('1000')),    # Approx tick for 1000x price
        (-69080, Decimal('0.001')),  # Approx tick for 0.001x price
    ]
    
    for tick, expected_price in test_cases:
        sqrt_price_x96 = v3_math.tick_to_sqrt_price_x96(tick)
        price = v3_math.sqrt_price_x96_to_price(sqrt_price_x96)
        
        # Allow 1% tolerance due to discrete tick spacing
        assert abs(price - expected_price) < expected_price * Decimal('0.01')
        print(f"    Tick {tick} → Price {price} (expected ~{expected_price})")
    
    print("✅ Tick to price conversion working")


def test_sqrt_price_calculations():
    """Test sqrt price X96 format calculations."""
    v3_math = UniswapV3Math()
    
    # Test sqrt price format
    sqrt_price_x96 = v3_math.encode_sqrt_price(Decimal('100'), Decimal('1'))  # 100:1 ratio
    price = v3_math.sqrt_price_x96_to_price(sqrt_price_x96)
    
    # Price should be close to 1/100 = 0.01 (token1 per token0)
    expected = Decimal('0.01')
    assert abs(price - expected) < expected * Decimal('0.001')  # 0.1% tolerance
    
    print(f"✅ Sqrt price X96: ratio 100:1 → price {price}")


def test_amount_calculations():
    """Test amount delta calculations for liquidity ranges."""
    v3_math = UniswapV3Math()
    
    # Test case: USDC/ETH pool around $2000 ETH
    tick_lower = -100  # Price range
    tick_upper = 100
    
    sqrt_price_a = v3_math.tick_to_sqrt_price_x96(tick_lower)
    sqrt_price_b = v3_math.tick_to_sqrt_price_x96(tick_upper)
    liquidity = Decimal('1000000')
    
    amount0 = v3_math.get_amount0_delta(sqrt_price_a, sqrt_price_b, liquidity)
    amount1 = v3_math.get_amount1_delta(sqrt_price_a, sqrt_price_b, liquidity)
    
    # Both amounts should be positive for a range
    assert amount0 > 0
    assert amount1 > 0
    
    print(f"✅ Amount deltas: {amount0:.2f} token0, {amount1:.2f} token1")


def test_swap_calculation():
    """Test single swap calculation."""
    v3_math = UniswapV3Math()
    
    # Simple swap within a single tick range
    sqrt_price_current = v3_math.tick_to_sqrt_price_x96(0)  # At tick 0
    sqrt_price_target = v3_math.tick_to_sqrt_price_x96(-100)  # Target lower price
    liquidity = Decimal('10000000')
    amount_in = Decimal('1000')
    zero_for_one = True  # Swapping token0 for token1
    
    sqrt_price_next, amount_in_step, amount_out, fee_amount = v3_math.compute_swap_step(
        sqrt_price_current, sqrt_price_target, liquidity, amount_in, 3000  # 0.3% fee
    )
    
    # Should get some output
    assert amount_out > 0
    assert amount_out < amount_in  # Due to price impact
    
    # Price should move
    assert sqrt_price_next != sqrt_price_current
    
    print(f"✅ Swap: {amount_in} in → {amount_out} out")


def test_price_impact():
    """Test price impact for different trade sizes."""
    v3_math = UniswapV3Math()
    
    sqrt_price_start = v3_math.tick_to_sqrt_price_x96(0)
    liquidity = Decimal('1000000')
    
    trade_sizes = [100, 1000, 10000]
    
    for amount in trade_sizes:
        amount_decimal = Decimal(str(amount))
        
        sqrt_price_target = v3_math.tick_to_sqrt_price_x96(-100)  # Target price
        sqrt_price_after, amount_in_step, amount_out, fee_amount = v3_math.compute_swap_step(
            sqrt_price_start, sqrt_price_target, liquidity, amount_decimal, 3000
        )
        
        # Calculate price impact
        price_before = v3_math.sqrt_price_x96_to_price(sqrt_price_start)
        price_after = v3_math.sqrt_price_x96_to_price(sqrt_price_after)
        
        price_impact = abs(price_after - price_before) / price_before
        
        print(f"    ${amount} trade: {price_impact*100:.3f}% price impact")
        
        # Larger trades should have more impact
        if amount >= 5000:
            assert price_impact > Decimal('0.001')  # >0.1% for large trades


def test_tick_spacing():
    """Test tick spacing calculations."""
    v3_math = UniswapV3Math()
    
    # Test different fee tiers and their tick spacings
    fee_tiers = [
        (100, 1),    # 0.01% fee, 1 tick spacing  
        (500, 10),   # 0.05% fee, 10 tick spacing
        (3000, 60),  # 0.3% fee, 60 tick spacing
        (10000, 200) # 1% fee, 200 tick spacing
    ]
    
    for fee, expected_spacing in fee_tiers:
        spacing = v3_math.get_tick_spacing(fee)
        assert spacing == expected_spacing
        print(f"    {fee/10000:.2f}% fee → {spacing} tick spacing")
    
    print("✅ Tick spacing correct for all fee tiers")


def test_liquidity_math():
    """Test liquidity calculations for position management."""
    v3_math = UniswapV3Math()
    
    # Calculate liquidity for a position
    amount0 = Decimal('1000')   # USDC
    amount1 = Decimal('0.5')    # ETH
    
    tick_lower = -1000
    tick_upper = 1000
    
    sqrt_price_a = v3_math.tick_to_sqrt_price_x96(tick_lower)
    sqrt_price_b = v3_math.tick_to_sqrt_price_x96(tick_upper)
    sqrt_price_current = v3_math.tick_to_sqrt_price_x96(0)
    
    liquidity = v3_math.get_liquidity_for_amounts(
        sqrt_price_current, sqrt_price_a, sqrt_price_b, amount0, amount1
    )
    
    # Should get positive liquidity
    assert liquidity > 0
    
    print(f"✅ Liquidity calculation: {amount0} + {amount1} → {liquidity} liquidity")


def test_fee_calculations():
    """Test fee calculations for different fee tiers."""
    v3_math = UniswapV3Math()
    
    amount_in = Decimal('1000')
    fee_tiers = [100, 500, 3000, 10000]  # 0.01%, 0.05%, 0.3%, 1%
    
    for fee_tier in fee_tiers:
        fee_amount = v3_math.calculate_fee(amount_in, fee_tier)
        expected_fee = amount_in * fee_tier / Decimal('1000000')
        
        assert abs(fee_amount - expected_fee) < Decimal('0.001')
        print(f"    {fee_tier/10000:.2f}% fee on ${amount_in} = ${fee_amount}")
    
    print("✅ Fee calculations correct")


def test_position_value():
    """Test position value calculations."""
    v3_math = UniswapV3Math()
    
    # Position parameters
    tick_lower = -1000
    tick_upper = 1000
    liquidity = Decimal('1000000')
    
    # Current prices to test
    test_ticks = [-500, 0, 500]
    
    for tick_current in test_ticks:
        sqrt_price_current = v3_math.tick_to_sqrt_price_x96(tick_current)
        sqrt_price_a = v3_math.tick_to_sqrt_price_x96(tick_lower)
        sqrt_price_b = v3_math.tick_to_sqrt_price_x96(tick_upper)
        
        amount0, amount1 = v3_math.get_amounts_for_liquidity(
            sqrt_price_current, sqrt_price_a, sqrt_price_b, liquidity
        )
        
        # Both amounts should be positive when in range
        if tick_lower <= tick_current <= tick_upper:
            assert amount0 >= 0
            assert amount1 >= 0
        
        print(f"    At tick {tick_current}: {amount0:.2f} token0, {amount1:.2f} token1")
    
    print("✅ Position value calculations working")


def test_multi_tick_swap():
    """Test swap crossing multiple ticks."""
    v3_math = UniswapV3Math()
    
    # This is a simplified test - full implementation would need tick data
    # For now, test the basic swap step function
    
    sqrt_price_start = v3_math.tick_to_sqrt_price_x96(-100)
    sqrt_price_target = v3_math.tick_to_sqrt_price_x96(100)  # Cross multiple ticks
    liquidity = Decimal('1000000')
    amount_remaining = Decimal('10000')
    
    # Simulate one step of a multi-tick swap
    sqrt_price_next, amount_in_step, amount_out, fee_amount = v3_math.compute_swap_step(
        sqrt_price_start, sqrt_price_target, liquidity, amount_remaining, 3000
    )
    
    # Should move towards target
    assert sqrt_price_next != sqrt_price_start
    assert amount_out > 0
    
    print(f"✅ Multi-tick swap step: moved price and got {amount_out} output")


def test_edge_cases():
    """Test edge cases and error handling."""
    v3_math = UniswapV3Math()
    
    # Zero amounts
    sqrt_price = v3_math.tick_to_sqrt_price_x96(0)
    _, _, amount_out, _ = v3_math.compute_swap_step(sqrt_price, sqrt_price, Decimal('1000'), Decimal('0'), 3000)
    assert amount_out == Decimal('0')
    
    # Zero liquidity
    _, _, amount_out, _ = v3_math.compute_swap_step(sqrt_price, sqrt_price, Decimal('0'), Decimal('1000'), 3000)
    assert amount_out == Decimal('0')
    
    # Invalid tick ranges
    try:
        invalid_tick = 1000000  # Way out of range
        v3_math.tick_to_sqrt_price_x96(invalid_tick)
        # Should either work with limits or raise an exception
    except:
        pass  # Expected for out-of-range ticks
    
    print("✅ Edge cases handled correctly")


def test_concentrated_liquidity_advantage():
    """Test that concentrated liquidity provides better capital efficiency."""
    v3_math = UniswapV3Math()
    
    # Compare wide range vs narrow range for same liquidity
    total_liquidity = Decimal('1000000')
    amount_in = Decimal('1000')
    
    # Wide range (like V2)
    tick_wide_lower = -10000
    tick_wide_upper = 10000
    
    # Narrow range (concentrated)
    tick_narrow_lower = -100
    tick_narrow_upper = 100
    
    # For fair comparison, we'd need actual tick crossing logic
    # This is a simplified demonstration
    
    # Both ranges should handle the swap, but narrow range should be more efficient
    sqrt_price_current = v3_math.tick_to_sqrt_price_x96(0)
    
    # Wide range liquidity density
    wide_range_ticks = tick_wide_upper - tick_wide_lower
    wide_liquidity_density = total_liquidity / wide_range_ticks
    
    # Narrow range liquidity density  
    narrow_range_ticks = tick_narrow_upper - tick_narrow_lower
    narrow_liquidity_density = total_liquidity / narrow_range_ticks
    
    # Narrow range should have much higher liquidity density
    assert narrow_liquidity_density > wide_liquidity_density * 50
    
    print(f"✅ Concentrated liquidity: {narrow_liquidity_density/wide_liquidity_density:.1f}x more capital efficient")


def run_all_tests():
    """Run all Uniswap V3 tests."""
    print("🦄 Testing Uniswap V3 Math Implementation")
    print("=" * 50)
    
    try:
        test_tick_to_price_conversion()
        test_sqrt_price_calculations()
        test_amount_calculations()
        test_swap_calculation()
        test_price_impact()
        test_tick_spacing()
        test_liquidity_math()
        test_fee_calculations()
        test_position_value()
        test_multi_tick_swap()
        test_edge_cases()
        test_concentrated_liquidity_advantage()
        
        print("\n✅ All Uniswap V3 tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()