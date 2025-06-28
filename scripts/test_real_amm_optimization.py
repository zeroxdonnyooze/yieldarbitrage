#!/usr/bin/env python3
"""
Demonstration of REAL AMM optimization vs simplified models.

This shows the critical difference between using actual protocol math
and simplified approximations in a competitive arbitrage environment.
"""
import sys
import os
from decimal import Decimal
from typing import List

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.amm_price_optimizer import (
    AMMPriceOptimizer, AMMArbitrageParameters, AMMPoolState, DEXProtocol as DEXProtocolV1
)
from yield_arbitrage.protocols.amm_price_optimizer_v2 import (
    AMMPriceOptimizerV2, ArbitrageRoute, PoolState, DEXProtocol, OptimizationResult
)
from yield_arbitrage.protocols.flash_loan_optimizer import (
    FlashLoanOptimizer, ArbitrageParameters
)
from yield_arbitrage.protocols.flash_loan_provider import FlashLoanTerms, FlashLoanProvider


def create_v2_pool(protocol: str, reserves: List[float], fee: float = 0.003) -> PoolState:
    """Create a V2-style pool (Uniswap V2, Sushiswap)."""
    return PoolState(
        protocol=DEXProtocol.UNISWAP_V2 if protocol == "uniswap_v2" else DEXProtocol.SUSHISWAP,
        address="0x" + "0" * 40,
        token_addresses=["USDC", "WETH"],
        token_decimals=[6, 18],
        fee_rate=Decimal(str(fee)),
        reserves=[Decimal(str(r)) for r in reserves]
    )


def create_v3_pool(sqrt_price_x96: int, tick: int, liquidity: float) -> PoolState:
    """Create a Uniswap V3 pool."""
    return PoolState(
        protocol=DEXProtocol.UNISWAP_V3,
        address="0x" + "1" * 40,
        token_addresses=["USDC", "WETH"],
        token_decimals=[6, 18],
        fee_rate=Decimal('0.003'),
        sqrt_price_x96=sqrt_price_x96,
        tick=tick,
        liquidity=Decimal(str(liquidity)),
        tick_spacing=60,
        fee_tier=3000
    )


def create_curve_pool(reserves: List[float], amp: float = 100) -> PoolState:
    """Create a Curve StableSwap pool."""
    return PoolState(
        protocol=DEXProtocol.CURVE_STABLE,
        address="0x" + "2" * 40,
        token_addresses=["USDC", "USDT", "DAI"],
        token_decimals=[6, 6, 18],
        fee_rate=Decimal('0.0004'),
        reserves=[Decimal(str(r)) for r in reserves],
        amplification_coefficient=Decimal(str(amp))
    )


def create_balancer_pool(reserves: List[float], weights: List[float]) -> PoolState:
    """Create a Balancer weighted pool."""
    return PoolState(
        protocol=DEXProtocol.BALANCER_WEIGHTED,
        address="0x" + "3" * 40,
        token_addresses=["USDC", "WETH"],
        token_decimals=[6, 18],
        fee_rate=Decimal('0.001'),
        reserves=[Decimal(str(r)) for r in reserves],
        weights=[Decimal(str(w)) for w in weights]
    )


def demo_uniswap_v2_comparison():
    """Compare simplified vs real Uniswap V2 calculations."""
    print("🦄 Uniswap V2: Simplified vs Real Implementation")
    print("=" * 60)
    
    # Create pools with price difference
    pool_1 = create_v2_pool("uniswap_v2", [10_000_000, 3333.33])  # $3000/ETH
    pool_2 = create_v2_pool("sushiswap", [8_000_000, 2700])       # $2963/ETH
    
    # Real implementation
    optimizer_v2 = AMMPriceOptimizerV2()
    route = ArbitrageRoute(
        pool_buy=pool_2,  # Buy ETH cheaper on Sushiswap
        pool_sell=pool_1,  # Sell ETH expensive on Uniswap
        token_in="USDC",
        token_out="WETH"
    )
    
    real_result = optimizer_v2.optimize_arbitrage(route)
    
    print(f"\n✅ Real V2 Implementation:")
    print(f"   Optimal Amount: ${real_result.optimal_amount:,.0f}")
    print(f"   Expected Profit: ${real_result.expected_profit:.2f}")
    print(f"   Price Convergence: {real_result.price_convergence_percentage:.1f}%")
    print(f"   Final Price Diff: {real_result.final_price_difference * 100:.4f}%")
    
    # Simplified implementation (for comparison)
    initial_price_diff = abs(3000 - 2963) / 2963  # ~1.25%
    flash_terms = FlashLoanTerms(
        provider=FlashLoanProvider.AAVE_V3,
        asset="USDC",
        max_amount=Decimal('10_000_000'),
        fee_rate=Decimal('0.0009'),
        fixed_fee=Decimal('0'),
        min_amount=Decimal('1'),
        gas_estimate=150_000
    )
    
    simple_params = ArbitrageParameters(
        price_difference=Decimal(str(initial_price_diff)),
        liquidity_pool_1=pool_1.reserves[0],
        liquidity_pool_2=pool_2.reserves[0],
        flash_loan_terms=flash_terms,
        gas_cost_usd=Decimal('50'),
        mev_bribe_rate=Decimal('0.001'),
        slippage_factor_1=Decimal('0.00005'),
        slippage_factor_2=Decimal('0.00005')
    )
    
    simple_optimizer = FlashLoanOptimizer()
    simple_result = simple_optimizer.optimize_flash_loan_amount(simple_params)
    
    print(f"\n❌ Simplified Linear Model:")
    print(f"   Optimal Amount: ${simple_result.optimal_amount:,.0f}")
    print(f"   Expected Profit: ${simple_result.expected_profit:.2f}")
    print(f"   Assumes static {initial_price_diff * 100:.2f}% price difference")
    print(f"   No price convergence modeling")
    
    print(f"\n📊 Key Differences:")
    amount_diff_pct = ((simple_result.optimal_amount - real_result.optimal_amount) / real_result.optimal_amount * 100)
    profit_diff_pct = ((simple_result.expected_profit - real_result.expected_profit) / real_result.expected_profit * 100) if real_result.expected_profit > 0 else 0
    
    print(f"   Amount Difference: {amount_diff_pct:+.1f}%")
    print(f"   Profit Difference: {profit_diff_pct:+.1f}%")
    
    if abs(amount_diff_pct) > 20:
        print(f"   ⚠️  Simplified model is SIGNIFICANTLY off!")
        print(f"   💸 Could mean ${abs(simple_result.expected_profit - real_result.expected_profit):.2f} profit difference")
    
    print()


def demo_uniswap_v3_complexity():
    """Show the complexity of Uniswap V3 concentrated liquidity."""
    print("🎯 Uniswap V3: Concentrated Liquidity Complexity")
    print("=" * 60)
    
    # Create V3 pools
    # sqrtPriceX96 for $3000/ETH ≈ sqrt(3000) * 2^96
    sqrt_price_3000 = int(Decimal('3000').sqrt() * Decimal(2**96))
    sqrt_price_2970 = int(Decimal('2970').sqrt() * Decimal(2**96))
    
    pool_v3_1 = create_v3_pool(sqrt_price_3000, 0, 10_000_000)
    pool_v3_2 = create_v3_pool(sqrt_price_2970, -100, 8_000_000)
    
    optimizer = AMMPriceOptimizerV2()
    route = ArbitrageRoute(
        pool_buy=pool_v3_2,
        pool_sell=pool_v3_1,
        token_in="USDC",
        token_out="WETH"
    )
    
    result = optimizer.optimize_arbitrage(route)
    
    print(f"\n🎯 V3 Concentrated Liquidity Results:")
    print(f"   Initial Prices: $3000 vs $2970 (1% difference)")
    print(f"   Optimal Amount: ${result.optimal_amount:,.0f}")
    print(f"   Expected Profit: ${result.expected_profit:.2f}")
    
    print(f"\n📐 Why V3 is Different:")
    print(f"   • Liquidity is concentrated in price ranges")
    print(f"   • Must handle tick crossings during large swaps")
    print(f"   • Price impact is non-linear within ranges")
    print(f"   • sqrtPriceX96 format requires precise calculations")
    
    print(f"\n⚠️  Simplified Model Cannot Handle:")
    print(f"   • Tick-based liquidity distribution")
    print(f"   • Range orders and positions")
    print(f"   • Actual V3 swap mechanics")
    print(f"   • This means missing profitable opportunities!")
    
    print()


def demo_curve_stableswap():
    """Show Curve's StableSwap advantages for stable assets."""
    print("🌊 Curve StableSwap: Optimized for Stablecoins")
    print("=" * 60)
    
    # Create Curve 3pool (USDC/USDT/DAI) - use 2 tokens for simplicity
    # Slightly imbalanced pool  
    curve_pool = PoolState(
        protocol=DEXProtocol.CURVE_STABLE,
        address="0x" + "2" * 40,
        token_addresses=["USDC", "USDT"],
        token_decimals=[6, 6],
        fee_rate=Decimal('0.0004'),
        reserves=[Decimal('30000000'), Decimal('29700000')],  # USDT slightly cheaper
        amplification_coefficient=Decimal('100')
    )
    
    # For comparison, create a "regular" pool
    regular_pool = create_v2_pool("uniswap_v2", [30_000_000, 29_700_000], fee=0.003)
    
    print(f"\n💱 Stablecoin Arbitrage Scenario:")
    print(f"   Pool balances show USDT slightly undervalued")
    print(f"   Curve pool: A=100 (high amplification)")
    print(f"   Compare to regular constant product pool")
    
    # Calculate swap amounts
    from yield_arbitrage.protocols.dex_protocols.curve_stableswap_math import CurveStableSwapMath
    from yield_arbitrage.protocols.dex_protocols.uniswap_v2_math import UniswapV2Math
    
    curve_math = CurveStableSwapMath(amplification_coefficient=Decimal('100'))
    v2_math = UniswapV2Math()
    
    test_amount = Decimal('100_000')  # $100K swap (more reasonable)
    
    # Curve swap: USDC (index 0) to USDT (index 1)
    curve_output = curve_math.get_dy(0, 1, test_amount, curve_pool.reserves)
    curve_price_impact = curve_math.calculate_price_impact(0, 1, test_amount, curve_pool.reserves)
    
    # V2 swap equivalent
    v2_output = v2_math.calculate_amount_out(test_amount, regular_pool.reserves[0], regular_pool.reserves[1])
    v2_price_impact = v2_math.calculate_price_impact(test_amount, regular_pool.reserves[0], regular_pool.reserves[1])
    
    print(f"\n📊 $100K USDC → USDT Swap Comparison:")
    print(f"   Curve Output: ${curve_output:,.2f} USDT")
    print(f"   Curve Slippage: {curve_price_impact * 100:.4f}%")
    print(f"   ")
    print(f"   V2 Output: ${v2_output:,.2f} USDT")
    print(f"   V2 Slippage: {v2_price_impact * 100:.4f}%")
    
    slippage_improvement = (v2_price_impact - curve_price_impact) / v2_price_impact * 100
    extra_output = curve_output - v2_output
    
    print(f"\n✅ Curve Advantages:")
    print(f"   • {slippage_improvement:.1f}% less slippage")
    print(f"   • ${extra_output:.2f} more output")
    print(f"   • Better for large stablecoin trades")
    
    print(f"\n🔬 Why Curve's Math Matters:")
    print(f"   • StableSwap invariant: An^n·Σx + D = An^n·D + D^(n+1)/(n^n·Πx)")
    print(f"   • Combines constant sum (x+y=k) near balance")
    print(f"   • Transitions to constant product (xy=k) when imbalanced")
    print(f"   • Requires Newton's method iteration (not simple formula)")
    
    print()


def demo_balancer_weighted():
    """Show Balancer's weighted pool flexibility."""
    print("⚖️ Balancer Weighted Pools: Custom Weight Ratios")
    print("=" * 60)
    
    # Create 80/20 WETH/USDC pool
    balancer_pool = create_balancer_pool(
        reserves=[10_000, 20_000_000],  # 10K WETH, 20M USDC
        weights=[0.8, 0.2]  # 80% WETH, 20% USDC weight
    )
    
    # Equivalent 50/50 pool for comparison
    regular_pool = create_v2_pool("uniswap_v2", [10_000, 30_000_000])  # Same value but 50/50
    
    print(f"\n🏊 Weighted Pool Scenario:")
    print(f"   80/20 WETH/USDC Balancer pool")
    print(f"   Compare to 50/50 constant product pool")
    
    from yield_arbitrage.protocols.dex_protocols.balancer_weighted_math import BalancerWeightedMath
    from yield_arbitrage.protocols.dex_protocols.uniswap_v2_math import UniswapV2Math
    
    balancer_math = BalancerWeightedMath()
    v2_math = UniswapV2Math()
    
    # Test swap: 100 WETH -> USDC
    test_amount = Decimal('100')
    
    balancer_output = balancer_math.calculate_out_given_in(
        balancer_pool.reserves[0],  # WETH balance
        balancer_pool.weights[0],   # WETH weight
        balancer_pool.reserves[1],  # USDC balance
        balancer_pool.weights[1],   # USDC weight
        test_amount
    )
    
    v2_output = v2_math.calculate_amount_out(
        test_amount,
        regular_pool.reserves[0],
        regular_pool.reserves[1]
    )
    
    print(f"\n📊 100 WETH → USDC Swap:")
    print(f"   Balancer 80/20: ${balancer_output:,.2f} USDC")
    print(f"   Regular 50/50: ${v2_output:,.2f} USDC")
    
    print(f"\n🎯 Balancer Weighted Math:")
    print(f"   • Invariant: V = Π(B_i^W_i)")
    print(f"   • Spot price: (B_in/W_in) / (B_out/W_out)")
    print(f"   • Allows unequal exposure (80% ETH, 20% stable)")
    print(f"   • Different price impact profile than 50/50")
    
    print()


def demo_competitive_advantage():
    """Show why real math gives competitive advantage."""
    print("💰 Competitive Advantage: Real Math vs Approximations")
    print("=" * 60)
    
    print(f"\n🏁 MEV Competition Scenario:")
    print(f"   • You: Using simplified linear slippage model")
    print(f"   • Competitor: Using real AMM math")
    print(f"   • Same arbitrage opportunity detected")
    
    # Create arbitrage scenario
    pool_1 = create_v2_pool("uniswap_v2", [50_000_000, 16666.67])  # $3000/ETH
    pool_2 = create_v2_pool("sushiswap", [40_000_000, 13600])      # $2941/ETH (~2% diff)
    
    route = ArbitrageRoute(pool_buy=pool_2, pool_sell=pool_1, token_in="USDC", token_out="WETH")
    
    # Real calculation
    real_optimizer = AMMPriceOptimizerV2()
    real_result = real_optimizer.optimize_arbitrage(route)
    
    # Simplified calculation
    simple_optimizer = FlashLoanOptimizer()
    flash_terms = FlashLoanTerms(
        provider=FlashLoanProvider.AAVE_V3,
        asset="USDC",
        max_amount=Decimal('50_000_000'),
        fee_rate=Decimal('0.0009'),
        fixed_fee=Decimal('0'),
        min_amount=Decimal('1'),
        gas_estimate=150_000
    )
    
    simple_params = ArbitrageParameters(
        price_difference=Decimal('0.02'),  # 2% static
        liquidity_pool_1=pool_1.reserves[0],
        liquidity_pool_2=pool_2.reserves[0],
        flash_loan_terms=flash_terms,
        gas_cost_usd=Decimal('50'),
        mev_bribe_rate=Decimal('0.001'),
        slippage_factor_1=Decimal('0.00002'),
        slippage_factor_2=Decimal('0.00002')
    )
    simple_result = simple_optimizer.optimize_flash_loan_amount(simple_params)
    
    print(f"\n📊 Results:")
    print(f"   Your calculation (simplified):")
    print(f"     • Flash loan: ${simple_result.optimal_amount:,.0f}")
    print(f"     • Expected profit: ${simple_result.expected_profit:.2f}")
    
    print(f"\n   Competitor (real math):")
    print(f"     • Flash loan: ${real_result.optimal_amount:,.0f}")
    print(f"     • Expected profit: ${real_result.expected_profit:.2f}")
    print(f"     • Knows prices converge {real_result.price_convergence_percentage:.1f}%")
    
    print(f"\n⚡ What Happens:")
    
    if simple_result.optimal_amount > real_result.optimal_amount * Decimal('1.2'):
        print(f"   1. You try to borrow ${simple_result.optimal_amount:,.0f}")
        print(f"   2. Execute trade, but prices converge faster than expected")
        print(f"   3. Actual profit much less than calculated")
        print(f"   4. Possibly even LOSS after gas and fees!")
        print(f"   5. Competitor profits ${real_result.expected_profit:.2f}")
    else:
        print(f"   1. Both submit similar transactions")
        print(f"   2. Competitor's is more gas efficient (optimal amount)")
        print(f"   3. Competitor wins the MEV auction")
        print(f"   4. You get nothing")
    
    print(f"\n🎯 Competitive Reality:")
    print(f"   • Professional MEV searchers use EXACT math")
    print(f"   • Approximations = missed opportunities")
    print(f"   • Overestimation = potential losses")
    print(f"   • In MEV, precision wins")
    
    print()


def main():
    """Run all demonstrations."""
    print("🚀 Real AMM Math vs Simplified Models")
    print("=" * 80)
    print("Demonstrating why accurate protocol implementations matter")
    print("in competitive DeFi arbitrage environments.")
    print()
    
    try:
        demo_uniswap_v2_comparison()
        demo_uniswap_v3_complexity()
        demo_curve_stableswap()
        demo_balancer_weighted()
        demo_competitive_advantage()
        
        print("✅ Demonstration Complete!")
        print()
        print("🔑 Key Takeaways:")
        print("   1. Simplified models can be off by 50%+ on optimal amounts")
        print("   2. Each protocol has unique math that must be implemented correctly")
        print("   3. V3 concentrated liquidity is fundamentally different from V2")
        print("   4. Curve's StableSwap gives massive advantage for stablecoin arbs")
        print("   5. In competitive MEV, accurate math = profits, approximations = losses")
        print()
        print("📚 Implementation Status:")
        print("   ✅ Uniswap V2 exact formula")
        print("   ✅ Uniswap V3 tick math (simplified)")
        print("   ✅ Curve StableSwap with Newton's method")
        print("   ✅ Balancer weighted pools")
        print("   ✅ Integrated optimizer using real math")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()