#!/usr/bin/env python3
"""
Demonstration script comparing simplified vs true AMM price reversion optimization.

This script shows the difference between linear slippage models and true
AMM constant product mechanics for flash loan optimization.
"""
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.amm_price_optimizer import (
    AMMPriceOptimizer, AMMArbitrageParameters, AMMPoolState, DEXProtocol
)
from yield_arbitrage.protocols.flash_loan_optimizer import (
    FlashLoanOptimizer, ArbitrageParameters, FlashLoanTerms, FlashLoanProvider
)


def create_amm_pools_with_price_difference():
    """Create AMM pools with realistic price difference."""
    # Uniswap V2 USDC/WETH pool (expensive WETH)
    pool_1 = AMMPoolState(
        protocol=DEXProtocol.UNISWAP_V2,
        token_a_reserves=Decimal('10_000_000'),  # 10M USDC
        token_b_reserves=Decimal('3333'),        # 3333 WETH = $3000/ETH
        fee_rate=Decimal('0.003')                # 0.3% fee
    )
    
    # Curve USDC/WETH pool (cheaper WETH) 
    pool_2 = AMMPoolState(
        protocol=DEXProtocol.CURVE_STABLE,
        token_a_reserves=Decimal('8_000_000'),   # 8M USDC  
        token_b_reserves=Decimal('2700'),        # 2700 WETH = $2963/ETH
        fee_rate=Decimal('0.0004'),              # 0.04% fee (Curve is cheaper)
        amplification_coefficient=Decimal('100') # Curve A parameter
    )
    
    return pool_1, pool_2


def demo_price_reversion_mechanics():
    """Demonstrate true AMM price reversion mechanics."""
    print("🎯 True AMM Price Reversion Mechanics:")
    print("   Modeling constant product formula and price convergence")
    
    pool_1, pool_2 = create_amm_pools_with_price_difference()
    
    # Show initial state
    price_1 = pool_1.price_a_to_b
    price_2 = pool_2.price_a_to_b
    price_diff = abs(price_1 - price_2) / max(price_1, price_2)
    
    print(f"\n📊 Initial Pool States:")
    print(f"   Pool 1 (Uniswap): {price_1:.2f} USDC/WETH (${1/price_1:.2f}/ETH)")
    print(f"   Pool 2 (Curve):   {price_2:.2f} USDC/WETH (${1/price_2:.2f}/ETH)")
    print(f"   Price Difference: {price_diff * 100:.3f}%")
    print(f"   Arbitrage: Buy WETH on Curve (cheaper), sell on Uniswap (expensive)")
    
    # Create parameters
    params = AMMArbitrageParameters(
        pool_1=pool_1,
        pool_2=pool_2,
        flash_loan_fee_rate=Decimal('0.0009'),  # Aave 0.09%
        gas_cost_usd=Decimal('50'),
        mev_bribe_rate=Decimal('0.001')
    )
    
    # Optimize
    optimizer = AMMPriceOptimizer()
    result = optimizer.optimize_arbitrage_amount(params)
    
    print(f"\n🎯 AMM Optimization Results:")
    print(f"   Optimal Flash Loan: ${result.optimal_amount:,.0f}")
    print(f"   Expected Profit: ${result.expected_profit:.2f}")
    print(f"   Price Convergence: {result.price_convergence_percentage:.1f}%")
    
    print(f"\n📈 Price Impact Analysis:")
    print(f"   Initial Price Diff: {result.initial_price_difference * 100:.4f}%")
    print(f"   Final Price Diff: {result.final_price_difference * 100:.4f}%") 
    print(f"   Arbitrage Exhausted: {'✅ Yes' if result.arbitrage_exhausted else '❌ No'}")
    
    print(f"\n🔄 Trade Execution:")
    print(f"   1. Flash loan ${result.optimal_amount:,.0f} USDC")
    print(f"   2. Buy {result.amount_out_pool_2:.2f} WETH on Curve for ${result.amount_in_pool_2:,.0f} USDC")
    print(f"   3. Sell {result.amount_in_pool_1:.2f} WETH on Uniswap for ${result.amount_out_pool_1:,.0f} USDC")
    print(f"   4. Repay flash loan + fee: ${result.optimal_amount + result.flash_loan_fee:,.2f}")
    
    print(f"\n💧 Slippage Analysis:")
    print(f"   Pool 1 Slippage: {result.slippage_pool_1 * 100:.3f}%")
    print(f"   Pool 2 Slippage: {result.slippage_pool_2 * 100:.3f}%")
    print(f"   Total Slippage: {result.total_slippage * 100:.3f}%")
    
    print(f"\n💰 Final Pool States:")
    print(f"   Pool 1 Final: {result.pool_1_final.price_a_to_b:.4f} USDC/WETH")
    print(f"   Pool 2 Final: {result.pool_2_final.price_a_to_b:.4f} USDC/WETH")
    final_diff = abs(result.pool_1_final.price_a_to_b - result.pool_2_final.price_a_to_b) / max(result.pool_1_final.price_a_to_b, result.pool_2_final.price_a_to_b)
    print(f"   Remaining Arbitrage: {final_diff * 100:.4f}%")
    
    print()
    return result


def demo_price_impact_analysis():
    """Show how trade size impacts price convergence."""
    print("📊 Price Impact Analysis:")
    print("   How trade size affects price convergence and profitability")
    
    pool_1, pool_2 = create_amm_pools_with_price_difference()
    
    params = AMMArbitrageParameters(
        pool_1=pool_1,
        pool_2=pool_2,
        flash_loan_fee_rate=Decimal('0.0009'),
        gas_cost_usd=Decimal('50'),
        mev_bribe_rate=Decimal('0.001')
    )
    
    optimizer = AMMPriceOptimizer()
    
    # Test different amounts
    amounts = [
        Decimal('50000'),   # $50K
        Decimal('100000'),  # $100K  
        Decimal('200000'),  # $200K
        Decimal('500000'),  # $500K
        Decimal('1000000'), # $1M
        Decimal('2000000'), # $2M
    ]
    
    results = optimizer.analyze_price_impact(params, amounts)
    
    print(f"\n📈 Trade Size Impact:")
    print(f"{'Amount':<12} {'Profit':<10} {'Convergence':<12} {'Profit/Dollar':<15} {'Remaining Arb':<15}")
    print("-" * 70)
    
    for amount in amounts:
        result = results[amount]
        profit = result["gross_profit"]
        convergence = result["price_convergence_pct"]
        profit_per_dollar = result["marginal_profit_per_dollar"] * 10000  # Per $10K
        remaining_arb = result["final_price_diff"] * 100
        
        print(f"${amount/1000:>7.0f}K    ${profit:>7.2f}   {convergence:>9.1f}%      ${profit_per_dollar:>9.2f}/$10K     {remaining_arb:>9.4f}%")
    
    print()


def demo_simplified_vs_true_comparison():
    """Compare simplified linear model vs true AMM mechanics."""
    print("⚖️ Simplified vs True AMM Model Comparison:")
    print("   Showing the difference between linear slippage and true price reversion")
    
    # Setup identical scenarios
    pool_1, pool_2 = create_amm_pools_with_price_difference()
    
    initial_price_diff = abs(pool_1.price_a_to_b - pool_2.price_a_to_b) / max(pool_1.price_a_to_b, pool_2.price_a_to_b)
    
    # True AMM model
    amm_params = AMMArbitrageParameters(
        pool_1=pool_1,
        pool_2=pool_2,
        flash_loan_fee_rate=Decimal('0.0009'),
        gas_cost_usd=Decimal('50'),
        mev_bribe_rate=Decimal('0.001')
    )
    
    amm_optimizer = AMMPriceOptimizer()
    amm_result = amm_optimizer.optimize_arbitrage_amount(amm_params)
    
    # Simplified linear model (convert AMM pools to linear parameters)
    flash_loan_terms = FlashLoanTerms(
        provider=FlashLoanProvider.AAVE_V3,
        asset="USDC",
        max_amount=Decimal('10_000_000'),
        fee_rate=Decimal('0.0009'),
        fixed_fee=Decimal('0'),
        min_amount=Decimal('1'),
        gas_estimate=150_000
    )
    
    linear_params = ArbitrageParameters(
        price_difference=initial_price_diff,
        liquidity_pool_1=pool_1.token_a_reserves,  # Use USDC reserves
        liquidity_pool_2=pool_2.token_a_reserves,
        flash_loan_terms=flash_loan_terms,
        gas_cost_usd=Decimal('50'),
        mev_bribe_rate=Decimal('0.001'),
        slippage_factor_1=Decimal('0.00001'),  # Rough approximation
        slippage_factor_2=Decimal('0.00001')
    )
    
    linear_optimizer = FlashLoanOptimizer()
    linear_result = linear_optimizer.optimize_flash_loan_amount(linear_params)
    
    print(f"\n📊 Model Comparison Results:")
    print(f"{'Metric':<25} {'True AMM':<15} {'Simplified':<15} {'Difference':<15}")
    print("-" * 75)
    
    optimal_diff = ((linear_result.optimal_amount - amm_result.optimal_amount) / amm_result.optimal_amount * 100) if amm_result.optimal_amount > 0 else 0
    profit_diff = ((linear_result.expected_profit - amm_result.expected_profit) / amm_result.expected_profit * 100) if amm_result.expected_profit > 0 else 0
    
    print(f"{'Optimal Amount':<25} ${amm_result.optimal_amount:>10,.0f}   ${linear_result.optimal_amount:>10,.0f}   {optimal_diff:>10.1f}%")
    print(f"{'Expected Profit':<25} ${amm_result.expected_profit:>10.2f}   ${linear_result.expected_profit:>10.2f}   {profit_diff:>10.1f}%")
    print(f"{'Price Convergence':<25} {amm_result.price_convergence_percentage:>10.1f}%   {'N/A':<15} {'N/A':<15}")
    print(f"{'Final Price Diff':<25} {amm_result.final_price_difference*100:>10.4f}%   {'N/A':<15} {'N/A':<15}")
    
    print(f"\n🔍 Key Differences:")
    print(f"   • True AMM Model: Models actual price convergence and diminishing returns")
    print(f"   • Simplified Model: Uses linear slippage approximation")
    print(f"   • AMM shows price convergence: {amm_result.price_convergence_percentage:.1f}%")
    print(f"   • AMM finds optimal point where marginal profit = marginal cost")
    print(f"   • Simplified model may over/under-estimate optimal amounts")
    
    if abs(optimal_diff) > 10:
        print(f"   ⚠️  Significant difference in optimal amounts ({optimal_diff:+.1f}%)")
    
    print()
    return amm_result, linear_result


def demo_exhausted_arbitrage():
    """Show what happens when arbitrage is nearly exhausted."""
    print("🔚 Arbitrage Exhaustion Analysis:")
    print("   Demonstrating optimal point where most arbitrage is captured")
    
    # Create pools with smaller price difference
    pool_1 = AMMPoolState(
        protocol=DEXProtocol.UNISWAP_V2,
        token_a_reserves=Decimal('5_000_000'),   # 5M USDC
        token_b_reserves=Decimal('1666'),        # 1666 WETH = $3003/ETH  
        fee_rate=Decimal('0.003')
    )
    
    pool_2 = AMMPoolState(
        protocol=DEXProtocol.CURVE_STABLE,
        token_a_reserves=Decimal('4_000_000'),   # 4M USDC
        token_b_reserves=Decimal('1335'),        # 1335 WETH = $2996.25/ETH
        fee_rate=Decimal('0.0004'),
        amplification_coefficient=Decimal('100')
    )
    
    params = AMMArbitrageParameters(
        pool_1=pool_1,
        pool_2=pool_2,
        flash_loan_fee_rate=Decimal('0.0009'),
        gas_cost_usd=Decimal('50'),
        mev_bribe_rate=Decimal('0.001')
    )
    
    optimizer = AMMPriceOptimizer()
    result = optimizer.optimize_arbitrage_amount(params)
    
    initial_diff = abs(pool_1.price_a_to_b - pool_2.price_a_to_b) / max(pool_1.price_a_to_b, pool_2.price_a_to_b)
    
    print(f"\n📊 Exhaustion Analysis:")
    print(f"   Initial Price Difference: {initial_diff * 100:.4f}%")
    print(f"   Optimal Flash Loan: ${result.optimal_amount:,.0f}")
    print(f"   Price Convergence: {result.price_convergence_percentage:.1f}%")
    print(f"   Arbitrage Exhausted: {'✅ Yes' if result.arbitrage_exhausted else '❌ No'}")
    print(f"   Remaining Opportunity: {result.final_price_difference * 100:.4f}%")
    
    if result.arbitrage_exhausted:
        print(f"\n✅ Optimal Point Found:")
        print(f"   • Captured {result.price_convergence_percentage:.1f}% of price difference")
        print(f"   • Further trading would be unprofitable due to slippage + fees")
        print(f"   • This is the true economic optimum")
    else:
        print(f"\n⚠️ Arbitrage Not Exhausted:")
        print(f"   • Only captured {result.price_convergence_percentage:.1f}% of price difference")
        print(f"   • Constraint hit: liquidity, gas costs, or flash loan limits")
    
    print()


def main():
    """Run all AMM price reversion demonstrations."""
    print("🎯 AMM Price Reversion vs Simplified Models")
    print("=" * 60)
    print()
    
    try:
        demo_price_reversion_mechanics()
        demo_price_impact_analysis()
        demo_simplified_vs_true_comparison()
        demo_exhausted_arbitrage()
        
        print("✅ All AMM price reversion demonstrations completed!")
        print()
        print("💡 Key Insights:")
        print("   • True AMM mechanics show actual price convergence")
        print("   • Optimal point is where marginal profit = marginal cost")
        print("   • Price reversion limits arbitrage opportunity")
        print("   • Simplified models may significantly over/under-estimate")
        print("   • Constant product formula shows diminishing returns")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()