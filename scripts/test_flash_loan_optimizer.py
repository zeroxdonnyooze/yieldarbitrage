#!/usr/bin/env python3
"""
Demonstration script for Flash Loan Amount Optimization.

This script shows how to calculate the optimal flash loan amount for arbitrage
opportunities considering slippage, fees, gas costs, and MEV protection.
"""
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.flash_loan_optimizer import (
    FlashLoanOptimizer, ArbitrageParameters, OptimizationResult
)
from yield_arbitrage.protocols.flash_loan_provider import FlashLoanTerms, FlashLoanProvider


def create_sample_flash_loan_terms(provider: FlashLoanProvider = FlashLoanProvider.AAVE_V3) -> FlashLoanTerms:
    """Create sample flash loan terms for demonstration."""
    if provider == FlashLoanProvider.AAVE_V3:
        return FlashLoanTerms(
            provider=provider,
            asset="USDC",
            max_amount=Decimal('10_000_000'),  # $10M max
            fee_rate=Decimal('0.0009'),        # 0.09% fee
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=150_000
        )
    elif provider == FlashLoanProvider.BALANCER:
        return FlashLoanTerms(
            provider=provider,
            asset="USDC", 
            max_amount=Decimal('5_000_000'),   # $5M max
            fee_rate=Decimal('0'),             # No fee
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=200_000
        )
    else:  # UNISWAP_V3
        return FlashLoanTerms(
            provider=provider,
            asset="USDC",
            max_amount=Decimal('3_000_000'),   # $3M max
            fee_rate=Decimal('0.0001'),        # Pool fees only
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=100_000
        )


def demo_basic_optimization():
    """Demonstrate basic flash loan amount optimization."""
    print("🎯 Basic Flash Loan Amount Optimization:")
    print("   USDC arbitrage between two DEXs with 0.5% price difference")
    
    optimizer = FlashLoanOptimizer()
    
    # Create arbitrage parameters
    params = ArbitrageParameters(
        price_difference=Decimal('0.005'),      # 0.5% price difference
        liquidity_pool_1=Decimal('2_000_000'),  # $2M liquidity in pool 1
        liquidity_pool_2=Decimal('1_500_000'),  # $1.5M liquidity in pool 2
        flash_loan_terms=create_sample_flash_loan_terms(FlashLoanProvider.AAVE_V3),
        gas_cost_usd=Decimal('25'),             # $25 gas cost
        mev_bribe_rate=Decimal('0.001'),        # 0.1% MEV bribe
        slippage_factor_1=Decimal('0.00005'),   # Low slippage pool
        slippage_factor_2=Decimal('0.0001'),    # Higher slippage pool
    )
    
    result = optimizer.optimize_flash_loan_amount(params)
    
    print(f"\n📊 Optimization Results:")
    print(f"   Optimal Flash Loan Amount: ${result.optimal_amount:,.2f}")
    print(f"   Expected Net Profit: ${result.expected_profit:.2f}")
    print(f"   Gross Profit: ${result.gross_profit:.2f}")
    print(f"   Total Costs: ${result.total_costs:.2f}")
    print(f"   Profit Margin: {result.profit_margin:.3f}%")
    print(f"   ROI vs Flash Loan Cost: {result.roi:.1f}%")
    
    print(f"\n💰 Cost Breakdown:")
    print(f"   Flash Loan Fee: ${result.flash_loan_fee:.2f}")
    print(f"   Gas Cost: ${result.gas_cost:.2f}")
    print(f"   MEV Bribe: ${result.mev_bribe:.2f}")
    print(f"   Slippage Cost: ${result.slippage_cost:.2f}")
    
    print(f"\n✅ Status:")
    print(f"   Profitable: {'✅ Yes' if result.is_profitable else '❌ No'}")
    print(f"   Max Amount Reached: {'⚠️ Yes' if result.max_amount_reached else '✅ No'}")
    print(f"   Liquidity Constrained: {'⚠️ Yes' if result.liquidity_constrained else '✅ No'}")
    
    print()


def demo_provider_comparison():
    """Compare optimization across different flash loan providers."""
    print("🔍 Flash Loan Provider Comparison:")
    print("   Same arbitrage opportunity across different providers")
    
    optimizer = FlashLoanOptimizer()
    
    # Base arbitrage parameters
    base_params = {
        'price_difference': Decimal('0.003'),     # 0.3% price difference
        'liquidity_pool_1': Decimal('1_000_000'), # $1M liquidity
        'liquidity_pool_2': Decimal('800_000'),   # $800K liquidity
        'gas_cost_usd': Decimal('30'),            # $30 gas cost
        'mev_bribe_rate': Decimal('0.0015'),      # 0.15% MEV bribe
        'slippage_factor_1': Decimal('0.0001'),
        'slippage_factor_2': Decimal('0.00015'),
    }
    
    providers = [FlashLoanProvider.AAVE_V3, FlashLoanProvider.BALANCER, FlashLoanProvider.UNISWAP_V3]
    
    print(f"\n📈 Provider Comparison Results:")
    
    best_result = None
    best_provider = None
    
    for provider in providers:
        params = ArbitrageParameters(
            **base_params,
            flash_loan_terms=create_sample_flash_loan_terms(provider)
        )
        
        result = optimizer.optimize_flash_loan_amount(params)
        
        print(f"\n   💳 {provider.value.upper()}:")
        print(f"      Optimal Amount: ${result.optimal_amount:,.2f}")
        print(f"      Net Profit: ${result.expected_profit:.2f}")
        print(f"      Flash Loan Fee: ${result.flash_loan_fee:.2f}")
        print(f"      Profit Margin: {result.profit_margin:.3f}%")
        print(f"      Profitable: {'✅' if result.is_profitable else '❌'}")
        
        if best_result is None or result.expected_profit > best_result.expected_profit:
            best_result = result
            best_provider = provider
    
    print(f"\n🏆 Best Provider: {best_provider.value.upper()}")
    print(f"   Superior by: ${best_result.expected_profit:.2f} profit")
    
    print()


def demo_sensitivity_analysis():
    """Demonstrate sensitivity analysis for key parameters."""
    print("📊 Sensitivity Analysis:")
    print("   How profit changes with key parameters")
    
    optimizer = FlashLoanOptimizer()
    
    # Base case
    base_params = ArbitrageParameters(
        price_difference=Decimal('0.004'),       # 0.4% base price difference
        liquidity_pool_1=Decimal('1_500_000'),
        liquidity_pool_2=Decimal('1_200_000'),
        flash_loan_terms=create_sample_flash_loan_terms(FlashLoanProvider.BALANCER),  # No fees
        gas_cost_usd=Decimal('20'),
        mev_bribe_rate=Decimal('0.001'),
        slippage_factor_1=Decimal('0.00008'),
        slippage_factor_2=Decimal('0.0001'),
    )
    
    base_result = optimizer.optimize_flash_loan_amount(base_params)
    base_amount = base_result.optimal_amount
    
    print(f"\n🎯 Base Case:")
    print(f"   Optimal Amount: ${base_amount:,.2f}")
    print(f"   Net Profit: ${base_result.expected_profit:.2f}")
    
    # Sensitivity variations
    variations = {
        "price_difference": [Decimal('0.002'), Decimal('0.003'), Decimal('0.004'), Decimal('0.005'), Decimal('0.006')],
        "gas_cost_usd": [Decimal('10'), Decimal('20'), Decimal('30'), Decimal('50'), Decimal('100')],
        "mev_bribe_rate": [Decimal('0.0005'), Decimal('0.001'), Decimal('0.0015'), Decimal('0.002'), Decimal('0.003')],
    }
    
    sensitivity_results = optimizer.analyze_sensitivity(base_params, base_amount, variations)
    
    for param_name, results in sensitivity_results.items():
        print(f"\n📈 {param_name.replace('_', ' ').title()} Sensitivity:")
        
        for value, profit in results:
            change_pct = ((profit - base_result.expected_profit) / base_result.expected_profit * 100) if base_result.expected_profit > 0 else 0
            print(f"   {value}: ${profit:.2f} ({change_pct:+.1f}% vs base)")
    
    print()


def demo_breakeven_analysis():
    """Demonstrate breakeven analysis."""
    print("⚖️ Breakeven Analysis:")
    print("   Finding minimum viable conditions")
    
    optimizer = FlashLoanOptimizer()
    
    # Parameters for breakeven analysis
    params = ArbitrageParameters(
        price_difference=Decimal('0.002'),       # Small 0.2% price difference
        liquidity_pool_1=Decimal('500_000'),     # Limited liquidity
        liquidity_pool_2=Decimal('400_000'),
        flash_loan_terms=create_sample_flash_loan_terms(FlashLoanProvider.AAVE_V3),
        gas_cost_usd=Decimal('40'),              # High gas cost
        mev_bribe_rate=Decimal('0.002'),         # High MEV bribe
        slippage_factor_1=Decimal('0.0002'),     # High slippage
        slippage_factor_2=Decimal('0.0003'),
    )
    
    result = optimizer.optimize_flash_loan_amount(params)
    breakeven = optimizer.calculate_breakeven_amount(params)
    max_profitable = optimizer.estimate_max_profitable_amount(params)
    
    print(f"\n📊 Breakeven Analysis Results:")
    print(f"   Optimal Amount: ${result.optimal_amount:,.2f}")
    print(f"   Optimal Profit: ${result.expected_profit:.2f}")
    print(f"   Breakeven Amount: ${breakeven:,.2f}" if breakeven else "   Breakeven Amount: No breakeven found")
    print(f"   Max Profitable Amount: ${max_profitable:,.2f}")
    
    if result.is_profitable:
        print(f"   Status: ✅ Profitable opportunity")
        print(f"   Profit Range: ${breakeven:,.2f} - ${max_profitable:,.2f}")
    else:
        print(f"   Status: ❌ Not profitable under these conditions")
        
        # Suggest improvements
        print(f"\n💡 Suggested Improvements:")
        if result.total_costs > result.gross_profit:
            print(f"   • Reduce costs (current: ${result.total_costs:.2f} vs profit: ${result.gross_profit:.2f})")
        if params.price_difference < Decimal('0.005'):
            print(f"   • Find opportunities with larger price differences (current: {params.price_difference * 100:.2f}%)")
        if params.liquidity_pool_1 < Decimal('1_000_000'):
            print(f"   • Target higher liquidity pools (current: ${params.liquidity_pool_1:,.0f})")
    
    print()


def demo_real_world_scenario():
    """Demonstrate optimization for a realistic scenario."""
    print("🌍 Real-World Scenario Analysis:")
    print("   ETH/USDC arbitrage: Uniswap vs Curve with current market conditions")
    
    optimizer = FlashLoanOptimizer()
    
    # Realistic market parameters
    params = ArbitrageParameters(
        price_difference=Decimal('0.0025'),      # 0.25% realistic arbitrage
        liquidity_pool_1=Decimal('50_000_000'),  # Large Uniswap pool
        liquidity_pool_2=Decimal('30_000_000'),  # Large Curve pool
        flash_loan_terms=create_sample_flash_loan_terms(FlashLoanProvider.BALANCER),  # Use Balancer (no fees)
        gas_cost_usd=Decimal('45'),              # Current high gas costs
        mev_bribe_rate=Decimal('0.0008'),        # Conservative MEV protection
        slippage_factor_1=Decimal('0.00002'),    # Low slippage (large pool)
        slippage_factor_2=Decimal('0.00003'),    # Slightly higher slippage
        min_profit_usd=Decimal('100'),           # $100 minimum profit
    )
    
    result = optimizer.optimize_flash_loan_amount(params)
    
    print(f"\n💰 Real-World Optimization Results:")
    print(f"   Scenario: ETH/USDC arbitrage with {params.price_difference * 100:.2f}% price difference")
    print(f"   Pool Liquidity: ${params.liquidity_pool_1:,.0f} (Uniswap) vs ${params.liquidity_pool_2:,.0f} (Curve)")
    
    print(f"\n🎯 Optimal Strategy:")
    print(f"   Flash Loan Amount: ${result.optimal_amount:,.2f}")
    print(f"   Expected Profit: ${result.expected_profit:.2f}")
    print(f"   Profit Margin: {result.profit_margin:.4f}%")
    print(f"   Cost Efficiency: {result.cost_efficiency:.2f}x")
    
    print(f"\n📊 Economics:")
    print(f"   Revenue: ${result.gross_profit:.2f}")
    print(f"   Costs: ${result.total_costs:.2f}")
    print(f"     • Gas: ${result.gas_cost:.2f}")
    print(f"     • MEV Bribe: ${result.mev_bribe:.2f}")
    print(f"     • Slippage: ${result.slippage_cost:.2f}")
    print(f"     • Flash Loan Fee: ${result.flash_loan_fee:.2f}")
    
    # Calculate some additional metrics
    volume_ratio = result.optimal_amount / params.liquidity_pool_2  # Use smaller pool
    gas_per_dollar = result.gas_cost / result.optimal_amount
    
    print(f"\n📈 Market Impact:")
    print(f"   Trade Volume vs Pool Size: {volume_ratio * 100:.3f}%")
    print(f"   Gas Cost per Dollar Traded: ${gas_per_dollar * 1000:.2f} per $1000")
    print(f"   Effective APR (if daily): {result.profit_margin * 365:.1f}%")
    
    if result.is_profitable:
        print(f"\n✅ Recommendation: Execute with ${result.optimal_amount:,.2f} flash loan")
        if result.liquidity_constrained:
            print(f"   ⚠️  Note: Constrained by liquidity - larger pools could increase profit")
        if result.max_amount_reached:
            print(f"   ⚠️  Note: Reached flash loan limit - higher capacity could increase profit")
    else:
        print(f"\n❌ Recommendation: Skip this opportunity")
        print(f"   Reason: Insufficient profit margin under current conditions")
    
    print()


def main():
    """Run all flash loan optimization demonstrations."""
    print("🎯 Flash Loan Amount Optimization Demonstrations")
    print("=" * 60)
    print()
    
    try:
        demo_basic_optimization()
        demo_provider_comparison()
        demo_sensitivity_analysis()
        demo_breakeven_analysis()
        demo_real_world_scenario()
        
        print("✅ All flash loan optimization demonstrations completed!")
        print()
        print("💡 Key Insights:")
        print("   • Optimal amount balances profit opportunity vs costs/slippage")
        print("   • Provider choice significantly impacts profitability")
        print("   • Sensitivity analysis reveals key risk factors")
        print("   • Breakeven analysis identifies minimum viable conditions")
        print("   • Real-world scenarios require careful cost/benefit analysis")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()