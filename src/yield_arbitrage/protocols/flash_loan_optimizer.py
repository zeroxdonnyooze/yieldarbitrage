"""
Flash Loan Amount Optimization for Arbitrage Strategies.

This module calculates the optimal flash loan amount for arbitrage opportunities
considering slippage, flash loan fees, gas costs, and MEV protection costs.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from decimal import Decimal
import math

from yield_arbitrage.protocols.flash_loan_provider import FlashLoanTerms

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageParameters:
    """Parameters for an arbitrage opportunity."""
    # Market parameters
    price_difference: Decimal  # Initial price difference (e.g., 0.01 = 1%)
    liquidity_pool_1: Decimal  # Liquidity in first pool
    liquidity_pool_2: Decimal  # Liquidity in second pool
    
    # Cost parameters
    flash_loan_terms: FlashLoanTerms
    gas_cost_usd: Decimal
    mev_bribe_rate: Decimal = Decimal('0.001')  # MEV bribe as % of trade size
    
    # Slippage parameters (simplified linear model)
    slippage_factor_1: Decimal = Decimal('0.0001')  # Slippage per dollar traded
    slippage_factor_2: Decimal = Decimal('0.0001')
    
    # Safety margins
    min_profit_usd: Decimal = Decimal('50')  # Minimum profit threshold
    max_amount_utilization: Decimal = Decimal('0.8')  # Max 80% of flash loan capacity


@dataclass
class OptimizationResult:
    """Result of flash loan amount optimization."""
    optimal_amount: Decimal
    expected_profit: Decimal
    total_costs: Decimal
    gross_profit: Decimal
    
    # Cost breakdown
    flash_loan_fee: Decimal
    gas_cost: Decimal
    mev_bribe: Decimal
    slippage_cost: Decimal
    
    # Efficiency metrics
    profit_margin: Decimal  # Profit as % of loan amount
    cost_efficiency: Decimal  # Profit / Total costs ratio
    roi: Decimal  # Return on investment considering flash loan as capital
    
    # Validity
    is_profitable: bool
    max_amount_reached: bool
    liquidity_constrained: bool


class FlashLoanOptimizer:
    """
    Optimizes flash loan amounts for arbitrage opportunities.
    
    Uses numerical optimization to find the amount that maximizes profit
    after accounting for all costs and slippage effects.
    """
    
    def __init__(self):
        """Initialize the optimizer."""
        self.precision = Decimal('0.01')  # $0.01 precision
        self.max_iterations = 1000
        
    def optimize_flash_loan_amount(self, params: ArbitrageParameters) -> OptimizationResult:
        """
        Find optimal flash loan amount for an arbitrage opportunity.
        
        Args:
            params: Arbitrage parameters including costs and slippage
            
        Returns:
            Optimization result with optimal amount and profit breakdown
        """
        logger.info("Optimizing flash loan amount for arbitrage opportunity")
        
        # Define bounds
        min_amount = Decimal('100')  # $100 minimum
        max_amount = min(
            params.flash_loan_terms.max_amount * params.max_amount_utilization,
            min(params.liquidity_pool_1, params.liquidity_pool_2) * Decimal('0.5')  # 50% of min liquidity
        )
        
        if min_amount >= max_amount:
            return self._create_infeasible_result("Insufficient liquidity or flash loan capacity")
        
        # Use golden section search for optimization
        optimal_amount = self._golden_section_search(
            params, min_amount, max_amount, self._profit_function
        )
        
        # Calculate final result
        result = self._calculate_profit_breakdown(params, optimal_amount)
        
        logger.info(f"Optimization complete: ${optimal_amount:,.2f} loan for ${result.expected_profit:.2f} profit")
        
        return result
    
    def _profit_function(self, amount: Decimal, params: ArbitrageParameters) -> Decimal:
        """
        Calculate net profit for a given flash loan amount.
        
        Args:
            amount: Flash loan amount
            params: Arbitrage parameters
            
        Returns:
            Net profit (can be negative)
        """
        if amount <= 0:
            return Decimal('-999999')  # Invalid amount
        
        # Calculate gross profit from arbitrage
        gross_profit = self._calculate_gross_profit(amount, params)
        
        # Calculate all costs
        flash_loan_fee = amount * params.flash_loan_terms.fee_rate + params.flash_loan_terms.fixed_fee
        gas_cost = params.gas_cost_usd
        mev_bribe = amount * params.mev_bribe_rate
        slippage_cost = self._calculate_slippage_cost(amount, params)
        
        total_costs = flash_loan_fee + gas_cost + mev_bribe + slippage_cost
        
        return gross_profit - total_costs
    
    def _calculate_gross_profit(self, amount: Decimal, params: ArbitrageParameters) -> Decimal:
        """Calculate gross profit before costs."""
        # Simplified model: profit = amount * price_difference * (1 - slippage_impact)
        base_profit = amount * params.price_difference
        
        # Reduce profit due to slippage impact
        slippage_impact = self._calculate_slippage_impact(amount, params)
        effective_profit = base_profit * (Decimal('1') - slippage_impact)
        
        return max(Decimal('0'), effective_profit)
    
    def _calculate_slippage_impact(self, amount: Decimal, params: ArbitrageParameters) -> Decimal:
        """Calculate total slippage impact as a fraction."""
        # Slippage impact grows with trade size relative to liquidity
        impact_1 = (amount / params.liquidity_pool_1) * params.slippage_factor_1
        impact_2 = (amount / params.liquidity_pool_2) * params.slippage_factor_2
        
        # Total impact is sum of both trades
        total_impact = impact_1 + impact_2
        
        # Cap at 99% (can't lose more than the trade)
        return min(total_impact, Decimal('0.99'))
    
    def _calculate_slippage_cost(self, amount: Decimal, params: ArbitrageParameters) -> Decimal:
        """Calculate dollar cost of slippage."""
        slippage_impact = self._calculate_slippage_impact(amount, params)
        return amount * slippage_impact
    
    def _golden_section_search(self, params: ArbitrageParameters, 
                             min_val: Decimal, max_val: Decimal,
                             func: Callable[[Decimal, ArbitrageParameters], Decimal]) -> Decimal:
        """
        Golden section search for maximum profit.
        
        Args:
            params: Arbitrage parameters
            min_val: Minimum search bound
            max_val: Maximum search bound
            func: Function to maximize
            
        Returns:
            Optimal amount
        """
        phi = Decimal('1.618033988749')  # Golden ratio
        resphi = 2 - phi
        
        # Initial points
        x1 = min_val + resphi * (max_val - min_val)
        x2 = max_val - resphi * (max_val - min_val)
        f1 = func(x1, params)
        f2 = func(x2, params)
        
        for _ in range(self.max_iterations):
            if abs(max_val - min_val) < self.precision:
                break
                
            if f1 > f2:
                max_val = x2
                x2 = x1
                f2 = f1
                x1 = min_val + resphi * (max_val - min_val)
                f1 = func(x1, params)
            else:
                min_val = x1
                x1 = x2
                f1 = f2
                x2 = max_val - resphi * (max_val - min_val)
                f2 = func(x2, params)
        
        return (min_val + max_val) / 2
    
    def _calculate_profit_breakdown(self, params: ArbitrageParameters, 
                                  amount: Decimal) -> OptimizationResult:
        """Calculate detailed profit breakdown for given amount."""
        # Calculate all components
        gross_profit = self._calculate_gross_profit(amount, params)
        flash_loan_fee = amount * params.flash_loan_terms.fee_rate + params.flash_loan_terms.fixed_fee
        gas_cost = params.gas_cost_usd
        mev_bribe = amount * params.mev_bribe_rate
        slippage_cost = self._calculate_slippage_cost(amount, params)
        
        total_costs = flash_loan_fee + gas_cost + mev_bribe + slippage_cost
        net_profit = gross_profit - total_costs
        
        # Calculate metrics
        profit_margin = (net_profit / amount) * 100 if amount > 0 else Decimal('0')
        cost_efficiency = net_profit / total_costs if total_costs > 0 else Decimal('0')
        roi = (net_profit / flash_loan_fee) * 100 if flash_loan_fee > 0 else Decimal('0')
        
        # Check constraints
        max_capacity = params.flash_loan_terms.max_amount * params.max_amount_utilization
        max_liquidity = min(params.liquidity_pool_1, params.liquidity_pool_2) * Decimal('0.5')
        
        return OptimizationResult(
            optimal_amount=amount,
            expected_profit=net_profit,
            total_costs=total_costs,
            gross_profit=gross_profit,
            flash_loan_fee=flash_loan_fee,
            gas_cost=gas_cost,
            mev_bribe=mev_bribe,
            slippage_cost=slippage_cost,
            profit_margin=profit_margin,
            cost_efficiency=cost_efficiency,
            roi=roi,
            is_profitable=net_profit >= params.min_profit_usd,
            max_amount_reached=amount >= max_capacity * Decimal('0.99'),
            liquidity_constrained=amount >= max_liquidity * Decimal('0.99')
        )
    
    def _create_infeasible_result(self, reason: str) -> OptimizationResult:
        """Create a result for infeasible optimization."""
        logger.warning(f"Flash loan optimization infeasible: {reason}")
        
        return OptimizationResult(
            optimal_amount=Decimal('0'),
            expected_profit=Decimal('0'),
            total_costs=Decimal('0'),
            gross_profit=Decimal('0'),
            flash_loan_fee=Decimal('0'),
            gas_cost=Decimal('0'),
            mev_bribe=Decimal('0'),
            slippage_cost=Decimal('0'),
            profit_margin=Decimal('0'),
            cost_efficiency=Decimal('0'),
            roi=Decimal('0'),
            is_profitable=False,
            max_amount_reached=False,
            liquidity_constrained=True
        )
    
    def analyze_sensitivity(self, params: ArbitrageParameters, 
                          base_amount: Decimal,
                          variations: Dict[str, List[Decimal]]) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """
        Perform sensitivity analysis on optimization parameters.
        
        Args:
            params: Base arbitrage parameters
            base_amount: Base flash loan amount
            variations: Parameter variations to test
            
        Returns:
            Sensitivity analysis results
        """
        results = {}
        
        for param_name, values in variations.items():
            param_results = []
            
            for value in values:
                # Create modified parameters
                modified_params = self._modify_params(params, param_name, value)
                
                # Calculate profit for this variation
                profit = self._profit_function(base_amount, modified_params)
                param_results.append((value, profit))
            
            results[param_name] = param_results
        
        return results
    
    def _modify_params(self, params: ArbitrageParameters, 
                      param_name: str, value: Decimal) -> ArbitrageParameters:
        """Create modified parameters for sensitivity analysis."""
        # Create a copy of params with modified value
        new_params = ArbitrageParameters(
            price_difference=params.price_difference,
            liquidity_pool_1=params.liquidity_pool_1,
            liquidity_pool_2=params.liquidity_pool_2,
            flash_loan_terms=params.flash_loan_terms,
            gas_cost_usd=params.gas_cost_usd,
            mev_bribe_rate=params.mev_bribe_rate,
            slippage_factor_1=params.slippage_factor_1,
            slippage_factor_2=params.slippage_factor_2,
            min_profit_usd=params.min_profit_usd,
            max_amount_utilization=params.max_amount_utilization
        )
        
        # Modify the specified parameter
        if param_name == "price_difference":
            new_params.price_difference = value
        elif param_name == "gas_cost_usd":
            new_params.gas_cost_usd = value
        elif param_name == "mev_bribe_rate":
            new_params.mev_bribe_rate = value
        elif param_name == "slippage_factor_1":
            new_params.slippage_factor_1 = value
        elif param_name == "slippage_factor_2":
            new_params.slippage_factor_2 = value
        
        return new_params
    
    def calculate_breakeven_amount(self, params: ArbitrageParameters) -> Optional[Decimal]:
        """
        Calculate the breakeven flash loan amount (zero profit).
        
        Args:
            params: Arbitrage parameters
            
        Returns:
            Breakeven amount or None if no breakeven exists
        """
        # Use binary search to find breakeven point
        min_amount = Decimal('1')
        max_amount = params.flash_loan_terms.max_amount
        
        for _ in range(100):  # Max iterations
            mid_amount = (min_amount + max_amount) / 2
            profit = self._profit_function(mid_amount, params)
            
            if abs(profit) < Decimal('0.01'):  # Close enough to zero
                return mid_amount
            elif profit > 0:
                min_amount = mid_amount
            else:
                max_amount = mid_amount
            
            if max_amount - min_amount < Decimal('0.01'):
                break
        
        return None
    
    def estimate_max_profitable_amount(self, params: ArbitrageParameters) -> Decimal:
        """
        Estimate maximum amount that remains profitable.
        
        This is useful for understanding the upper bounds of the arbitrage.
        """
        # Start from optimal and increase until profit becomes negative
        optimal_result = self.optimize_flash_loan_amount(params)
        
        if not optimal_result.is_profitable:
            return Decimal('0')
        
        # Search upward from optimal
        amount = optimal_result.optimal_amount
        step = optimal_result.optimal_amount * Decimal('0.1')  # 10% steps
        
        while amount <= params.flash_loan_terms.max_amount:
            profit = self._profit_function(amount, params)
            if profit <= 0:
                return amount - step
            amount += step
        
        return params.flash_loan_terms.max_amount