"""
True AMM Price Reversion Flash Loan Optimizer.

This module implements proper AMM mechanics to calculate optimal flash loan amounts
by modeling how swap mechanics cause price reversion and arbitrage opportunity shrinkage.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Union
from decimal import Decimal
import math
from enum import Enum

logger = logging.getLogger(__name__)

class DEXProtocol(Enum):
    """Supported DEX protocols with different AMM formulas."""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    CURVE_STABLE = "curve_stable"
    CURVE_CRYPTO = "curve_crypto"
    BALANCER_WEIGHTED = "balancer_weighted"
    BALANCER_STABLE = "balancer_stable"
    BANCOR_V3 = "bancor_v3"


@dataclass
class AMMPoolState:
    """State of an AMM pool for price calculation."""
    protocol: DEXProtocol
    token_a_reserves: Decimal  # Token A reserves  
    token_b_reserves: Decimal  # Token B reserves
    fee_rate: Decimal = Decimal('0.003')  # Default fee rate
    
    # Uniswap V3 specific
    current_tick: Optional[int] = None
    sqrt_price_x96: Optional[int] = None
    liquidity: Optional[Decimal] = None
    tick_lower: Optional[int] = None
    tick_upper: Optional[int] = None
    
    # Curve specific
    amplification_coefficient: Optional[Decimal] = None  # A parameter for Curve
    
    # Balancer specific
    token_weights: Optional[List[Decimal]] = None  # [weight_a, weight_b] for weighted pools
    
    # Multi-asset pools
    additional_reserves: Optional[List[Decimal]] = None  # For 3+ token pools
    additional_weights: Optional[List[Decimal]] = None
    
    @property
    def price_a_to_b(self) -> Decimal:
        """Current price of token A in terms of token B (protocol-aware)."""
        if self.protocol == DEXProtocol.UNISWAP_V3:
            return self._calculate_v3_price()
        elif self.protocol in [DEXProtocol.CURVE_STABLE, DEXProtocol.CURVE_CRYPTO]:
            return self._calculate_curve_price()
        elif self.protocol in [DEXProtocol.BALANCER_WEIGHTED, DEXProtocol.BALANCER_STABLE]:
            return self._calculate_balancer_price()
        else:  # Default to constant product
            return self.token_b_reserves / self.token_a_reserves if self.token_a_reserves > 0 else Decimal('0')
    
    def _calculate_v3_price(self) -> Decimal:
        """Calculate Uniswap V3 price from current tick."""
        if self.current_tick is not None:
            # Price = 1.0001^tick
            return Decimal('1.0001') ** self.current_tick
        # Fallback to reserves ratio
        return self.token_b_reserves / self.token_a_reserves if self.token_a_reserves > 0 else Decimal('0')
    
    def _calculate_curve_price(self) -> Decimal:
        """Calculate Curve price (simplified - actual formula is more complex)."""
        # For demonstration - actual Curve pricing is more sophisticated
        return self.token_b_reserves / self.token_a_reserves if self.token_a_reserves > 0 else Decimal('0')
    
    def _calculate_balancer_price(self) -> Decimal:
        """Calculate Balancer weighted pool price."""
        if self.token_weights and len(self.token_weights) >= 2:
            # Price = (reserve_b / weight_b) / (reserve_a / weight_a)
            weight_a, weight_b = self.token_weights[0], self.token_weights[1]
            if weight_a > 0 and weight_b > 0 and self.token_a_reserves > 0:
                return (self.token_b_reserves / weight_b) / (self.token_a_reserves / weight_a)
        return self.token_b_reserves / self.token_a_reserves if self.token_a_reserves > 0 else Decimal('0')
    
    @property
    def k_constant(self) -> Decimal:
        """Invariant constant (protocol-specific)."""
        if self.protocol == DEXProtocol.UNISWAP_V2:
            return self.token_a_reserves * self.token_b_reserves
        elif self.protocol == DEXProtocol.BALANCER_WEIGHTED and self.token_weights:
            # Balancer invariant: ∏(reserve_i^weight_i)
            weight_a, weight_b = self.token_weights[0], self.token_weights[1]
            return (self.token_a_reserves ** weight_a) * (self.token_b_reserves ** weight_b)
        else:
            # Default fallback
            return self.token_a_reserves * self.token_b_reserves


@dataclass
class AMMArbitrageParameters:
    """Parameters for AMM arbitrage with true price reversion."""
    # Pool states
    pool_1: AMMPoolState  # First DEX (e.g., Uniswap)
    pool_2: AMMPoolState  # Second DEX (e.g., Curve)
    
    # Flash loan parameters  
    flash_loan_fee_rate: Decimal = Decimal('0.0009')  # 0.09%
    gas_cost_usd: Decimal = Decimal('25')
    mev_bribe_rate: Decimal = Decimal('0.001')  # 0.1% of trade size
    
    # Safety margins
    min_profit_usd: Decimal = Decimal('50')
    max_slippage_tolerance: Decimal = Decimal('0.02')  # 2% max slippage


@dataclass 
class AMMOptimizationResult:
    """Result of true AMM arbitrage optimization."""
    optimal_amount: Decimal
    expected_profit: Decimal
    
    # Price analysis
    initial_price_difference: Decimal
    final_price_difference: Decimal
    price_convergence_percentage: Decimal
    
    # Trade execution details
    amount_in_pool_1: Decimal
    amount_out_pool_1: Decimal
    amount_in_pool_2: Decimal
    amount_out_pool_2: Decimal
    
    # Slippage analysis
    slippage_pool_1: Decimal
    slippage_pool_2: Decimal
    total_slippage: Decimal
    
    # Cost breakdown
    flash_loan_fee: Decimal
    gas_cost: Decimal
    mev_bribe: Decimal
    total_costs: Decimal
    
    # Final pool states
    pool_1_final: AMMPoolState
    pool_2_final: AMMPoolState
    
    # Validation
    is_profitable: bool
    arbitrage_exhausted: bool  # True if we've taken most of the opportunity


class AMMPriceOptimizer:
    """
    Optimizes flash loan amounts using true AMM price reversion mechanics.
    
    Models how swap mechanics cause prices to converge and finds the optimal
    amount that maximizes profit before arbitrage opportunity is exhausted.
    """
    
    def __init__(self):
        self.precision = Decimal('0.01')  # $0.01 precision
        self.max_iterations = 1000
    
    def optimize_arbitrage_amount(self, params: AMMArbitrageParameters) -> AMMOptimizationResult:
        """
        Find optimal arbitrage amount using true AMM price reversion.
        
        Models the constant product formula to show how prices converge
        as trade size increases, finding the optimal point.
        """
        logger.info("Optimizing arbitrage with true AMM price reversion modeling")
        
        # Determine arbitrage direction
        price_1 = params.pool_1.price_a_to_b
        price_2 = params.pool_2.price_a_to_b
        
        if price_1 > price_2:
            # Buy on pool_2 (cheaper), sell on pool_1 (expensive)
            buy_pool = params.pool_2
            sell_pool = params.pool_1
            direction = "pool2_to_pool1"
        else:
            # Buy on pool_1 (cheaper), sell on pool_2 (expensive)
            buy_pool = params.pool_1
            sell_pool = params.pool_2
            direction = "pool1_to_pool2"
        
        initial_price_diff = abs(price_1 - price_2) / max(price_1, price_2)
        logger.info(f"Initial price difference: {initial_price_diff * 100:.4f}%")
        
        # Find optimal amount using ternary search on profit function
        min_amount = Decimal('100')
        max_amount = min(
            buy_pool.token_a_reserves * Decimal('0.5'),  # Max 50% of pool
            sell_pool.token_b_reserves * Decimal('0.5')
        )
        
        optimal_amount = self._ternary_search(
            params, min_amount, max_amount, direction
        )
        
        # Calculate final result
        result = self._calculate_detailed_result(params, optimal_amount, direction)
        
        logger.info(f"Optimization complete: ${optimal_amount:,.2f} for ${result.expected_profit:.2f} profit")
        
        return result
    
    def _profit_function(self, amount: Decimal, params: AMMArbitrageParameters, direction: str) -> Decimal:
        """
        Calculate profit for given amount using true AMM mechanics.
        
        This models how prices change due to swaps and calculates actual profit.
        """
        if amount <= 0:
            return Decimal('-999999')
        
        try:
            # Simulate the arbitrage trade
            trade_result = self._simulate_arbitrage_trade(amount, params, direction)
            
            if not trade_result:
                return Decimal('-999999')
            
            gross_profit, _, _ = trade_result
            
            # Calculate costs
            flash_loan_fee = amount * params.flash_loan_fee_rate
            gas_cost = params.gas_cost_usd
            mev_bribe = amount * params.mev_bribe_rate
            total_costs = flash_loan_fee + gas_cost + mev_bribe
            
            net_profit = gross_profit - total_costs
            
            return net_profit
            
        except Exception as e:
            logger.debug(f"Error in profit calculation for amount {amount}: {e}")
            return Decimal('-999999')
    
    def _simulate_arbitrage_trade(self, amount: Decimal, params: AMMArbitrageParameters, 
                                direction: str) -> Optional[Tuple[Decimal, AMMPoolState, AMMPoolState]]:
        """
        Simulate complete arbitrage trade using AMM mechanics.
        
        Returns: (gross_profit, final_pool_1_state, final_pool_2_state)
        """
        # Create copies of pool states
        pool_1 = AMMPoolState(
            protocol=params.pool_1.protocol,
            token_a_reserves=params.pool_1.token_a_reserves,
            token_b_reserves=params.pool_1.token_b_reserves,
            fee_rate=params.pool_1.fee_rate,
            current_tick=params.pool_1.current_tick,
            sqrt_price_x96=params.pool_1.sqrt_price_x96,
            liquidity=params.pool_1.liquidity,
            tick_lower=params.pool_1.tick_lower,
            tick_upper=params.pool_1.tick_upper,
            amplification_coefficient=params.pool_1.amplification_coefficient,
            token_weights=params.pool_1.token_weights,
            additional_reserves=params.pool_1.additional_reserves,
            additional_weights=params.pool_1.additional_weights
        )
        pool_2 = AMMPoolState(
            protocol=params.pool_2.protocol,
            token_a_reserves=params.pool_2.token_a_reserves, 
            token_b_reserves=params.pool_2.token_b_reserves,
            fee_rate=params.pool_2.fee_rate,
            current_tick=params.pool_2.current_tick,
            sqrt_price_x96=params.pool_2.sqrt_price_x96,
            liquidity=params.pool_2.liquidity,
            tick_lower=params.pool_2.tick_lower,
            tick_upper=params.pool_2.tick_upper,
            amplification_coefficient=params.pool_2.amplification_coefficient,
            token_weights=params.pool_2.token_weights,
            additional_reserves=params.pool_2.additional_reserves,
            additional_weights=params.pool_2.additional_weights
        )
        
        if direction == "pool1_to_pool2":
            # Buy token B on pool_1, sell on pool_2
            # Step 1: Swap amount (token A) for token B on pool_1
            amount_b_received = self._calculate_swap_output(amount, pool_1, token_in_is_a=True)
            
            if amount_b_received <= 0:
                return None
            
            # Update pool_1 state
            pool_1.token_a_reserves += amount
            pool_1.token_b_reserves -= amount_b_received
            
            # Step 2: Swap token B for token A on pool_2
            amount_a_final = self._calculate_swap_output(amount_b_received, pool_2, token_in_is_a=False)
            
            if amount_a_final <= 0:
                return None
            
            # Update pool_2 state
            pool_2.token_b_reserves += amount_b_received
            pool_2.token_a_reserves -= amount_a_final
            
            # Gross profit = final amount - initial amount
            gross_profit = amount_a_final - amount
            
        else:  # pool2_to_pool1
            # Buy token B on pool_2, sell on pool_1
            # Step 1: Swap amount (token A) for token B on pool_2
            amount_b_received = self._calculate_swap_output(amount, pool_2, token_in_is_a=True)
            
            if amount_b_received <= 0:
                return None
            
            # Update pool_2 state
            pool_2.token_a_reserves += amount
            pool_2.token_b_reserves -= amount_b_received
            
            # Step 2: Swap token B for token A on pool_1
            amount_a_final = self._calculate_swap_output(amount_b_received, pool_1, token_in_is_a=False)
            
            if amount_a_final <= 0:
                return None
            
            # Update pool_1 state
            pool_1.token_b_reserves += amount_b_received
            pool_1.token_a_reserves -= amount_a_final
            
            # Gross profit = final amount - initial amount  
            gross_profit = amount_a_final - amount
        
        return (gross_profit, pool_1, pool_2)
    
    def _calculate_swap_output(self, amount_in: Decimal, pool_state: AMMPoolState, 
                             token_in_is_a: bool = True) -> Decimal:
        """
        Calculate swap output using protocol-specific formulas.
        
        Args:
            amount_in: Amount of input token
            pool_state: Current pool state with protocol info
            token_in_is_a: True if swapping token A for B, False for B to A
            
        Returns:
            Amount of output token
        """
        if amount_in <= 0:
            return Decimal('0')
        
        protocol = pool_state.protocol
        
        if protocol == DEXProtocol.UNISWAP_V2:
            return self._calculate_v2_swap(amount_in, pool_state, token_in_is_a)
        elif protocol == DEXProtocol.UNISWAP_V3:
            return self._calculate_v3_swap(amount_in, pool_state, token_in_is_a)
        elif protocol in [DEXProtocol.CURVE_STABLE, DEXProtocol.CURVE_CRYPTO]:
            return self._calculate_curve_swap(amount_in, pool_state, token_in_is_a)
        elif protocol in [DEXProtocol.BALANCER_WEIGHTED, DEXProtocol.BALANCER_STABLE]:
            return self._calculate_balancer_swap(amount_in, pool_state, token_in_is_a)
        else:
            # Fallback to V2 formula
            return self._calculate_v2_swap(amount_in, pool_state, token_in_is_a)
    
    def _calculate_v2_swap(self, amount_in: Decimal, pool_state: AMMPoolState, 
                          token_in_is_a: bool) -> Decimal:
        """Uniswap V2 constant product formula: x * y = k"""
        if token_in_is_a:
            reserve_in = pool_state.token_a_reserves
            reserve_out = pool_state.token_b_reserves
        else:
            reserve_in = pool_state.token_b_reserves
            reserve_out = pool_state.token_a_reserves
        
        if reserve_in <= 0 or reserve_out <= 0:
            return Decimal('0')
        
        # Apply fee
        amount_in_with_fee = amount_in * (Decimal('1') - pool_state.fee_rate)
        
        # Constant product formula
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        
        if denominator <= 0:
            return Decimal('0')
        
        amount_out = numerator / denominator
        
        # Ensure we don't drain the pool
        if amount_out >= reserve_out * Decimal('0.99'):  # Leave 1% buffer
            return Decimal('0')
        
        return amount_out
    
    def _calculate_v3_swap(self, amount_in: Decimal, pool_state: AMMPoolState, 
                          token_in_is_a: bool) -> Decimal:
        """
        Uniswap V3 concentrated liquidity formula (simplified).
        
        Note: This is a simplified implementation. Real V3 swaps involve:
        - Multiple price ranges with different liquidity
        - Tick-based price calculations
        - Complex liquidity distribution
        """
        if pool_state.liquidity is None or pool_state.current_tick is None:
            # Fallback to V2 if V3 data not available
            return self._calculate_v2_swap(amount_in, pool_state, token_in_is_a)
        
        # Simplified V3 calculation using current tick and liquidity
        # In reality, this would involve iterating through ticks
        current_price = Decimal('1.0001') ** pool_state.current_tick
        sqrt_price = current_price.sqrt()
        
        # Simplified calculation - actual V3 is much more complex
        if token_in_is_a:
            # Approximate amount out using current liquidity and price
            delta_sqrt_price = amount_in / (pool_state.liquidity * sqrt_price)
            new_sqrt_price = sqrt_price + delta_sqrt_price
            amount_out = pool_state.liquidity * (sqrt_price - new_sqrt_price)
        else:
            delta_sqrt_price = (amount_in * sqrt_price) / pool_state.liquidity
            new_sqrt_price = sqrt_price - delta_sqrt_price
            amount_out = pool_state.liquidity * (new_sqrt_price - sqrt_price) / (sqrt_price * new_sqrt_price)
        
        # Apply fee
        amount_out *= (Decimal('1') - pool_state.fee_rate)
        
        return max(Decimal('0'), amount_out)
    
    def _calculate_curve_swap(self, amount_in: Decimal, pool_state: AMMPoolState, 
                            token_in_is_a: bool) -> Decimal:
        """
        Curve StableSwap formula (simplified).
        
        Curve uses: A * n^n * sum(x_i) + D = A * D * n^n + D^(n+1) / (n^n * prod(x_i))
        Where A is amplification coefficient, D is invariant
        """
        if pool_state.amplification_coefficient is None:
            # Fallback to V2 if A parameter not available
            return self._calculate_v2_swap(amount_in, pool_state, token_in_is_a)
        
        A = pool_state.amplification_coefficient
        x = pool_state.token_a_reserves if token_in_is_a else pool_state.token_b_reserves
        y = pool_state.token_b_reserves if token_in_is_a else pool_state.token_a_reserves
        
        # Simplified Curve calculation (actual is iterative)
        # For 2-asset pools: y = (A*n^n*S + D - A*n^n*x - D^(n+1)/(n^n*x)) where S = x + y
        n = Decimal('2')  # 2 assets
        S = x + y
        
        # Calculate new x after adding amount_in
        new_x = x + amount_in
        
        # Approximate new y using simplified Curve formula
        # This is a rough approximation - actual Curve uses Newton's method
        ratio = new_x / x
        slippage_factor = Decimal('1') + (ratio - Decimal('1')) / A
        new_y = y / slippage_factor
        
        amount_out = y - new_y
        
        # Apply fee
        amount_out *= (Decimal('1') - pool_state.fee_rate)
        
        return max(Decimal('0'), amount_out)
    
    def _calculate_balancer_swap(self, amount_in: Decimal, pool_state: AMMPoolState, 
                               token_in_is_a: bool) -> Decimal:
        """
        Balancer weighted pool formula.
        
        Formula: amount_out = balance_out * (1 - (balance_in / (balance_in + amount_in))^(weight_in/weight_out))
        """
        if not pool_state.token_weights or len(pool_state.token_weights) < 2:
            # Fallback to V2 if weights not available
            return self._calculate_v2_swap(amount_in, pool_state, token_in_is_a)
        
        if token_in_is_a:
            balance_in = pool_state.token_a_reserves
            balance_out = pool_state.token_b_reserves
            weight_in = pool_state.token_weights[0]
            weight_out = pool_state.token_weights[1]
        else:
            balance_in = pool_state.token_b_reserves
            balance_out = pool_state.token_a_reserves
            weight_in = pool_state.token_weights[1]
            weight_out = pool_state.token_weights[0]
        
        if balance_in <= 0 or balance_out <= 0 or weight_in <= 0 or weight_out <= 0:
            return Decimal('0')
        
        # Balancer formula
        base = balance_in / (balance_in + amount_in)
        exponent = weight_in / weight_out
        
        try:
            # Convert to float for power calculation, then back to Decimal
            power_result = float(base) ** float(exponent)
            amount_out = balance_out * (Decimal('1') - Decimal(str(power_result)))
        except (OverflowError, ValueError):
            return Decimal('0')
        
        # Apply fee
        amount_out *= (Decimal('1') - pool_state.fee_rate)
        
        return max(Decimal('0'), amount_out)
    
    def _ternary_search(self, params: AMMArbitrageParameters, min_val: Decimal, 
                       max_val: Decimal, direction: str) -> Decimal:
        """
        Ternary search for maximum profit.
        
        More efficient than golden section search for smooth unimodal functions.
        """
        for _ in range(self.max_iterations):
            if abs(max_val - min_val) < self.precision:
                break
            
            # Divide range into thirds
            mid1 = min_val + (max_val - min_val) / 3
            mid2 = max_val - (max_val - min_val) / 3
            
            profit1 = self._profit_function(mid1, params, direction)
            profit2 = self._profit_function(mid2, params, direction)
            
            if profit1 > profit2:
                max_val = mid2
            else:
                min_val = mid1
        
        return (min_val + max_val) / 2
    
    def _calculate_detailed_result(self, params: AMMArbitrageParameters, 
                                 amount: Decimal, direction: str) -> AMMOptimizationResult:
        """Calculate detailed optimization result with all metrics."""
        
        # Get trade simulation
        trade_result = self._simulate_arbitrage_trade(amount, params, direction)
        
        if not trade_result:
            # Return empty result for failed trades
            return AMMOptimizationResult(
                optimal_amount=Decimal('0'),
                expected_profit=Decimal('0'),
                initial_price_difference=Decimal('0'),
                final_price_difference=Decimal('0'),
                price_convergence_percentage=Decimal('0'),
                amount_in_pool_1=Decimal('0'),
                amount_out_pool_1=Decimal('0'),
                amount_in_pool_2=Decimal('0'), 
                amount_out_pool_2=Decimal('0'),
                slippage_pool_1=Decimal('0'),
                slippage_pool_2=Decimal('0'),
                total_slippage=Decimal('0'),
                flash_loan_fee=Decimal('0'),
                gas_cost=Decimal('0'),
                mev_bribe=Decimal('0'),
                total_costs=Decimal('0'),
                pool_1_final=params.pool_1,
                pool_2_final=params.pool_2,
                is_profitable=False,
                arbitrage_exhausted=False
            )
        
        gross_profit, pool_1_final, pool_2_final = trade_result
        
        # Calculate price differences
        initial_price_1 = params.pool_1.price_a_to_b
        initial_price_2 = params.pool_2.price_a_to_b
        initial_price_diff = abs(initial_price_1 - initial_price_2) / max(initial_price_1, initial_price_2)
        
        final_price_1 = pool_1_final.price_a_to_b
        final_price_2 = pool_2_final.price_a_to_b
        final_price_diff = abs(final_price_1 - final_price_2) / max(final_price_1, final_price_2)
        
        price_convergence = (initial_price_diff - final_price_diff) / initial_price_diff * 100
        
        # Calculate slippage
        if direction == "pool1_to_pool2":
            expected_price_1 = initial_price_1
            actual_price_1 = (amount * pool_1_final.token_b_reserves) / (amount * pool_1_final.token_a_reserves) if amount > 0 else expected_price_1
            slippage_1 = abs(actual_price_1 - expected_price_1) / expected_price_1
            
            expected_price_2 = initial_price_2  
            actual_price_2 = final_price_2
            slippage_2 = abs(actual_price_2 - expected_price_2) / expected_price_2
        else:
            expected_price_2 = initial_price_2
            actual_price_2 = final_price_2
            slippage_2 = abs(actual_price_2 - expected_price_2) / expected_price_2
            
            expected_price_1 = initial_price_1
            actual_price_1 = final_price_1
            slippage_1 = abs(actual_price_1 - expected_price_1) / expected_price_1
        
        total_slippage = slippage_1 + slippage_2
        
        # Calculate costs
        flash_loan_fee = amount * params.flash_loan_fee_rate
        gas_cost = params.gas_cost_usd
        mev_bribe = amount * params.mev_bribe_rate
        total_costs = flash_loan_fee + gas_cost + mev_bribe
        
        net_profit = gross_profit - total_costs
        
        # Determine trade amounts for each pool
        if direction == "pool1_to_pool2":
            amount_in_1 = amount
            amount_out_1 = self._calculate_swap_output(amount, params.pool_1, token_in_is_a=True)
            amount_in_2 = amount_out_1
            amount_out_2 = self._calculate_swap_output(amount_out_1, params.pool_2, token_in_is_a=False)
        else:
            amount_in_2 = amount
            amount_out_2 = self._calculate_swap_output(amount, params.pool_2, token_in_is_a=True)
            amount_in_1 = amount_out_2
            amount_out_1 = self._calculate_swap_output(amount_out_2, params.pool_1, token_in_is_a=False)
        
        return AMMOptimizationResult(
            optimal_amount=amount,
            expected_profit=net_profit,
            initial_price_difference=initial_price_diff,
            final_price_difference=final_price_diff,
            price_convergence_percentage=price_convergence,
            amount_in_pool_1=amount_in_1,
            amount_out_pool_1=amount_out_1,
            amount_in_pool_2=amount_in_2,
            amount_out_pool_2=amount_out_2,
            slippage_pool_1=slippage_1,
            slippage_pool_2=slippage_2,
            total_slippage=total_slippage,
            flash_loan_fee=flash_loan_fee,
            gas_cost=gas_cost,
            mev_bribe=mev_bribe,
            total_costs=total_costs,
            pool_1_final=pool_1_final,
            pool_2_final=pool_2_final,
            is_profitable=net_profit >= params.min_profit_usd,
            arbitrage_exhausted=price_convergence > 90  # 90%+ price convergence
        )
    
    def analyze_price_impact(self, params: AMMArbitrageParameters, 
                           amounts: List[Decimal]) -> Dict[Decimal, Dict]:
        """
        Analyze how different trade amounts impact prices and convergence.
        
        Useful for understanding the diminishing returns of larger trades.
        """
        results = {}
        
        # Determine direction
        price_1 = params.pool_1.price_a_to_b
        price_2 = params.pool_2.price_a_to_b
        direction = "pool1_to_pool2" if price_1 < price_2 else "pool2_to_pool1"
        
        for amount in amounts:
            trade_result = self._simulate_arbitrage_trade(amount, params, direction)
            
            if trade_result:
                gross_profit, pool_1_final, pool_2_final = trade_result
                
                # Calculate metrics
                initial_price_diff = abs(price_1 - price_2) / max(price_1, price_2)
                final_price_1 = pool_1_final.price_a_to_b
                final_price_2 = pool_2_final.price_a_to_b
                final_price_diff = abs(final_price_1 - final_price_2) / max(final_price_1, final_price_2)
                
                convergence = (initial_price_diff - final_price_diff) / initial_price_diff * 100
                
                results[amount] = {
                    "gross_profit": gross_profit,
                    "price_convergence_pct": convergence,
                    "final_price_diff": final_price_diff,
                    "marginal_profit_per_dollar": gross_profit / amount if amount > 0 else Decimal('0')
                }
            else:
                results[amount] = {
                    "gross_profit": Decimal('0'),
                    "price_convergence_pct": Decimal('0'), 
                    "final_price_diff": Decimal('0'),
                    "marginal_profit_per_dollar": Decimal('0')
                }
        
        return results