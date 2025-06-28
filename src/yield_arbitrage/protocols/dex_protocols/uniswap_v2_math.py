"""
Uniswap V2 Math Implementation.

Implements the exact constant product formula (x * y = k) used by Uniswap V2
and its forks (SushiSwap, PancakeSwap, etc.).
"""
from decimal import Decimal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UniswapV2Math:
    """
    Exact implementation of Uniswap V2 constant product AMM math.
    
    This is the baseline for many DEXs and serves as a good reference
    for more complex implementations.
    """
    
    def __init__(self, fee_rate: Decimal = Decimal('0.003')):
        """
        Initialize Uniswap V2 math.
        
        Args:
            fee_rate: Trading fee rate (default 0.3%)
        """
        self.fee_rate = fee_rate
        self.fee_precision = Decimal('1000')  # Uniswap V2 uses 1000 for fee calculations
        
    def calculate_amount_out(self, 
                           amount_in: Decimal,
                           reserve_in: Decimal,
                           reserve_out: Decimal) -> Decimal:
        """
        Calculate output amount for a given input using exact Uniswap V2 formula.
        
        Formula: amountOut = (amountIn * 997 * reserveOut) / (reserveIn * 1000 + amountIn * 997)
        
        The 997/1000 factor accounts for the 0.3% fee.
        
        Args:
            amount_in: Amount of input token
            reserve_in: Reserve of input token in pool
            reserve_out: Reserve of output token in pool
            
        Returns:
            Amount of output token
        """
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return Decimal('0')
        
        # Apply fee to input amount (997/1000 for 0.3% fee)
        amount_in_with_fee = amount_in * (self.fee_precision - self.fee_rate * self.fee_precision)
        
        # Constant product formula with fee
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * self.fee_precision + amount_in_with_fee
        
        amount_out = numerator / denominator
        
        # Safety check: can't drain the pool
        if amount_out >= reserve_out:
            logger.warning(f"Trade would drain pool: {amount_out} >= {reserve_out}")
            return Decimal('0')
        
        return amount_out
    
    def calculate_amount_in(self,
                          amount_out: Decimal,
                          reserve_in: Decimal,
                          reserve_out: Decimal) -> Decimal:
        """
        Calculate required input amount for a desired output.
        
        Formula: amountIn = (reserveIn * amountOut * 1000) / ((reserveOut - amountOut) * 997) + 1
        
        Args:
            amount_out: Desired amount of output token
            reserve_in: Reserve of input token in pool
            reserve_out: Reserve of output token in pool
            
        Returns:
            Required amount of input token
        """
        if amount_out <= 0 or amount_out >= reserve_out:
            return Decimal('999999999999')  # Effectively infinite
        
        if reserve_in <= 0 or reserve_out <= 0:
            return Decimal('999999999999')
        
        numerator = reserve_in * amount_out * self.fee_precision
        denominator = (reserve_out - amount_out) * (self.fee_precision - self.fee_rate * self.fee_precision)
        
        # Add 1 to round up (ensure sufficient input)
        amount_in = numerator / denominator + Decimal('1')
        
        return amount_in
    
    def calculate_price_impact(self,
                             amount_in: Decimal,
                             reserve_in: Decimal,
                             reserve_out: Decimal) -> Decimal:
        """
        Calculate the price impact of a trade.
        
        Price impact = 1 - (post_trade_price / pre_trade_price)
        
        Args:
            amount_in: Amount of input token
            reserve_in: Reserve of input token in pool
            reserve_out: Reserve of output token in pool
            
        Returns:
            Price impact as a decimal (0.01 = 1% impact)
        """
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return Decimal('0')
        
        # Pre-trade price (output per input)
        pre_trade_price = reserve_out / reserve_in
        
        # Calculate output amount
        amount_out = self.calculate_amount_out(amount_in, reserve_in, reserve_out)
        
        if amount_out <= 0:
            return Decimal('1')  # 100% impact if trade fails
        
        # Post-trade reserves
        new_reserve_in = reserve_in + amount_in
        new_reserve_out = reserve_out - amount_out
        
        # Post-trade price
        post_trade_price = new_reserve_out / new_reserve_in
        
        # Price impact
        price_impact = Decimal('1') - (post_trade_price / pre_trade_price)
        
        return price_impact
    
    def get_spot_price(self,
                      reserve_in: Decimal,
                      reserve_out: Decimal,
                      include_fee: bool = True) -> Decimal:
        """
        Get the current spot price (including or excluding fees).
        
        Args:
            reserve_in: Reserve of input token
            reserve_out: Reserve of output token
            include_fee: Whether to include the fee in the price
            
        Returns:
            Spot price (output tokens per input token)
        """
        if reserve_in <= 0 or reserve_out <= 0:
            return Decimal('0')
        
        spot_price = reserve_out / reserve_in
        
        if include_fee:
            # Adjust for fee
            spot_price = spot_price * (self.fee_precision - self.fee_rate * self.fee_precision) / self.fee_precision
        
        return spot_price
    
    def calculate_liquidity_value(self,
                                reserve_0: Decimal,
                                reserve_1: Decimal) -> Decimal:
        """
        Calculate the invariant k = x * y.
        
        Args:
            reserve_0: Reserve of token 0
            reserve_1: Reserve of token 1
            
        Returns:
            The constant product invariant
        """
        return reserve_0 * reserve_1
    
    def simulate_swap_to_price(self,
                             current_reserve_in: Decimal,
                             current_reserve_out: Decimal,
                             target_price: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Calculate how much to swap to reach a target price.
        
        This is useful for arbitrage calculations to find the optimal swap amount
        that brings two pools to the same price.
        
        Args:
            current_reserve_in: Current reserve of input token
            current_reserve_out: Current reserve of output token
            target_price: Target price (output/input)
            
        Returns:
            Tuple of (amount_in, amount_out) to reach target price
        """
        if current_reserve_in <= 0 or current_reserve_out <= 0 or target_price <= 0:
            return (Decimal('0'), Decimal('0'))
        
        # Current price
        current_price = current_reserve_out / current_reserve_in
        
        if abs(current_price - target_price) < Decimal('0.0001'):
            return (Decimal('0'), Decimal('0'))  # Already at target
        
        # Calculate required reserves for target price
        # Using the invariant k = x * y and price = y/x
        k = current_reserve_in * current_reserve_out
        
        # Solve for new reserves
        # k = x * y and p = y/x => y = p*x and k = x * p*x = p*x²
        # So x = sqrt(k/p) and y = sqrt(k*p)
        new_reserve_in = (k / target_price).sqrt()
        new_reserve_out = (k * target_price).sqrt()
        
        # Adjust for fees
        fee_multiplier = (self.fee_precision - self.fee_rate * self.fee_precision) / self.fee_precision
        
        # Calculate swap amounts
        if new_reserve_in > current_reserve_in:
            # Need to add to input reserve (swap in)
            amount_in_no_fee = new_reserve_in - current_reserve_in
            amount_in = amount_in_no_fee / fee_multiplier
            amount_out = current_reserve_out - new_reserve_out
        else:
            # This case would require swapping in the opposite direction
            amount_in = Decimal('0')
            amount_out = Decimal('0')
        
        return (amount_in, amount_out)
    
    def calculate_arbitrage_profit(self,
                                 amount_in: Decimal,
                                 pool1_reserve_in: Decimal,
                                 pool1_reserve_out: Decimal,
                                 pool2_reserve_in: Decimal,
                                 pool2_reserve_out: Decimal) -> Decimal:
        """
        Calculate profit from arbitraging between two pools.
        
        Simulates: Token A -> Token B (Pool 1) -> Token A (Pool 2)
        
        Args:
            amount_in: Amount of token A to start with
            pool1_reserve_in: Token A reserves in pool 1
            pool1_reserve_out: Token B reserves in pool 1
            pool2_reserve_in: Token B reserves in pool 2  
            pool2_reserve_out: Token A reserves in pool 2
            
        Returns:
            Net profit in token A (can be negative)
        """
        # Step 1: Swap A for B in pool 1
        amount_b = self.calculate_amount_out(amount_in, pool1_reserve_in, pool1_reserve_out)
        
        if amount_b <= 0:
            return Decimal('-999999')
        
        # Step 2: Swap B for A in pool 2
        amount_a_final = self.calculate_amount_out(amount_b, pool2_reserve_in, pool2_reserve_out)
        
        # Profit = final amount - initial amount
        profit = amount_a_final - amount_in
        
        return profit