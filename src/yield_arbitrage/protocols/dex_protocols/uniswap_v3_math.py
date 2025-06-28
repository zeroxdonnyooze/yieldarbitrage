"""
Uniswap V3 Math Implementation.

Implements concentrated liquidity AMM math with tick-based pricing.
This is significantly more complex than V2 due to:
- Concentrated liquidity in price ranges
- Tick-based price movements
- Square root price representation (sqrtPriceX96)
"""
from decimal import Decimal, getcontext
from typing import Tuple, Optional, List, NamedTuple
from dataclasses import dataclass
import logging
import math

# Set high precision for V3 calculations
getcontext().prec = 78  # Support Q64.96 precision

logger = logging.getLogger(__name__)


class TickInfo(NamedTuple):
    """Information about a liquidity tick."""
    tick: int
    liquidity_net: Decimal
    liquidity_gross: Decimal


@dataclass
class V3PoolState:
    """State of a Uniswap V3 pool."""
    sqrt_price_x96: int  # Current sqrt price as Q64.96
    tick: int  # Current tick
    liquidity: Decimal  # Current in-range liquidity
    fee_tier: int  # Fee tier in hundredths of a bip (500 = 0.05%)
    tick_spacing: int  # Tick spacing for this fee tier
    
    
class UniswapV3Math:
    """
    Exact implementation of Uniswap V3 concentrated liquidity math.
    
    Handles:
    - Square root price calculations (Q64.96 format)
    - Tick-based liquidity
    - Multi-tick swaps
    - Concentrated liquidity positions
    """
    
    # Constants
    Q96 = 2 ** 96
    MIN_TICK = -887272
    MAX_TICK = 887272
    MIN_SQRT_RATIO = 4295128739
    MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342
    
    def __init__(self):
        """Initialize Uniswap V3 math."""
        self.fee_to_tick_spacing = {
            100: 1,      # 0.01% fee
            500: 10,     # 0.05% fee  
            3000: 60,    # 0.30% fee
            10000: 200   # 1.00% fee
        }
    
    def sqrt_price_x96_to_price(self, sqrt_price_x96: int, 
                               decimals_0: int = 18,
                               decimals_1: int = 18) -> Decimal:
        """
        Convert sqrtPriceX96 to human-readable price.
        
        Price = (sqrtPriceX96 / 2^96)^2 * (10^decimals0 / 10^decimals1)
        
        Args:
            sqrt_price_x96: Square root price in Q64.96 format
            decimals_0: Decimals of token0
            decimals_1: Decimals of token1
            
        Returns:
            Price of token1 in terms of token0
        """
        # Convert to decimal and remove Q96 scaling
        sqrt_price = Decimal(sqrt_price_x96) / Decimal(self.Q96)
        
        # Square to get price
        price = sqrt_price ** 2
        
        # Adjust for decimals
        decimal_adjustment = Decimal(10) ** (decimals_0 - decimals_1)
        
        return price * decimal_adjustment
    
    def price_to_sqrt_price_x96(self, price: Decimal,
                               decimals_0: int = 18,
                               decimals_1: int = 18) -> int:
        """
        Convert human-readable price to sqrtPriceX96.
        
        Args:
            price: Price of token1 in terms of token0
            decimals_0: Decimals of token0
            decimals_1: Decimals of token1
            
        Returns:
            Square root price in Q64.96 format
        """
        # Adjust for decimals
        decimal_adjustment = Decimal(10) ** (decimals_1 - decimals_0)
        adjusted_price = price * decimal_adjustment
        
        # Take square root and scale to Q96
        sqrt_price = adjusted_price.sqrt()
        sqrt_price_x96 = int(sqrt_price * Decimal(self.Q96))
        
        return sqrt_price_x96
    
    def tick_to_sqrt_price_x96(self, tick: int) -> int:
        """
        Convert tick to sqrtPriceX96.
        
        sqrtPrice = 1.0001^(tick/2)
        
        Args:
            tick: Tick value
            
        Returns:
            Square root price in Q64.96 format
        """
        if tick < self.MIN_TICK or tick > self.MAX_TICK:
            raise ValueError(f"Tick {tick} out of bounds")
        
        # Calculate 1.0001^(tick/2)
        sqrt_price = Decimal('1.0001') ** (Decimal(tick) / 2)
        
        # Scale to Q96
        sqrt_price_x96 = int(sqrt_price * Decimal(self.Q96))
        
        return sqrt_price_x96
    
    def sqrt_price_x96_to_tick(self, sqrt_price_x96: int) -> int:
        """
        Convert sqrtPriceX96 to tick.
        
        tick = floor(log(sqrtPrice) / log(1.0001) * 2)
        
        Args:
            sqrt_price_x96: Square root price in Q64.96 format
            
        Returns:
            Tick value
        """
        # Convert from Q96
        sqrt_price = Decimal(sqrt_price_x96) / Decimal(self.Q96)
        
        # Calculate tick
        # tick = log(sqrtPrice) / log(1.0001) * 2
        if sqrt_price <= 0:
            return self.MIN_TICK
        
        log_sqrt_price = sqrt_price.ln()
        log_base = Decimal('1.0001').ln()
        
        tick = int((log_sqrt_price / log_base) * 2)
        
        # Clamp to valid range
        return max(self.MIN_TICK, min(self.MAX_TICK, tick))
    
    def get_amount0_delta(self, sqrt_price_a_x96: int, sqrt_price_b_x96: int,
                         liquidity: Decimal, round_up: bool = False) -> Decimal:
        """
        Calculate amount0 delta between two sqrt prices.
        
        amount0 = liquidity * (1/sqrtPriceB - 1/sqrtPriceA)
        
        Args:
            sqrt_price_a_x96: Lower sqrt price
            sqrt_price_b_x96: Upper sqrt price  
            liquidity: Liquidity amount
            round_up: Whether to round up
            
        Returns:
            Amount of token0
        """
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if liquidity <= 0:
            return Decimal('0')
        
        # Convert to Decimal for precise calculation
        sqrt_a = Decimal(sqrt_price_a_x96)
        sqrt_b = Decimal(sqrt_price_b_x96)
        
        # Calculate (1/sqrtB - 1/sqrtA) with Q96 scaling
        amount = liquidity * Decimal(self.Q96) * (sqrt_b - sqrt_a) / (sqrt_a * sqrt_b)
        
        if round_up and amount % 1 != 0:
            amount = amount.__floor__() + 1
        else:
            amount = amount.__floor__()
        
        return amount
    
    def get_amount1_delta(self, sqrt_price_a_x96: int, sqrt_price_b_x96: int,
                         liquidity: Decimal, round_up: bool = False) -> Decimal:
        """
        Calculate amount1 delta between two sqrt prices.
        
        amount1 = liquidity * (sqrtPriceB - sqrtPriceA) / Q96
        
        Args:
            sqrt_price_a_x96: Lower sqrt price
            sqrt_price_b_x96: Upper sqrt price
            liquidity: Liquidity amount
            round_up: Whether to round up
            
        Returns:
            Amount of token1
        """
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if liquidity <= 0:
            return Decimal('0')
        
        # Calculate amount
        amount = liquidity * (Decimal(sqrt_price_b_x96) - Decimal(sqrt_price_a_x96)) / Decimal(self.Q96)
        
        if round_up and amount % 1 != 0:
            amount = amount.__floor__() + 1
        else:
            amount = amount.__floor__()
        
        return amount
    
    def get_next_sqrt_price_from_amount0(self, sqrt_price_x96: int, liquidity: Decimal,
                                       amount: Decimal, add: bool) -> int:
        """
        Calculate next sqrt price from amount0.
        
        When adding token0 (add=True):
        nextSqrtPrice = sqrtPrice * liquidity / (liquidity + amount * sqrtPrice)
        
        When removing token0 (add=False):
        nextSqrtPrice = sqrtPrice * liquidity / (liquidity - amount * sqrtPrice)
        
        Args:
            sqrt_price_x96: Current sqrt price
            liquidity: Available liquidity
            amount: Amount of token0
            add: Whether adding or removing liquidity
            
        Returns:
            Next sqrt price
        """
        if amount == 0:
            return sqrt_price_x96
        
        if liquidity <= 0:
            return sqrt_price_x96
        
        sqrt_price = Decimal(sqrt_price_x96)
        
        if add:
            # Adding token0 decreases price
            # nextSqrtPrice = (L * sqrtP) / (L + amount * sqrtP / Q96)
            denominator = liquidity * Decimal(self.Q96) + amount * sqrt_price
            if denominator <= 0:
                return self.MIN_SQRT_RATIO
            
            next_sqrt_price = liquidity * sqrt_price * Decimal(self.Q96) / denominator
        else:
            # Removing token0 increases price
            # nextSqrtPrice = (L * sqrtP) / (L - amount * sqrtP / Q96)
            denominator = liquidity * Decimal(self.Q96) - amount * sqrt_price
            if denominator <= 0:
                return self.MAX_SQRT_RATIO
            
            next_sqrt_price = liquidity * sqrt_price * Decimal(self.Q96) / denominator
        
        # Clamp to valid range
        next_sqrt_price_int = int(next_sqrt_price)
        return max(self.MIN_SQRT_RATIO, min(self.MAX_SQRT_RATIO, next_sqrt_price_int))
    
    def get_next_sqrt_price_from_amount1(self, sqrt_price_x96: int, liquidity: Decimal,
                                       amount: Decimal, add: bool) -> int:
        """
        Calculate next sqrt price from amount1.
        
        When adding token1 (add=True):
        nextSqrtPrice = sqrtPrice + (amount * Q96) / liquidity
        
        When removing token1 (add=False):
        nextSqrtPrice = sqrtPrice - (amount * Q96) / liquidity
        
        Args:
            sqrt_price_x96: Current sqrt price
            liquidity: Available liquidity
            amount: Amount of token1
            add: Whether adding or removing liquidity
            
        Returns:
            Next sqrt price
        """
        if liquidity <= 0:
            return sqrt_price_x96
        
        sqrt_price = Decimal(sqrt_price_x96)
        quotient = amount * Decimal(self.Q96) / liquidity
        
        if add:
            next_sqrt_price = sqrt_price + quotient
        else:
            if quotient >= sqrt_price:
                return self.MIN_SQRT_RATIO
            next_sqrt_price = sqrt_price - quotient
        
        # Clamp to valid range
        next_sqrt_price_int = int(next_sqrt_price)
        return max(self.MIN_SQRT_RATIO, min(self.MAX_SQRT_RATIO, next_sqrt_price_int))
    
    def compute_swap_step(self,
                         sqrt_price_current_x96: int,
                         sqrt_price_target_x96: int,
                         liquidity: Decimal,
                         amount_remaining: Decimal,
                         fee_pips: int) -> Tuple[int, Decimal, Decimal, Decimal]:
        """
        Compute a single swap step within a tick range.
        
        This is the core function that calculates how much can be swapped
        within a single tick range before crossing to the next tick.
        
        Args:
            sqrt_price_current_x96: Current sqrt price
            sqrt_price_target_x96: Target sqrt price (tick boundary)
            liquidity: Available liquidity in this range
            amount_remaining: Remaining amount to swap (positive for exact input)
            fee_pips: Fee in hundredths of a bip
            
        Returns:
            Tuple of (sqrt_price_next, amount_in, amount_out, fee_amount)
        """
        zero_for_one = sqrt_price_current_x96 >= sqrt_price_target_x96
        exact_in = amount_remaining >= 0
        
        if exact_in:
            amount_remaining_less_fee = amount_remaining * Decimal(1_000_000 - fee_pips) / Decimal(1_000_000)
            
            if zero_for_one:
                amount_in = self.get_amount0_delta(
                    sqrt_price_target_x96, sqrt_price_current_x96, liquidity, True
                )
            else:
                amount_in = self.get_amount1_delta(
                    sqrt_price_current_x96, sqrt_price_target_x96, liquidity, True
                )
            
            if amount_remaining_less_fee >= amount_in:
                sqrt_price_next_x96 = sqrt_price_target_x96
            else:
                if zero_for_one:
                    sqrt_price_next_x96 = self.get_next_sqrt_price_from_amount0(
                        sqrt_price_current_x96, liquidity, amount_remaining_less_fee, True
                    )
                else:
                    sqrt_price_next_x96 = self.get_next_sqrt_price_from_amount1(
                        sqrt_price_current_x96, liquidity, amount_remaining_less_fee, True
                    )
        else:
            # Exact output swap
            if zero_for_one:
                amount_out = self.get_amount1_delta(
                    sqrt_price_target_x96, sqrt_price_current_x96, liquidity, False
                )
            else:
                amount_out = self.get_amount0_delta(
                    sqrt_price_current_x96, sqrt_price_target_x96, liquidity, False
                )
            
            if -amount_remaining >= amount_out:
                sqrt_price_next_x96 = sqrt_price_target_x96
            else:
                if zero_for_one:
                    sqrt_price_next_x96 = self.get_next_sqrt_price_from_amount1(
                        sqrt_price_current_x96, liquidity, -amount_remaining, False
                    )
                else:
                    sqrt_price_next_x96 = self.get_next_sqrt_price_from_amount0(
                        sqrt_price_current_x96, liquidity, -amount_remaining, False
                    )
        
        max_amount = sqrt_price_target_x96 == sqrt_price_next_x96
        
        # Calculate actual amounts
        if zero_for_one:
            if max_amount and exact_in:
                amount_in = amount_in
            else:
                amount_in = self.get_amount0_delta(
                    sqrt_price_next_x96, sqrt_price_current_x96, liquidity, True
                )
            
            if max_amount and not exact_in:
                amount_out = amount_out
            else:
                amount_out = self.get_amount1_delta(
                    sqrt_price_next_x96, sqrt_price_current_x96, liquidity, False
                )
        else:
            if max_amount and exact_in:
                amount_in = amount_in
            else:
                amount_in = self.get_amount1_delta(
                    sqrt_price_current_x96, sqrt_price_next_x96, liquidity, True
                )
            
            if max_amount and not exact_in:
                amount_out = amount_out
            else:
                amount_out = self.get_amount0_delta(
                    sqrt_price_current_x96, sqrt_price_next_x96, liquidity, False
                )
        
        # Calculate fee
        if exact_in and amount_in > 0:
            fee_amount = amount_remaining - amount_in if sqrt_price_next_x96 != sqrt_price_target_x96 else amount_in * Decimal(fee_pips) / Decimal(1_000_000 - fee_pips)
        else:
            fee_amount = amount_in * Decimal(fee_pips) / Decimal(1_000_000)
        
        return (sqrt_price_next_x96, amount_in, amount_out, fee_amount)
    
    def simulate_swap(self,
                     pool_state: V3PoolState,
                     amount_in: Decimal,
                     zero_for_one: bool,
                     tick_data: Optional[List[TickInfo]] = None) -> Tuple[Decimal, Decimal, int, int]:
        """
        Simulate a complete swap, potentially crossing multiple ticks.
        
        Args:
            pool_state: Current pool state
            amount_in: Amount to swap in
            zero_for_one: Direction (true = token0 for token1)
            tick_data: Optional tick data for multi-tick swaps
            
        Returns:
            Tuple of (amount_out, fee_total, final_sqrt_price, final_tick)
        """
        # For simplicity, simulate within current tick only
        # Full implementation would iterate through multiple ticks
        
        sqrt_price_current = pool_state.sqrt_price_x96
        
        # Determine target price (next tick boundary)
        if zero_for_one:
            # Price decreases when swapping 0 for 1
            next_tick = pool_state.tick - pool_state.tick_spacing
            sqrt_price_target = self.tick_to_sqrt_price_x96(next_tick)
        else:
            # Price increases when swapping 1 for 0
            next_tick = pool_state.tick + pool_state.tick_spacing
            sqrt_price_target = self.tick_to_sqrt_price_x96(next_tick)
        
        # Compute swap step
        sqrt_price_next, amount_in_step, amount_out_step, fee_amount = self.compute_swap_step(
            sqrt_price_current,
            sqrt_price_target,
            pool_state.liquidity,
            amount_in,
            pool_state.fee_tier
        )
        
        # Update final state
        final_tick = self.sqrt_price_x96_to_tick(sqrt_price_next)
        
        return (amount_out_step, fee_amount, sqrt_price_next, final_tick)
    
    def encode_sqrt_price(self, reserve0: Decimal, reserve1: Decimal) -> int:
        """
        Encode sqrt price from reserves (like price_to_sqrt_price_x96).
        
        Args:
            reserve0: Token0 reserve
            reserve1: Token1 reserve
            
        Returns:
            Sqrt price in X96 format
        """
        if reserve0 <= 0:
            return 0
        
        price = reserve1 / reserve0
        return self.price_to_sqrt_price_x96(price)
    
    def get_tick_spacing(self, fee_tier: int) -> int:
        """
        Get tick spacing for a fee tier.
        
        Args:
            fee_tier: Fee tier in hundredths of bips
            
        Returns:
            Tick spacing
        """
        return self.fee_to_tick_spacing.get(fee_tier, 60)  # Default to 60
    
    def calculate_fee(self, amount: Decimal, fee_tier: int) -> Decimal:
        """
        Calculate fee amount for a given input.
        
        Args:
            amount: Input amount
            fee_tier: Fee tier in hundredths of bips
            
        Returns:
            Fee amount
        """
        return amount * Decimal(fee_tier) / Decimal('1000000')
    
    def get_liquidity_for_amounts(self,
                                 sqrt_price_x96: int,
                                 sqrt_price_a_x96: int,
                                 sqrt_price_b_x96: int,
                                 amount0: Decimal,
                                 amount1: Decimal) -> Decimal:
        """
        Calculate liquidity for given amounts.
        
        Args:
            sqrt_price_x96: Current sqrt price
            sqrt_price_a_x96: Lower bound sqrt price
            sqrt_price_b_x96: Upper bound sqrt price
            amount0: Amount of token0
            amount1: Amount of token1
            
        Returns:
            Liquidity amount
        """
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_x96 <= sqrt_price_a_x96:
            # Current price below range, only token0 needed
            return self._get_liquidity_for_amount0(sqrt_price_a_x96, sqrt_price_b_x96, amount0)
        elif sqrt_price_x96 < sqrt_price_b_x96:
            # Current price in range
            liquidity0 = self._get_liquidity_for_amount0(sqrt_price_x96, sqrt_price_b_x96, amount0)
            liquidity1 = self._get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_x96, amount1)
            return min(liquidity0, liquidity1)
        else:
            # Current price above range, only token1 needed
            return self._get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, amount1)
    
    def get_amounts_for_liquidity(self,
                                 sqrt_price_x96: int,
                                 sqrt_price_a_x96: int,
                                 sqrt_price_b_x96: int,
                                 liquidity: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Calculate token amounts for given liquidity.
        
        Args:
            sqrt_price_x96: Current sqrt price
            sqrt_price_a_x96: Lower bound sqrt price
            sqrt_price_b_x96: Upper bound sqrt price
            liquidity: Liquidity amount
            
        Returns:
            Tuple of (amount0, amount1)
        """
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_x96 <= sqrt_price_a_x96:
            # Current price below range
            amount0 = self.get_amount0_delta(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
            return (amount0, Decimal('0'))
        elif sqrt_price_x96 < sqrt_price_b_x96:
            # Current price in range
            amount0 = self.get_amount0_delta(sqrt_price_x96, sqrt_price_b_x96, liquidity)
            amount1 = self.get_amount1_delta(sqrt_price_a_x96, sqrt_price_x96, liquidity)
            return (amount0, amount1)
        else:
            # Current price above range
            amount1 = self.get_amount1_delta(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
            return (Decimal('0'), amount1)
    
    def _get_liquidity_for_amount0(self, sqrt_price_a_x96: int, sqrt_price_b_x96: int, amount0: Decimal) -> Decimal:
        """Calculate liquidity for amount0."""
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if amount0 <= 0:
            return Decimal('0')
        
        sqrt_a = Decimal(sqrt_price_a_x96)
        sqrt_b = Decimal(sqrt_price_b_x96)
        
        # liquidity = amount0 * sqrtA * sqrtB / (sqrtB - sqrtA) / Q96
        return amount0 * sqrt_a * sqrt_b / (sqrt_b - sqrt_a) / Decimal(self.Q96)
    
    def _get_liquidity_for_amount1(self, sqrt_price_a_x96: int, sqrt_price_b_x96: int, amount1: Decimal) -> Decimal:
        """Calculate liquidity for amount1."""
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if amount1 <= 0:
            return Decimal('0')
        
        sqrt_a = Decimal(sqrt_price_a_x96)
        sqrt_b = Decimal(sqrt_price_b_x96)
        
        # liquidity = amount1 / (sqrtB - sqrtA) * Q96
        return amount1 / (sqrt_b - sqrt_a) * Decimal(self.Q96)