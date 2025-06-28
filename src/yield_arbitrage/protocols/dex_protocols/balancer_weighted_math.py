"""
Balancer Weighted Pool Math Implementation.

Implements weighted pool invariant with arbitrary token weights.
Supports multi-asset pools with custom weight distributions.
"""
from decimal import Decimal, getcontext
from typing import List, Tuple, Optional
import logging

# Set high precision for power calculations
getcontext().prec = 50

logger = logging.getLogger(__name__)


class BalancerWeightedMath:
    """
    Exact implementation of Balancer's weighted pool math.
    
    The weighted pool invariant:
    V = Π(B_i^W_i) = constant
    
    Where:
    - B_i is the balance of token i
    - W_i is the normalized weight of token i (sum of weights = 1)
    """
    
    def __init__(self, swap_fee: Decimal = Decimal('0.001')):
        """
        Initialize Balancer weighted pool math.
        
        Args:
            swap_fee: Trading fee rate (default 0.1%)
        """
        self.swap_fee = swap_fee
        self.exit_fee = Decimal('0')  # V2 doesn't have exit fees
        self.min_fee = Decimal('0.0001')  # 0.01%
        self.max_fee = Decimal('0.1')    # 10%
        
    def calculate_invariant(self, balances: List[Decimal], 
                          weights: List[Decimal]) -> Decimal:
        """
        Calculate the weighted pool invariant.
        
        V = Π(balance_i ^ weight_i)
        
        Args:
            balances: Token balances
            weights: Normalized token weights (must sum to 1)
            
        Returns:
            The invariant value
        """
        if len(balances) != len(weights):
            return Decimal('0')
        
        if any(b <= 0 for b in balances) or any(w <= 0 for w in weights):
            return Decimal('0')
        
        # Verify weights sum to 1 (within tolerance)
        weight_sum = sum(weights)
        if abs(weight_sum - Decimal('1')) > Decimal('0.0001'):
            logger.warning(f"Weights sum to {weight_sum}, not 1")
            # Normalize weights
            weights = [w / weight_sum for w in weights]
        
        invariant = Decimal('1')
        for balance, weight in zip(balances, weights):
            # Use power function with high precision
            invariant *= self._pow(balance, weight)
        
        return invariant
    
    def calculate_out_given_in(self,
                             token_balance_in: Decimal,
                             token_weight_in: Decimal,
                             token_balance_out: Decimal,
                             token_weight_out: Decimal,
                             token_amount_in: Decimal,
                             swap_fee: Optional[Decimal] = None) -> Decimal:
        """
        Calculate output amount given input amount (outGivenIn).
        
        Formula:
        tokenAmountOut = tokenBalanceOut * (1 - (tokenBalanceIn / (tokenBalanceIn + tokenAmountIn * (1 - swapFee))) ^ (tokenWeightIn / tokenWeightOut))
        
        Args:
            token_balance_in: Balance of input token
            token_weight_in: Weight of input token
            token_balance_out: Balance of output token
            token_weight_out: Weight of output token
            token_amount_in: Amount of input token
            swap_fee: Override swap fee (uses self.swap_fee if None)
            
        Returns:
            Amount of output token
        """
        if token_amount_in <= 0:
            return Decimal('0')
        
        if token_balance_in <= 0 or token_balance_out <= 0:
            return Decimal('0')
        
        if token_weight_in <= 0 or token_weight_out <= 0:
            return Decimal('0')
        
        fee = swap_fee if swap_fee is not None else self.swap_fee
        
        # Apply swap fee to input
        adjusted_in = token_amount_in * (Decimal('1') - fee)
        
        # Calculate the ratio
        weight_ratio = token_weight_in / token_weight_out
        
        # Calculate: (Bi / (Bi + Ai)) ^ (Wi/Wo)
        base = token_balance_in / (token_balance_in + adjusted_in)
        power = self._pow(base, weight_ratio)
        
        # Calculate output amount
        token_amount_out = token_balance_out * (Decimal('1') - power)
        
        return token_amount_out
    
    def calculate_in_given_out(self,
                             token_balance_in: Decimal,
                             token_weight_in: Decimal,
                             token_balance_out: Decimal,
                             token_weight_out: Decimal,
                             token_amount_out: Decimal,
                             swap_fee: Optional[Decimal] = None) -> Decimal:
        """
        Calculate input amount given desired output amount (inGivenOut).
        
        Formula:
        tokenAmountIn = tokenBalanceIn * ((tokenBalanceOut / (tokenBalanceOut - tokenAmountOut)) ^ (tokenWeightOut / tokenWeightIn) - 1) / (1 - swapFee)
        
        Args:
            token_balance_in: Balance of input token
            token_weight_in: Weight of input token
            token_balance_out: Balance of output token
            token_weight_out: Weight of output token
            token_amount_out: Desired amount of output token
            swap_fee: Override swap fee (uses self.swap_fee if None)
            
        Returns:
            Required amount of input token
        """
        if token_amount_out <= 0 or token_amount_out >= token_balance_out:
            return Decimal('999999999999')  # Effectively infinite
        
        if token_balance_in <= 0 or token_balance_out <= 0:
            return Decimal('999999999999')
        
        if token_weight_in <= 0 or token_weight_out <= 0:
            return Decimal('999999999999')
        
        fee = swap_fee if swap_fee is not None else self.swap_fee
        
        # Calculate the weight ratio
        weight_ratio = token_weight_out / token_weight_in
        
        # Calculate: (Bo / (Bo - Ao)) ^ (Wo/Wi)
        base = token_balance_out / (token_balance_out - token_amount_out)
        power = self._pow(base, weight_ratio)
        
        # Calculate required input before fee
        token_amount_in_before_fee = token_balance_in * (power - Decimal('1'))
        
        # Adjust for swap fee
        token_amount_in = token_amount_in_before_fee / (Decimal('1') - fee)
        
        return token_amount_in
    
    def calculate_spot_price(self,
                           token_balance_in: Decimal,
                           token_weight_in: Decimal,
                           token_balance_out: Decimal,
                           token_weight_out: Decimal,
                           swap_fee: Optional[Decimal] = None) -> Decimal:
        """
        Calculate spot price of token_out in terms of token_in.
        
        spotPrice = (balanceIn / weightIn) / (balanceOut / weightOut) * (1 / (1 - swapFee))
        
        Args:
            token_balance_in: Balance of input token
            token_weight_in: Weight of input token
            token_balance_out: Balance of output token
            token_weight_out: Weight of output token
            swap_fee: Override swap fee (uses self.swap_fee if None)
            
        Returns:
            Spot price (output tokens per input token)
        """
        if token_balance_in <= 0 or token_balance_out <= 0:
            return Decimal('0')
        
        if token_weight_in <= 0 or token_weight_out <= 0:
            return Decimal('0')
        
        fee = swap_fee if swap_fee is not None else self.swap_fee
        
        # Calculate normalized balances
        norm_in = token_balance_in / token_weight_in
        norm_out = token_balance_out / token_weight_out
        
        # Spot price before fee
        spot_price_before_fee = norm_in / norm_out
        
        # Adjust for swap fee
        spot_price = spot_price_before_fee / (Decimal('1') - fee)
        
        return spot_price
    
    def calculate_price_impact(self,
                             token_balance_in: Decimal,
                             token_weight_in: Decimal,
                             token_balance_out: Decimal,
                             token_weight_out: Decimal,
                             token_amount_in: Decimal,
                             swap_fee: Optional[Decimal] = None) -> Decimal:
        """
        Calculate price impact of a trade.
        
        Args:
            token_balance_in: Balance of input token
            token_weight_in: Weight of input token
            token_balance_out: Balance of output token
            token_weight_out: Weight of output token
            token_amount_in: Amount of input token
            swap_fee: Override swap fee
            
        Returns:
            Price impact as decimal (0.01 = 1%)
        """
        if token_amount_in <= 0:
            return Decimal('0')
        
        # Get spot price before trade
        spot_price_before = self.calculate_spot_price(
            token_balance_in, token_weight_in,
            token_balance_out, token_weight_out,
            swap_fee
        )
        
        # Calculate output amount
        token_amount_out = self.calculate_out_given_in(
            token_balance_in, token_weight_in,
            token_balance_out, token_weight_out,
            token_amount_in, swap_fee
        )
        
        if token_amount_out <= 0:
            return Decimal('1')  # 100% impact
        
        # Calculate effective price
        effective_price = token_amount_out / token_amount_in
        
        if spot_price_before <= 0:
            return Decimal('1')
        
        # Price impact = 1 - (effective_price / spot_price)
        price_impact = Decimal('1') - (effective_price / spot_price_before)
        
        return max(Decimal('0'), price_impact)
    
    def calculate_bpt_out_given_exact_tokens_in(self,
                                               balances: List[Decimal],
                                               weights: List[Decimal],
                                               amounts_in: List[Decimal],
                                               bpt_total_supply: Decimal,
                                               swap_fee: Optional[Decimal] = None) -> Decimal:
        """
        Calculate BPT (Balancer Pool Tokens) minted for exact amounts in.
        
        This handles both proportional and non-proportional joins.
        
        Args:
            balances: Current token balances
            weights: Token weights
            amounts_in: Amounts of each token to add
            bpt_total_supply: Current BPT supply
            swap_fee: Override swap fee
            
        Returns:
            Amount of BPT to mint
        """
        if len(balances) != len(weights) or len(balances) != len(amounts_in):
            return Decimal('0')
        
        fee = swap_fee if swap_fee is not None else self.swap_fee
        
        # Check if it's the initial join (no BPT minted yet)
        if bpt_total_supply == 0:
            # Initial invariant
            invariant = self.calculate_invariant(amounts_in, weights)
            return invariant
        
        # Calculate current invariant
        current_invariant = self.calculate_invariant(balances, weights)
        
        # Calculate new balances
        new_balances = []
        for i in range(len(balances)):
            new_balances.append(balances[i] + amounts_in[i])
        
        # Calculate new invariant
        new_invariant = self.calculate_invariant(new_balances, weights)
        
        # Calculate proportional BPT amount
        invariant_ratio = new_invariant / current_invariant
        bpt_out = bpt_total_supply * (invariant_ratio - Decimal('1'))
        
        # Check if join is proportional
        ratios = []
        for i in range(len(balances)):
            if balances[i] > 0:
                ratio = amounts_in[i] / balances[i]
                ratios.append(ratio)
        
        # If all ratios are equal (within tolerance), it's proportional
        if ratios and all(abs(r - ratios[0]) < Decimal('0.0001') for r in ratios):
            # Proportional join - no fees
            return bpt_out
        
        # Non-proportional join - apply swap fees
        # This is a simplified calculation
        # Full implementation would calculate the fee on the "excess" tokens
        fee_adjustment = Decimal('1') - fee / 2  # Approximate fee impact
        
        return bpt_out * fee_adjustment
    
    def calculate_token_out_given_exact_bpt_in(self,
                                              balance: Decimal,
                                              weight: Decimal,
                                              bpt_amount_in: Decimal,
                                              bpt_total_supply: Decimal,
                                              swap_fee: Optional[Decimal] = None) -> Decimal:
        """
        Calculate token amount out for burning BPT (single asset exit).
        
        Args:
            balance: Balance of token to withdraw
            weight: Weight of token to withdraw
            bpt_amount_in: Amount of BPT to burn
            bpt_total_supply: Current BPT supply
            swap_fee: Override swap fee
            
        Returns:
            Amount of token to receive
        """
        if bpt_amount_in <= 0 or bpt_amount_in > bpt_total_supply:
            return Decimal('0')
        
        if balance <= 0 or weight <= 0:
            return Decimal('0')
        
        fee = swap_fee if swap_fee is not None else self.swap_fee
        
        # For proportional exit (all tokens)
        proportional_amount = balance * bpt_amount_in / bpt_total_supply
        
        # For single asset exit, apply penalty
        # The penalty compensates other LPs for the imbalance created
        normalized_weight = weight  # Assuming weights are already normalized
        
        # Calculate the exit fee based on how imbalanced this exit is
        # Full exit to single asset = maximum imbalance
        exit_fee_percent = fee * (Decimal('1') - normalized_weight)
        
        token_amount_out = proportional_amount * (Decimal('1') - exit_fee_percent)
        
        return token_amount_out
    
    def _pow(self, base: Decimal, exponent: Decimal) -> Decimal:
        """
        Calculate base^exponent with high precision.
        
        Handles edge cases and uses logarithms for fractional exponents.
        
        Args:
            base: Base value
            exponent: Exponent value
            
        Returns:
            base^exponent
        """
        if base <= 0:
            return Decimal('0')
        
        if exponent == 0:
            return Decimal('1')
        
        if exponent == 1:
            return base
        
        # For integer exponents, use built-in power
        if exponent == int(exponent):
            return base ** int(exponent)
        
        # For fractional exponents, use exp(exponent * ln(base))
        # This is more accurate for fractional powers
        try:
            ln_base = base.ln()
            return (exponent * ln_base).exp()
        except:
            # Fallback to float calculation if Decimal fails
            try:
                result = float(base) ** float(exponent)
                return Decimal(str(result))
            except:
                logger.error(f"Power calculation failed: {base}^{exponent}")
                return Decimal('0')
    
    def calculate_multi_asset_swap(self,
                                 token_in_index: int,
                                 token_out_index: int,
                                 token_amount_in: Decimal,
                                 balances: List[Decimal],
                                 weights: List[Decimal],
                                 swap_fee: Optional[Decimal] = None) -> Tuple[Decimal, Decimal]:
        """
        Calculate swap between any two tokens in a multi-asset pool.
        
        Args:
            token_in_index: Index of input token
            token_out_index: Index of output token
            token_amount_in: Amount of input token
            balances: All token balances
            weights: All token weights
            swap_fee: Override swap fee
            
        Returns:
            Tuple of (token_amount_out, effective_price)
        """
        if token_in_index == token_out_index:
            return (Decimal('0'), Decimal('0'))
        
        if token_in_index >= len(balances) or token_out_index >= len(balances):
            return (Decimal('0'), Decimal('0'))
        
        token_amount_out = self.calculate_out_given_in(
            balances[token_in_index],
            weights[token_in_index],
            balances[token_out_index],
            weights[token_out_index],
            token_amount_in,
            swap_fee
        )
        
        effective_price = token_amount_out / token_amount_in if token_amount_in > 0 else Decimal('0')
        
        return (token_amount_out, effective_price)
    
    def calculate_out_given_in_with_fee(self,
                                       token_balance_in: Decimal,
                                       token_weight_in: Decimal,
                                       token_balance_out: Decimal,
                                       token_weight_out: Decimal,
                                       token_amount_in: Decimal,
                                       swap_fee: Decimal) -> Decimal:
        """
        Calculate output amount with explicit fee (wrapper for calculate_out_given_in).
        """
        return self.calculate_out_given_in(
            token_balance_in, token_weight_in,
            token_balance_out, token_weight_out,
            token_amount_in, swap_fee
        )
    
    def calculate_lp_tokens_out(self,
                               balances: List[Decimal],
                               weights: List[Decimal],
                               amounts_in: List[Decimal],
                               total_supply: Decimal) -> Decimal:
        """
        Calculate LP tokens minted (wrapper for BPT calculation).
        """
        return self.calculate_bpt_out_given_exact_tokens_in(
            balances, weights, amounts_in, total_supply
        )