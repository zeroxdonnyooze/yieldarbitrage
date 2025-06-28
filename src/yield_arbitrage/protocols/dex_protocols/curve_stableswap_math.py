"""
Curve StableSwap Math Implementation.

Implements the StableSwap invariant with Newton's method for solving.
This is designed for stable assets (stablecoins, ETH/stETH, etc.) and
provides extremely low slippage near equilibrium.
"""
from decimal import Decimal, getcontext
from typing import List, Tuple, Optional
import logging

# Set high precision for iterative calculations
getcontext().prec = 50

logger = logging.getLogger(__name__)


class CurveStableSwapMath:
    """
    Exact implementation of Curve's StableSwap AMM math.
    
    The StableSwap invariant combines constant sum and constant product:
    A·n^n·Σx_i + D = A·D·n^n + D^(n+1)/(n^n·Πx_i)
    
    Where:
    - A is the amplification coefficient
    - n is the number of tokens
    - x_i are the token balances
    - D is the invariant
    """
    
    def __init__(self, amplification_coefficient: Decimal = Decimal('100')):
        """
        Initialize Curve StableSwap math.
        
        Args:
            amplification_coefficient: The A parameter (higher = more like constant sum)
        """
        self.A = amplification_coefficient
        self.fee_rate = Decimal('0.0004')  # 0.04% default
        self.admin_fee_rate = Decimal('0.5')  # 50% of fees go to admin
        self.convergence_precision = Decimal('0.0000000001')  # 1e-10
        self.max_iterations = 255
        
    def get_D(self, balances: List[Decimal], amp: Optional[Decimal] = None) -> Decimal:
        """
        Calculate the StableSwap invariant D.
        
        Uses Newton's method to solve:
        f(D) = D^(n+1)/(n^n·Πx_i) + (A·n^n - 1)·D - A·n^n·Σx_i = 0
        
        Args:
            balances: List of token balances
            amp: Amplification coefficient (uses self.A if not provided)
            
        Returns:
            The invariant D
        """
        if not balances or any(b <= 0 for b in balances):
            return Decimal('0')
        
        n = len(balances)
        if n < 2:
            return Decimal('0')
        
        A = amp if amp is not None else self.A
        
        # Initial guess for D
        S = sum(balances)
        D = S
        Ann = A * n ** n
        
        # Newton's method iteration
        for _ in range(self.max_iterations):
            D_P = D
            for balance in balances:
                D_P = D_P * D / (n * balance)
            
            D_prev = D
            
            # Newton's formula: D_next = (Ann·S + n·D_P)·D / ((Ann - 1)·D + (n + 1)·D_P)
            numerator = (Ann * S + n * D_P) * D
            denominator = (Ann - 1) * D + (n + 1) * D_P
            
            if denominator == 0:
                return Decimal('0')
            
            D = numerator / denominator
            
            # Check convergence
            if abs(D - D_prev) <= self.convergence_precision:
                return D
        
        # Failed to converge
        logger.warning(f"get_D failed to converge after {self.max_iterations} iterations")
        return D
    
    def get_y(self, i: int, j: int, x: Decimal, balances: List[Decimal]) -> Decimal:
        """
        Calculate the balance of token j after swapping x amount of token i.
        
        Solves the invariant equation for y_j given new x_i.
        
        Args:
            i: Index of input token
            j: Index of output token
            x: New balance of token i (after adding input)
            balances: Current token balances
            
        Returns:
            New balance of token j
        """
        n = len(balances)
        if i == j or i >= n or j >= n or i < 0 or j < 0:
            return Decimal('0')
        
        # Calculate D
        D = self.get_D(balances)
        if D == 0:
            return Decimal('0')
        
        # Create new balances array with x at position i
        new_balances = balances.copy()
        new_balances[i] = x
        
        Ann = self.A * (n ** n)
        
        # Use Newton's method for all cases (more reliable)
        return self._get_y_newton(i, j, x, balances, D, Ann)
    
    def _get_y_newton(self, i: int, j: int, x: Decimal, balances: List[Decimal], 
                     D: Decimal, Ann: Decimal) -> Decimal:
        """
        Use Newton's method to solve for y in the StableSwap invariant.
        
        Solves: A*n^n*S + D = A*D*n^n + D^(n+1)/(n^n*P*y) for y
        where S = sum of balances, P = product of balances except y
        """
        n = len(balances)
        
        # Set up the new state with x at position i
        new_balances = balances.copy()
        new_balances[i] = x
        
        # Sum all balances except j
        S = sum(new_balances[k] for k in range(n) if k != j)
        
        # Product of all balances except j  
        P = Decimal('1')
        for k in range(n):
            if k != j:
                P *= new_balances[k]
        
        # Initial guess
        y = D / n
        
        # Newton's method to solve the invariant
        for iteration in range(self.max_iterations):
            y_prev = y
            
            if y <= Decimal('0'):
                y = D / n
                continue
            
            # Calculate D^(n+1) / (n^n * P * y)
            if P <= 0:
                return Decimal('0')
                
            K0 = D ** (n + 1) / (n ** n * P * y)
            
            # Function: f(y) = A*n^n*(S+y) + D - A*D*n^n - K0 = 0
            f = Ann * (S + y) + D - Ann * D - K0
            
            # Derivative: f'(y) = A*n^n + K0/y
            f_prime = Ann + K0 / y
            
            if abs(f_prime) < self.convergence_precision:
                break
            
            # Newton's update
            y_new = y - f / f_prime
            
            # Ensure y stays positive and reasonable
            if y_new <= Decimal('0'):
                y_new = y / 2
            elif y_new > balances[j] * 2:  # Sanity check
                y_new = balances[j]
            
            # Check convergence
            if abs(y_new - y) <= self.convergence_precision:
                y = y_new
                break
            
            y = y_new
        
        return max(Decimal('0'), y)
    
    def get_dy(self, i: int, j: int, dx: Decimal, balances: List[Decimal],
               include_fee: bool = True) -> Decimal:
        """
        Calculate output amount for a given input amount.
        
        Args:
            i: Index of input token
            j: Index of output token  
            dx: Amount of input token
            balances: Current token balances
            include_fee: Whether to include trading fee
            
        Returns:
            Amount of output token
        """
        if dx <= 0 or i == j or i >= len(balances) or j >= len(balances) or i < 0 or j < 0:
            return Decimal('0')
        
        # Calculate new balance of token i
        x_new = balances[i] + dx
        
        # Calculate new balance of token j
        y_new = self.get_y(i, j, x_new, balances)
        
        if y_new >= balances[j]:
            return Decimal('0')
        
        # Calculate output amount
        dy = balances[j] - y_new
        
        # Apply fee if requested
        if include_fee:
            dy = dy * (Decimal('1') - self.fee_rate)
        
        return dy
    
    def get_exchange_rate(self, i: int, j: int, balances: List[Decimal]) -> Decimal:
        """
        Get the current exchange rate between two tokens.
        
        Args:
            i: Index of input token
            j: Index of output token
            balances: Current token balances
            
        Returns:
            Exchange rate (output per input)
        """
        # Use a small amount to minimize price impact
        dx = min(balances[i] * Decimal('0.0001'), Decimal('1'))
        dy = self.get_dy(i, j, dx, balances, include_fee=False)
        
        if dx == 0:
            return Decimal('0')
        
        return dy / dx
    
    def calculate_price_impact(self, i: int, j: int, dx: Decimal, 
                             balances: List[Decimal]) -> Decimal:
        """
        Calculate the price impact of a trade.
        
        Args:
            i: Index of input token
            j: Index of output token
            dx: Amount of input token
            balances: Current token balances
            
        Returns:
            Price impact as decimal (0.01 = 1%)
        """
        if dx <= 0:
            return Decimal('0')
        
        # Get spot rate
        spot_rate = self.get_exchange_rate(i, j, balances)
        
        # Get execution rate
        dy = self.get_dy(i, j, dx, balances, include_fee=False)
        execution_rate = dy / dx if dx > 0 else Decimal('0')
        
        if spot_rate <= 0:
            return Decimal('1')  # 100% impact
        
        # Price impact = 1 - (execution_rate / spot_rate)
        price_impact = Decimal('1') - (execution_rate / spot_rate)
        
        return max(Decimal('0'), price_impact)
    
    def get_virtual_price(self, balances: List[Decimal], 
                         total_supply: Decimal) -> Decimal:
        """
        Calculate the virtual price of the LP token.
        
        Virtual price = D / total_supply
        
        Args:
            balances: Current token balances
            total_supply: Total supply of LP tokens
            
        Returns:
            Virtual price
        """
        if total_supply <= 0:
            return Decimal('0')
        
        D = self.get_D(balances)
        return D / total_supply
    
    def calculate_withdraw_one_coin(self, token_amount: Decimal, i: int,
                                  balances: List[Decimal], 
                                  total_supply: Decimal) -> Decimal:
        """
        Calculate amount received when withdrawing to a single coin.
        
        Args:
            token_amount: Amount of LP tokens to burn
            i: Index of token to receive
            balances: Current token balances
            total_supply: Total supply of LP tokens
            
        Returns:
            Amount of token i to receive
        """
        if token_amount <= 0 or token_amount > total_supply:
            return Decimal('0')
        
        n = len(balances)
        
        # Calculate D before withdrawal
        D0 = self.get_D(balances)
        
        # Calculate D after withdrawal
        D1 = D0 * (total_supply - token_amount) / total_supply
        
        # Calculate new balances that satisfy D1
        # Start with proportional withdrawal
        new_balances = []
        for balance in balances:
            new_balance = balance * D1 / D0
            new_balances.append(new_balance)
        
        # Adjust balance i to maintain invariant
        # This requires solving for the new balance
        fee = self.fee_rate * n / (4 * (n - 1))
        
        # Iterate to find the correct balance
        for _ in range(self.max_iterations):
            y_prev = new_balances[i]
            
            # Set other balances
            for j in range(n):
                if j != i:
                    new_balances[j] = balances[j] - (balances[j] - new_balances[j]) * fee
            
            # Solve for balance i
            y = self._get_y_D(i, new_balances, D1)
            
            if abs(y - y_prev) <= self.convergence_precision:
                break
            
            new_balances[i] = y
        
        # Amount to withdraw
        dy = balances[i] - new_balances[i]
        
        return dy
    
    def _get_y_D(self, i: int, balances: List[Decimal], D: Decimal) -> Decimal:
        """
        Helper to calculate balance of token i given D and other balances.
        
        Args:
            i: Index of token to solve for
            balances: Token balances (balance[i] is ignored)
            D: Target invariant value
            
        Returns:
            Balance of token i
        """
        n = len(balances)
        Ann = self.A * n ** n
        
        # Calculate sum and product excluding i
        S = Decimal('0')
        P = Decimal('1')
        
        for j in range(n):
            if j != i:
                S += balances[j]
                P *= balances[j]
        
        # Initial guess
        y = D / n
        
        # Newton's method
        for _ in range(self.max_iterations):
            y_prev = y
            
            # K0 = D^(n+1) / (n^n * P)
            K0 = D ** (n + 1) / (n ** n * P) if P > 0 else Decimal('0')
            
            # Newton iteration
            S_plus_y = S + y
            
            numerator = y * y + K0 / Ann
            denominator = 2 * y + S_plus_y - D + K0 / Ann / y if y > 0 else Decimal('1')
            
            if denominator == 0:
                return Decimal('0')
            
            y = numerator / denominator
            
            if abs(y - y_prev) <= self.convergence_precision:
                return y
        
        return y
    
    def calculate_add_liquidity(self, amounts: List[Decimal],
                               balances: List[Decimal],
                               total_supply: Decimal) -> Decimal:
        """
        Calculate LP tokens minted when adding liquidity.
        
        Args:
            amounts: Amounts of each token to add
            balances: Current token balances
            total_supply: Current LP token supply
            
        Returns:
            Amount of LP tokens to mint
        """
        n = len(balances)
        if len(amounts) != n:
            return Decimal('0')
        
        # Calculate D before
        D0 = self.get_D(balances) if total_supply > 0 else Decimal('0')
        
        # New balances
        new_balances = []
        for i in range(n):
            new_balances.append(balances[i] + amounts[i])
        
        # Calculate D after
        D1 = self.get_D(new_balances)
        
        if total_supply == 0:
            # Initial deposit
            return D1
        else:
            # Subsequent deposits
            # Check if deposit is balanced
            ideal_amounts = []
            for i in range(n):
                ideal = balances[i] * D1 / D0 - balances[i]
                ideal_amounts.append(ideal)
            
            # Calculate fee for imbalanced deposits
            mint_amount = total_supply * (D1 - D0) / D0
            
            # Apply fee if imbalanced
            fee_applied = False
            for i in range(n):
                if abs(amounts[i] - ideal_amounts[i]) > self.convergence_precision:
                    fee_applied = True
                    break
            
            if fee_applied:
                # Simplified fee calculation
                fee = self.fee_rate * n / (4 * (n - 1))
                mint_amount = mint_amount * (Decimal('1') - fee)
            
            return mint_amount