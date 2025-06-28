"""
MEV Opportunity Data Models.

Defines data structures for representing different types of MEV opportunities
discovered through mempool monitoring and transaction analysis.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class MEVOpportunityType(str, Enum):
    """Types of MEV opportunities."""
    BACK_RUN = "back_run"           # Back-run opportunity after target transaction
    SANDWICH = "sandwich"           # Sandwich attack opportunity
    ARBITRAGE = "arbitrage"         # Cross-DEX arbitrage opportunity
    LIQUIDATION = "liquidation"     # Liquidation opportunity
    JIT_LIQUIDITY = "jit_liquidity" # Just-in-time liquidity provision
    FRONTRUN = "frontrun"           # General frontrunning opportunity


class OpportunityStatus(str, Enum):
    """Status of MEV opportunity lifecycle."""
    DETECTED = "detected"           # Just discovered
    VALIDATED = "validated"         # Validated as profitable
    QUEUED = "queued"              # Queued for execution
    EXECUTING = "executing"         # Currently being executed
    COMPLETED = "completed"         # Successfully completed
    FAILED = "failed"              # Execution failed
    EXPIRED = "expired"            # Opportunity expired
    CANCELLED = "cancelled"        # Manually cancelled


@dataclass
class TransactionData:
    """Data about a target transaction for MEV opportunities."""
    hash: str
    to: str
    from_address: str
    value: int
    gas_price: int
    gas_limit: int
    data: str
    nonce: int
    
    # Decoded transaction data
    decoded_function: Optional[str] = None
    decoded_params: Optional[Dict[str, Any]] = None
    
    # Market impact analysis
    estimated_price_impact: Optional[float] = None
    affected_pools: List[str] = field(default_factory=list)
    affected_tokens: List[str] = field(default_factory=list)


class MEVOpportunity(BaseModel):
    """Base model for MEV opportunities."""
    
    opportunity_id: str = Field(..., description="Unique opportunity identifier")
    opportunity_type: MEVOpportunityType = Field(..., description="Type of MEV opportunity")
    status: OpportunityStatus = Field(default=OpportunityStatus.DETECTED, description="Current status")
    
    # Target transaction info
    target_transaction_hash: str = Field(..., description="Hash of target transaction")
    target_block_number: Optional[int] = Field(None, description="Target block number")
    
    # Profitability analysis
    estimated_profit_usd: float = Field(..., description="Estimated profit in USD", ge=0)
    confidence_score: float = Field(..., description="Confidence in profit estimate (0-1)", ge=0, le=1)
    required_capital_usd: float = Field(..., description="Required capital in USD", ge=0)
    
    # Execution parameters
    max_gas_price: int = Field(..., description="Maximum gas price willing to pay", ge=0)
    execution_deadline: float = Field(..., description="Unix timestamp deadline for execution")
    requires_flashloan: bool = Field(default=False, description="Whether flashloan is required")
    
    # Chain and protocol info
    chain_id: int = Field(..., description="Blockchain network ID")
    involved_protocols: List[str] = Field(default_factory=list, description="DeFi protocols involved")
    involved_tokens: List[str] = Field(default_factory=list, description="Token addresses involved")
    
    # Timing
    detected_at: float = Field(default_factory=time.time, description="When opportunity was detected")
    expires_at: Optional[float] = Field(None, description="When opportunity expires")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional opportunity data")
    
    def is_expired(self) -> bool:
        """Check if opportunity has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def time_to_expiry(self) -> Optional[float]:
        """Get seconds until expiry, None if no expiry set."""
        if self.expires_at is None:
            return None
        return max(0, self.expires_at - time.time())
    
    def profit_to_capital_ratio(self) -> float:
        """Calculate profit to capital ratio."""
        if self.required_capital_usd == 0:
            return float('inf')
        return self.estimated_profit_usd / self.required_capital_usd


class BackRunOpportunity(MEVOpportunity):
    """Specific model for back-running opportunities."""
    
    opportunity_type: MEVOpportunityType = Field(default=MEVOpportunityType.BACK_RUN, frozen=True)
    
    # Back-run specific fields
    source_asset: str = Field(..., description="Asset to use for back-run")
    target_asset: str = Field(..., description="Asset to receive from back-run")
    optimal_amount: float = Field(..., description="Optimal amount for back-run", ge=0)
    
    # Price impact from target transaction
    expected_price_movement: float = Field(..., description="Expected price movement from target tx")
    pool_address: str = Field(..., description="Pool/DEX address for back-run")
    
    # Execution strategy
    execution_method: str = Field(default="flashbots", description="Preferred execution method")
    bundle_with_target: bool = Field(default=True, description="Bundle with target transaction")
    
    def calculate_effective_profit(self, gas_cost_usd: float) -> float:
        """Calculate profit after gas costs."""
        return max(0, self.estimated_profit_usd - gas_cost_usd)


class SandwichOpportunity(MEVOpportunity):
    """Specific model for sandwich attack opportunities."""
    
    opportunity_type: MEVOpportunityType = Field(default=MEVOpportunityType.SANDWICH, frozen=True)
    
    # Sandwich specific fields
    frontrun_transaction: Dict[str, Any] = Field(..., description="Frontrun transaction details")
    backrun_transaction: Dict[str, Any] = Field(..., description="Backrun transaction details")
    
    # Victim transaction analysis
    victim_slippage_tolerance: float = Field(..., description="Victim's slippage tolerance")
    extractable_value: float = Field(..., description="Value extractable from victim", ge=0)
    
    # Pool analysis
    pool_liquidity: float = Field(..., description="Pool liquidity in USD", ge=0)
    optimal_sandwich_size: float = Field(..., description="Optimal sandwich size", ge=0)


class ArbitrageOpportunity(MEVOpportunity):
    """Specific model for arbitrage opportunities created by transactions."""
    
    opportunity_type: MEVOpportunityType = Field(default=MEVOpportunityType.ARBITRAGE, frozen=True)
    
    # Arbitrage specific fields
    source_dex: str = Field(..., description="Source DEX for arbitrage")
    target_dex: str = Field(..., description="Target DEX for arbitrage")
    arbitrage_token: str = Field(..., description="Token to arbitrage")
    
    # Price analysis
    price_difference: float = Field(..., description="Price difference between DEXs", ge=0)
    optimal_arbitrage_amount: float = Field(..., description="Optimal arbitrage amount", ge=0)
    
    # Route analysis
    arbitrage_path: List[str] = Field(..., description="Complete arbitrage path")
    execution_complexity: int = Field(..., description="Number of transactions required", ge=1)


class LiquidationOpportunity(MEVOpportunity):
    """Specific model for liquidation opportunities."""
    
    opportunity_type: MEVOpportunityType = Field(default=MEVOpportunityType.LIQUIDATION, frozen=True)
    
    # Liquidation specific fields
    protocol: str = Field(..., description="Lending protocol (e.g., Aave, Compound)")
    borrower_address: str = Field(..., description="Address of borrower to liquidate")
    collateral_token: str = Field(..., description="Collateral token address")
    debt_token: str = Field(..., description="Debt token address")
    
    # Financial analysis
    collateral_value: float = Field(..., description="Collateral value in USD", ge=0)
    debt_value: float = Field(..., description="Debt value in USD", ge=0)
    liquidation_bonus: float = Field(..., description="Liquidation bonus percentage", ge=0)
    health_factor: float = Field(..., description="Current health factor", ge=0)


class JITLiquidityOpportunity(MEVOpportunity):
    """Specific model for just-in-time liquidity provision opportunities."""
    
    opportunity_type: MEVOpportunityType = Field(default=MEVOpportunityType.JIT_LIQUIDITY, frozen=True)
    
    # JIT specific fields
    pool_address: str = Field(..., description="Pool address for JIT liquidity")
    tick_range: tuple[int, int] = Field(..., description="Optimal tick range for liquidity")
    liquidity_amount: float = Field(..., description="Amount of liquidity to provide", ge=0)
    
    # Strategy analysis
    expected_fees: float = Field(..., description="Expected fees from JIT strategy", ge=0)
    hold_duration: int = Field(..., description="Expected hold duration in blocks", ge=1)
    removal_strategy: str = Field(..., description="Strategy for removing liquidity")


# Union type for all opportunity types
AnyMEVOpportunity = Union[
    BackRunOpportunity,
    SandwichOpportunity, 
    ArbitrageOpportunity,
    LiquidationOpportunity,
    JITLiquidityOpportunity
]


class OpportunityQueue(BaseModel):
    """Queue for managing MEV opportunities by priority."""
    
    opportunities: List[MEVOpportunity] = Field(default_factory=list)
    max_size: int = Field(default=1000, description="Maximum queue size")
    
    def add_opportunity(self, opportunity: MEVOpportunity) -> bool:
        """Add opportunity to queue, sorted by profit potential."""
        if len(self.opportunities) >= self.max_size:
            # Remove lowest profit opportunity if queue is full
            self.opportunities.sort(key=lambda x: x.estimated_profit_usd)
            if opportunity.estimated_profit_usd > self.opportunities[0].estimated_profit_usd:
                self.opportunities.pop(0)
            else:
                return False
        
        self.opportunities.append(opportunity)
        # Keep queue sorted by profit (descending)
        self.opportunities.sort(key=lambda x: x.estimated_profit_usd, reverse=True)
        return True
    
    def get_next_opportunity(self) -> Optional[MEVOpportunity]:
        """Get next highest profit opportunity."""
        # Filter expired opportunities
        self.opportunities = [op for op in self.opportunities if not op.is_expired()]
        
        if not self.opportunities:
            return None
            
        return self.opportunities.pop(0)
    
    def get_opportunities_by_type(self, opportunity_type: MEVOpportunityType) -> List[MEVOpportunity]:
        """Get all opportunities of a specific type."""
        return [op for op in self.opportunities if op.opportunity_type == opportunity_type]
    
    def cleanup_expired(self) -> int:
        """Remove expired opportunities and return count removed."""
        initial_count = len(self.opportunities)
        self.opportunities = [op for op in self.opportunities if not op.is_expired()]
        return initial_count - len(self.opportunities)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        if not self.opportunities:
            return {
                "total_opportunities": 0,
                "total_estimated_profit": 0.0,
                "average_profit": 0.0,
                "opportunities_by_type": {}
            }
        
        total_profit = sum(op.estimated_profit_usd for op in self.opportunities)
        opportunities_by_type = {}
        
        for op in self.opportunities:
            op_type = op.opportunity_type.value
            if op_type not in opportunities_by_type:
                opportunities_by_type[op_type] = 0
            opportunities_by_type[op_type] += 1
        
        return {
            "total_opportunities": len(self.opportunities),
            "total_estimated_profit": total_profit,
            "average_profit": total_profit / len(self.opportunities),
            "opportunities_by_type": opportunities_by_type,
            "highest_profit": self.opportunities[0].estimated_profit_usd if self.opportunities else 0.0
        }