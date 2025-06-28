"""Core graph data structures for the yield arbitrage system."""
import time
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator


class EdgeType(str, Enum):
    """Types of edges representing different DeFi operations."""
    TRADE = "TRADE"
    SPLIT = "SPLIT"
    COMBINE = "COMBINE"
    BRIDGE = "BRIDGE"
    LEND = "LEND"
    BORROW = "BORROW"
    STAKE = "STAKE"
    WAIT = "WAIT"
    SHORT = "SHORT"
    FLASH_LOAN = "FLASH_LOAN"
    FLASH_REPAY = "FLASH_REPAY"
    BACK_RUN = "BACK_RUN"


class EdgeExecutionProperties(BaseModel):
    """Properties that define how an edge can be executed."""
    
    # Core execution constraints
    supports_synchronous: bool = Field(
        True,
        description="Whether this edge can execute in the same transaction as others"
    )
    requires_time_delay: Optional[int] = Field(
        None,
        description="Required time delay in seconds before edge can be executed",
        ge=0
    )
    requires_bridge: bool = Field(
        False,
        description="Whether this edge requires cross-chain bridge functionality"
    )
    requires_capital_holding: bool = Field(
        False,
        description="Whether this edge requires holding capital over time (e.g., lending positions)"
    )
    
    # Risk and MEV properties
    max_slippage: float = Field(
        0.05,
        description="Maximum acceptable slippage for this edge",
        ge=0.0,
        le=1.0
    )
    mev_sensitivity: float = Field(
        0.5,
        description="MEV sensitivity score from 0 (no MEV risk) to 1 (high MEV risk)",
        ge=0.0,
        le=1.0
    )
    supports_private_mempool: bool = Field(
        True,
        description="Whether this edge can be executed via private mempools (Flashbots, etc.)"
    )
    
    # Gas and cost properties
    gas_estimate: int = Field(
        100000,
        description="Estimated gas cost for executing this edge",
        ge=0
    )
    requires_approval: bool = Field(
        True,
        description="Whether this edge requires token approval transactions"
    )
    
    # Liquidity and market properties
    min_liquidity_required: float = Field(
        10000.0,
        description="Minimum USD liquidity required for this edge to be viable",
        ge=0
    )
    max_impact_allowed: float = Field(
        0.01,
        description="Maximum allowed price impact (0.01 = 1%)",
        ge=0.0,
        le=1.0
    )


class EdgeConstraints(BaseModel):
    """Constraints for edge execution."""
    min_input_amount: Optional[float] = Field(
        None, 
        description="Minimum input amount for this edge",
        ge=0
    )
    max_input_amount: Optional[float] = Field(
        None,
        description="Maximum input amount for this edge", 
        ge=0
    )

    @field_validator('max_input_amount')
    @classmethod
    def max_greater_than_min(cls, v, info):
        """Ensure max is greater than min if both are set."""
        if v is not None and info.data.get('min_input_amount') is not None:
            if v < info.data['min_input_amount']:
                raise ValueError('max_input_amount must be >= min_input_amount')
        return v


class EdgeState(BaseModel):
    """Dynamic state of an edge."""
    conversion_rate: Optional[float] = Field(
        None,
        description="Current conversion rate from source to target asset",
        ge=0
    )
    liquidity_usd: Optional[float] = Field(
        None,
        description="Available liquidity in USD",
        ge=0
    )
    gas_cost_usd: Optional[float] = Field(
        None,
        description="Estimated gas cost in USD",
        ge=0
    )
    delta_exposure: Optional[Dict[str, float]] = Field(
        None,
        description="Market exposure by asset (e.g., {'ETH': 1.0})"
    )
    last_updated_timestamp: Optional[float] = Field(
        None,
        description="Unix timestamp of last update"
    )
    confidence_score: float = Field(
        1.0,
        description="Confidence in the data quality (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if the edge state is stale."""
        if self.last_updated_timestamp is None:
            return True
        return time.time() - self.last_updated_timestamp > max_age_seconds

    def mark_updated(self) -> None:
        """Mark the edge state as recently updated."""
        self.last_updated_timestamp = time.time()


class YieldGraphEdge(BaseModel):
    """Represents a transformation between two assets in the graph."""
    edge_id: str = Field(
        ...,
        description="Unique identifier (e.g., ETH_MAINNET_UNISWAPV3_TRADE_WETH_USDC)",
        min_length=1
    )
    source_asset_id: str = Field(
        ...,
        description="Source asset ID (e.g., ETH_MAINNET_0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2)",
        min_length=1
    )
    target_asset_id: str = Field(
        ...,
        description="Target asset ID",
        min_length=1
    )
    edge_type: EdgeType = Field(
        ...,
        description="Type of operation this edge represents"
    )
    protocol_name: str = Field(
        ...,
        description="Name of the protocol (e.g., UniswapV3, Aave)",
        min_length=1
    )
    chain_name: str = Field(
        ...,
        description="Name of the blockchain (e.g., ethereum, arbitrum)",
        min_length=1
    )
    constraints: EdgeConstraints = Field(
        default_factory=EdgeConstraints,
        description="Constraints for this edge"
    )
    state: EdgeState = Field(
        default_factory=EdgeState,
        description="Current dynamic state of this edge"
    )
    execution_properties: EdgeExecutionProperties = Field(
        default_factory=EdgeExecutionProperties,
        description="Properties defining how this edge can be executed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for protocol-specific information"
    )

    @field_validator('target_asset_id')
    @classmethod
    def target_different_from_source(cls, v, info):
        """Ensure target asset is different from source asset."""
        if v == info.data.get('source_asset_id'):
            raise ValueError('target_asset_id must be different from source_asset_id')
        return v

    def calculate_output(self, input_amount: float, current_state: Optional[EdgeState] = None) -> Dict:
        """Calculate output amount for given input."""
        state = current_state or self.state
        
        if state.conversion_rate is None:
            return {
                "output_amount": 0.0,
                "error": "Missing conversion rate"
            }
        
        if input_amount <= 0:
            return {
                "output_amount": 0.0,
                "error": "Input amount must be positive"
            }
        
        # Check constraints
        if self.constraints.min_input_amount and input_amount < self.constraints.min_input_amount:
            return {
                "output_amount": 0.0,
                "error": f"Input amount {input_amount} below minimum {self.constraints.min_input_amount}"
            }
        
        if self.constraints.max_input_amount and input_amount > self.constraints.max_input_amount:
            return {
                "output_amount": 0.0,
                "error": f"Input amount {input_amount} above maximum {self.constraints.max_input_amount}"
            }
        
        # Calculate base output
        output = input_amount * state.conversion_rate
        
        # Apply simplified slippage and fee deduction
        # This is a placeholder - real implementation would be protocol-specific
        slippage_fee_factor = 0.997  # 0.3% fee assumption
        final_output = output * slippage_fee_factor
        
        return {
            "output_amount": final_output,
            "gas_cost_usd": state.gas_cost_usd or 5.0,
            "effective_rate": final_output / input_amount,
            "confidence": state.confidence_score
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "edge_id": "ETH_MAINNET_UNISWAPV3_TRADE_WETH_USDC",
                "source_asset_id": "ETH_MAINNET_0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "target_asset_id": "ETH_MAINNET_0xA0b86a33E6441b9435B654f6D26Cc98b6e1D0a3A",
                "edge_type": "TRADE",
                "protocol_name": "UniswapV3",
                "chain_name": "ethereum"
            }
        }
    }


class UniversalYieldGraph:
    """Graph representation of the DeFi ecosystem."""
    
    def __init__(self):
        """Initialize empty graph."""
        self.nodes: set[str] = set()  # asset_ids
        self.edges: Dict[str, YieldGraphEdge] = {}  # edge_id -> YieldGraphEdge
        self.adj: Dict[str, List[YieldGraphEdge]] = defaultdict(list)  # source_asset -> [edges]

    def add_edge(self, edge: YieldGraphEdge) -> bool:
        """Add an edge to the graph."""
        if edge.edge_id in self.edges:
            return False  # Edge already exists
        
        self.edges[edge.edge_id] = edge
        self.nodes.add(edge.source_asset_id)
        self.nodes.add(edge.target_asset_id)
        self.adj[edge.source_asset_id].append(edge)
        return True

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        del self.edges[edge_id]
        
        # Remove from adjacency list
        self.adj[edge.source_asset_id].remove(edge)
        
        # Clean up empty adjacency lists and orphaned nodes
        if not self.adj[edge.source_asset_id]:
            del self.adj[edge.source_asset_id]
        
        # Check if nodes are orphaned
        self._cleanup_orphaned_nodes()
        return True

    def get_edges_from(self, asset_id: str) -> List[YieldGraphEdge]:
        """Get all outgoing edges from an asset."""
        return self.adj.get(asset_id, [])

    def get_edges_to(self, asset_id: str) -> List[YieldGraphEdge]:
        """Get all incoming edges to an asset."""
        return [edge for edge in self.edges.values() if edge.target_asset_id == asset_id]

    def get_edge(self, edge_id: str) -> Optional[YieldGraphEdge]:
        """Get edge by ID."""
        return self.edges.get(edge_id)

    def update_edge_state(self, edge_id: str, new_state: EdgeState) -> bool:
        """Update the state of an edge."""
        if edge_id not in self.edges:
            return False
        
        self.edges[edge_id].state = new_state
        new_state.mark_updated()
        return True

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        total_edges = len(self.edges)
        active_edges = sum(1 for edge in self.edges.values() 
                          if edge.state.conversion_rate is not None)
        stale_edges = sum(1 for edge in self.edges.values() 
                         if edge.state.is_stale())
        
        edge_types = defaultdict(int)
        protocols = defaultdict(int)
        chains = defaultdict(int)
        
        for edge in self.edges.values():
            edge_types[edge.edge_type] += 1
            protocols[edge.protocol_name] += 1
            chains[edge.chain_name] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": total_edges,
            "active_edges": active_edges,
            "stale_edges": stale_edges,
            "edge_types": dict(edge_types),
            "protocols": dict(protocols),
            "chains": dict(chains)
        }

    def _cleanup_orphaned_nodes(self) -> None:
        """Remove nodes that have no incoming or outgoing edges."""
        connected_nodes = set()
        for edge in self.edges.values():
            connected_nodes.add(edge.source_asset_id)
            connected_nodes.add(edge.target_asset_id)
        
        self.nodes = connected_nodes

    def __len__(self) -> int:
        """Return number of edges in the graph."""
        return len(self.edges)

    def __contains__(self, edge_id: str) -> bool:
        """Check if edge exists in graph."""
        return edge_id in self.edges


class FlashLoanEdge(YieldGraphEdge):
    """Special edge type for flash loans that provide capital but require synchronous execution."""
    
    provider: str = Field(..., description="Flash loan provider (e.g., 'aave', 'balancer')")
    asset: str = Field(..., description="Asset symbol (e.g., 'WETH', 'USDC')")
    max_amount: float = Field(..., description="Maximum available flash loan amount", gt=0)
    fee_percentage: float = Field(0.0009, description="Flash loan fee as percentage", ge=0, le=1)
    
    def __init__(
        self,
        chain_name: str,
        provider: str,
        asset: str,
        max_amount: float,
        fee_percentage: float = 0.0009,  # 0.09% typical Aave flash loan fee
        **kwargs
    ):
        """Initialize a flash loan edge.
        
        Args:
            chain_name: The blockchain network (e.g., 'ethereum')
            provider: Flash loan provider (e.g., 'aave', 'balancer')
            asset: The asset symbol (e.g., 'WETH', 'USDC')
            max_amount: Maximum available flash loan amount
            fee_percentage: Flash loan fee as percentage (0.0009 = 0.09%)
        """
        # Create asset IDs for flash loan
        source_asset_id = f"{chain_name.upper()}_FLASH_{asset.upper()}"
        target_asset_id = f"{chain_name.upper()}_{asset.upper()}"
        
        # Flash loans require synchronous execution
        execution_props = EdgeExecutionProperties(
            supports_synchronous=True,
            requires_time_delay=None,
            requires_capital_holding=False,
            mev_sensitivity=0.3,  # Lower MEV risk for flash loans
            gas_estimate=150000,  # Higher gas for flash loan setup
            requires_approval=False,  # Flash loans don't need approval
            min_liquidity_required=max_amount
        )
        
        super().__init__(
            edge_id=f"{chain_name.upper()}_{provider.upper()}_FLASH_LOAN_{asset.upper()}",
            source_asset_id=source_asset_id,
            target_asset_id=target_asset_id,
            edge_type=EdgeType.FLASH_LOAN,
            protocol_name=provider,
            chain_name=chain_name.lower(),
            execution_properties=execution_props,
            provider=provider,
            asset=asset,
            max_amount=max_amount,
            fee_percentage=fee_percentage,
            **kwargs
        )
    
    def calculate_output(self, input_amount: float, current_state: Optional[EdgeState] = None) -> Dict:
        """Calculate flash loan output (same as input, fee charged on repayment)."""
        if input_amount <= 0:
            return {
                "output_amount": 0.0,
                "error": "Input amount must be positive"
            }
        
        if input_amount > self.max_amount:
            return {
                "output_amount": 0.0,
                "error": f"Requested amount {input_amount} exceeds max flash loan {self.max_amount}"
            }
        
        # Flash loan provides the full amount but requires repayment with fee
        fee_amount = input_amount * self.fee_percentage
        
        return {
            "output_amount": input_amount,  # Flash loan provides full amount
            "flash_loan_fee": fee_amount,
            "repayment_required": input_amount + fee_amount,
            "gas_cost_usd": self.execution_properties.gas_estimate * 0.00002,  # Estimate gas cost
            "effective_rate": 1.0,  # 1:1 ratio for flash loan
            "confidence": 1.0
        }


class BackRunEdge(YieldGraphEdge):
    """Special edge type for MEV back-running opportunities."""
    
    target_transaction: str = Field(..., description="Transaction hash or identifier to back-run")
    expected_profit: float = Field(..., description="Expected profit from the back-run in USD", ge=0)
    
    def __init__(
        self,
        chain_name: str,
        target_transaction: str,
        source_asset: str,
        target_asset: str,
        expected_profit: float,
        **kwargs
    ):
        """Initialize a back-running edge.
        
        Args:
            chain_name: The blockchain network
            target_transaction: Transaction hash or identifier to back-run
            source_asset: Asset to use for back-running
            target_asset: Asset to receive from back-running
            expected_profit: Expected profit from the back-run in USD
        """
        # Back-running edges are time-sensitive and have no MEV risk themselves
        execution_props = EdgeExecutionProperties(
            supports_synchronous=True,
            requires_time_delay=None,
            requires_capital_holding=False,
            mev_sensitivity=0.0,  # No frontrun risk for back-runs
            gas_estimate=200000,  # Higher gas for MEV operations
            requires_approval=True,
            min_liquidity_required=expected_profit * 10  # Require 10x profit in liquidity
        )
        
        super().__init__(
            edge_id=f"{chain_name.upper()}_BACKRUN_{target_transaction[:8]}_{source_asset}_{target_asset}",
            source_asset_id=f"{chain_name.upper()}_{source_asset}",
            target_asset_id=f"{chain_name.upper()}_{target_asset}",
            edge_type=EdgeType.BACK_RUN,
            protocol_name="MEV_BACKRUN",
            chain_name=chain_name.lower(),
            execution_properties=execution_props,
            target_transaction=target_transaction,
            expected_profit=expected_profit,
            **kwargs
        )
    
    def calculate_output(self, input_amount: float, current_state: Optional[EdgeState] = None) -> Dict:
        """Calculate back-run output based on expected profit."""
        if input_amount <= 0:
            return {
                "output_amount": 0.0,
                "error": "Input amount must be positive"
            }
        
        # Back-run profit is typically a percentage of the transaction size
        profit_rate = self.expected_profit / input_amount if input_amount > 0 else 0
        output_amount = input_amount * (1 + profit_rate)
        
        return {
            "output_amount": output_amount,
            "expected_profit_usd": self.expected_profit,
            "target_transaction": self.target_transaction,
            "gas_cost_usd": self.execution_properties.gas_estimate * 0.00002,
            "effective_rate": output_amount / input_amount,
            "confidence": 0.7  # Back-runs have inherent uncertainty
        }