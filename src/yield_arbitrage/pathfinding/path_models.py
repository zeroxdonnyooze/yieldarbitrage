"""Shared data models for pathfinding components."""
import time
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class PathStatus(str, Enum):
    """Status of a path during beam search."""
    ACTIVE = "active"          # Path is being explored
    COMPLETE = "complete"      # Path reached target asset
    PRUNED = "pruned"         # Path was pruned from beam
    INVALID = "invalid"       # Path became invalid (e.g., insufficient liquidity)
    TIMEOUT = "timeout"       # Path search timed out


@dataclass
class PathNode:
    """Represents a node in a potential arbitrage path."""
    asset_id: str
    amount: float
    gas_cost_accumulated: float = 0.0
    confidence_accumulated: float = 1.0
    edge_path: List[str] = None
    
    def __post_init__(self):
        if self.edge_path is None:
            self.edge_path = []


@dataclass
class SearchPath:
    """Represents a complete search path with scoring."""
    nodes: List[PathNode]
    total_score: float
    status: PathStatus = PathStatus.ACTIVE
    creation_time: float = None
    
    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = time.time()
    
    @property
    def start_asset(self) -> str:
        """Get the starting asset of the path."""
        return self.nodes[0].asset_id if self.nodes else ""
    
    @property
    def end_asset(self) -> str:
        """Get the ending asset of the path."""
        return self.nodes[-1].asset_id if self.nodes else ""
    
    @property
    def path_length(self) -> int:
        """Get the number of edges in the path."""
        return len(self.nodes) - 1 if len(self.nodes) > 1 else 0
    
    @property
    def total_gas_cost(self) -> float:
        """Get the total gas cost for the path."""
        return self.nodes[-1].gas_cost_accumulated if self.nodes else 0.0
    
    @property
    def final_amount(self) -> float:
        """Get the final amount after all conversions."""
        return self.nodes[-1].amount if self.nodes else 0.0
    
    @property
    def net_profit(self) -> float:
        """Calculate net profit (final amount - initial amount - gas costs)."""
        if len(self.nodes) < 2:
            return 0.0
        initial_amount = self.nodes[0].amount
        return self.final_amount - initial_amount - self.total_gas_cost