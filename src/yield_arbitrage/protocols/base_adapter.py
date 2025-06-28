"""Base adapter class for protocol integrations."""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from ..graph_engine.models import YieldGraphEdge, EdgeState

logger = logging.getLogger(__name__)


class ProtocolAdapter(ABC):
    """
    Abstract base class for protocol adapters.
    
    This class defines the common interface that all protocol integrations
    must implement to work with the yield arbitrage system.
    """
    
    def __init__(self, chain_name: str, provider: Any):
        """
        Initialize the protocol adapter.
        
        Args:
            chain_name: Name of the blockchain (e.g., 'ethereum', 'arbitrum')
            provider: Blockchain provider instance for making RPC calls
        """
        self.chain_name = chain_name.lower()
        self.provider = provider
        self.logger = logging.getLogger(f"{self.__class__.__name__}({chain_name})")
        
        # Protocol-specific attributes to be set by subclasses
        self.protocol_name: str = "Unknown"
        self.supported_edge_types: List[str] = []
        self.is_initialized: bool = False
        
        # Performance tracking
        self._discovery_stats = {
            "last_discovery": None,
            "edges_discovered": 0,
            "discovery_errors": 0
        }
        
        self._update_stats = {
            "last_update": None,
            "updates_performed": 0,
            "update_errors": 0,
            "avg_update_time": 0.0
        }
    
    @abstractmethod
    async def discover_edges(self) -> List[YieldGraphEdge]:
        """
        Discover all available edges for this protocol.
        
        This method should scan the protocol's contracts/pools and return
        a list of YieldGraphEdge objects representing available opportunities.
        
        Returns:
            List of YieldGraphEdge objects discovered from the protocol
            
        Raises:
            ProtocolError: If discovery fails due to protocol-specific issues
            NetworkError: If network/RPC issues prevent discovery
        """
        pass
    
    @abstractmethod
    async def update_edge_state(self, edge: YieldGraphEdge) -> EdgeState:
        """
        Update the state of a specific edge.
        
        This method should fetch current market data for the edge and return
        an updated EdgeState with current conversion rates, liquidity, etc.
        
        Args:
            edge: The edge to update
            
        Returns:
            Updated EdgeState with current market data
            
        Raises:
            ProtocolError: If state update fails due to protocol-specific issues
            NetworkError: If network/RPC issues prevent state update
        """
        pass
    
    async def initialize(self) -> bool:
        """
        Initialize the protocol adapter.
        
        This method should perform any setup required for the adapter to function,
        such as validating contract addresses, checking chain compatibility, etc.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing {self.protocol_name} adapter for {self.chain_name}")
            
            # Validate chain support
            if not self.is_chain_supported():
                self.logger.error(f"Chain {self.chain_name} not supported by {self.protocol_name}")
                return False
            
            # Perform protocol-specific initialization
            success = await self._protocol_specific_init()
            
            if success:
                self.is_initialized = True
                self.logger.info(f"{self.protocol_name} adapter initialized successfully")
            else:
                self.logger.error(f"Failed to initialize {self.protocol_name} adapter")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error initializing {self.protocol_name} adapter: {e}")
            return False
    
    async def _protocol_specific_init(self) -> bool:
        """
        Protocol-specific initialization logic.
        
        Subclasses can override this method to perform additional initialization
        such as contract validation, configuration loading, etc.
        
        Returns:
            True if initialization successful, False otherwise
        """
        return True
    
    def is_chain_supported(self) -> bool:
        """
        Check if the current chain is supported by this protocol.
        
        Subclasses should override this method to implement chain-specific logic.
        
        Returns:
            True if chain is supported, False otherwise
        """
        return True
    
    async def batch_update_edges(self, edges: List[YieldGraphEdge]) -> Dict[str, EdgeState]:
        """
        Update multiple edges in a batch operation.
        
        This method provides a default implementation that updates edges sequentially.
        Subclasses can override this to implement more efficient batch operations.
        
        Args:
            edges: List of edges to update
            
        Returns:
            Dictionary mapping edge IDs to updated EdgeState objects
        """
        results = {}
        errors = 0
        
        for edge in edges:
            try:
                updated_state = await self.update_edge_state(edge)
                results[edge.edge_id] = updated_state
            except Exception as e:
                self.logger.warning(f"Failed to update edge {edge.edge_id}: {e}")
                errors += 1
                # Keep the existing state if update fails
                results[edge.edge_id] = edge.state
        
        self.logger.info(f"Batch update completed: {len(results)} edges, {errors} errors")
        return results
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics for monitoring."""
        return {
            "protocol": self.protocol_name,
            "chain": self.chain_name,
            "discovery_stats": self._discovery_stats.copy(),
            "update_stats": self._update_stats.copy(),
            "is_initialized": self.is_initialized
        }
    
    def _record_discovery_success(self, edges_count: int):
        """Record successful discovery operation."""
        self._discovery_stats.update({
            "last_discovery": datetime.now(timezone.utc),
            "edges_discovered": edges_count
        })
    
    def _record_discovery_error(self):
        """Record failed discovery operation."""
        self._discovery_stats["discovery_errors"] += 1
    
    def _record_update_success(self, duration: float):
        """Record successful update operation."""
        self._update_stats["updates_performed"] += 1
        self._update_stats["last_update"] = datetime.now(timezone.utc)
        
        # Update rolling average
        current_avg = self._update_stats["avg_update_time"]
        count = self._update_stats["updates_performed"]
        self._update_stats["avg_update_time"] = ((current_avg * (count - 1)) + duration) / count
    
    def _record_update_error(self):
        """Record failed update operation."""
        self._update_stats["update_errors"] += 1
    
    def __str__(self) -> str:
        """String representation of the adapter."""
        return f"{self.protocol_name}Adapter({self.chain_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the adapter."""
        return (f"{self.__class__.__name__}("
                f"protocol={self.protocol_name}, "
                f"chain={self.chain_name}, "
                f"initialized={self.is_initialized})")


class ProtocolError(Exception):
    """Exception raised for protocol-specific errors."""
    
    def __init__(self, message: str, protocol: str = None, chain: str = None):
        """
        Initialize protocol error.
        
        Args:
            message: Error message
            protocol: Protocol name where error occurred
            chain: Chain name where error occurred
        """
        self.protocol = protocol
        self.chain = chain
        super().__init__(message)
    
    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = super().__str__()
        if self.protocol and self.chain:
            return f"[{self.protocol}@{self.chain}] {base_msg}"
        elif self.protocol:
            return f"[{self.protocol}] {base_msg}"
        return base_msg


class NetworkError(Exception):
    """Exception raised for network-related errors."""
    
    def __init__(self, message: str, chain: str = None, retry_count: int = 0):
        """
        Initialize network error.
        
        Args:
            message: Error message
            chain: Chain name where error occurred
            retry_count: Number of retries attempted
        """
        self.chain = chain
        self.retry_count = retry_count
        super().__init__(message)
    
    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = super().__str__()
        if self.chain:
            return f"[{self.chain}] {base_msg} (retries: {self.retry_count})"
        return f"{base_msg} (retries: {self.retry_count})"