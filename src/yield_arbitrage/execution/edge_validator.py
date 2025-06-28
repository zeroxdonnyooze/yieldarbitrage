"""Edge validation module for pre-execution validation of edges."""
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from ..graph_engine.models import YieldGraphEdge, EdgeState, EdgeType
from .hybrid_simulator import SimulationResult, SimulationMode

logger = logging.getLogger(__name__)


@dataclass
class EdgeValidationResult:
    """Result of edge validation."""
    is_valid: bool
    edge_id: str
    
    # Validation details
    has_sufficient_liquidity: bool = True
    has_fresh_data: bool = True
    meets_confidence_threshold: bool = True
    conversion_rate_valid: bool = True
    gas_cost_reasonable: bool = True
    
    # Warnings and errors
    warnings: List[str] = None
    errors: List[str] = None
    
    # Metrics
    liquidity_usd: Optional[float] = None
    confidence_score: Optional[float] = None
    data_age_seconds: Optional[float] = None
    estimated_gas_cost_usd: Optional[float] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class EdgeValidator:
    """
    Validates edge viability before execution.
    
    Performs various checks including liquidity, data freshness,
    conversion rates, and optionally uses simulation for validation.
    """
    
    def __init__(
        self,
        min_liquidity_usd: float = 10_000.0,
        max_data_age_seconds: float = 300.0,  # 5 minutes
        min_confidence_score: float = 0.7,
        max_gas_cost_usd: float = 100.0,
        tenderly_simulator: Optional[Any] = None,
        local_simulator: Optional[Any] = None
    ):
        """
        Initialize edge validator.
        
        Args:
            min_liquidity_usd: Minimum required liquidity
            max_data_age_seconds: Maximum age for edge state data
            min_confidence_score: Minimum confidence threshold
            max_gas_cost_usd: Maximum acceptable gas cost
            tenderly_simulator: Optional Tenderly simulator for validation
            local_simulator: Optional local simulator as fallback
        """
        self.min_liquidity_usd = min_liquidity_usd
        self.max_data_age_seconds = max_data_age_seconds
        self.min_confidence_score = min_confidence_score
        self.max_gas_cost_usd = max_gas_cost_usd
        self.tenderly_simulator = tenderly_simulator
        self.local_simulator = local_simulator
        
        # Statistics
        self._stats = {
            "edges_validated": 0,
            "edges_valid": 0,
            "edges_invalid": 0,
            "liquidity_failures": 0,
            "stale_data_failures": 0,
            "confidence_failures": 0,
            "simulation_failures": 0,
        }
    
    async def validate_edge(
        self,
        edge: YieldGraphEdge,
        edge_state: EdgeState,
        input_amount: Optional[float] = None,
        use_simulation: bool = False
    ) -> EdgeValidationResult:
        """
        Validate an edge for execution viability.
        
        Args:
            edge: The edge to validate
            edge_state: Current state of the edge
            input_amount: Optional input amount for simulation
            use_simulation: Whether to use simulation for validation
            
        Returns:
            EdgeValidationResult with validation details
        """
        self._stats["edges_validated"] += 1
        
        result = EdgeValidationResult(
            is_valid=True,
            edge_id=edge.edge_id
        )
        
        # Check liquidity
        if edge_state.liquidity_usd is not None:
            result.liquidity_usd = edge_state.liquidity_usd
            if edge_state.liquidity_usd < self.min_liquidity_usd:
                result.has_sufficient_liquidity = False
                result.is_valid = False
                result.errors.append(
                    f"Insufficient liquidity: ${edge_state.liquidity_usd:.2f} < ${self.min_liquidity_usd:.2f}"
                )
                self._stats["liquidity_failures"] += 1
        else:
            result.warnings.append("Liquidity data unavailable")
        
        # Check data freshness
        if edge_state.last_updated_timestamp is not None:
            import time
            data_age = time.time() - edge_state.last_updated_timestamp
            result.data_age_seconds = data_age
            
            if data_age > self.max_data_age_seconds:
                result.has_fresh_data = False
                result.is_valid = False
                result.errors.append(
                    f"Stale data: {data_age:.1f}s > {self.max_data_age_seconds}s"
                )
                self._stats["stale_data_failures"] += 1
        else:
            result.warnings.append("No timestamp on edge state data")
        
        # Check confidence score
        result.confidence_score = edge_state.confidence_score
        if edge_state.confidence_score < self.min_confidence_score:
            result.meets_confidence_threshold = False
            result.is_valid = False
            result.errors.append(
                f"Low confidence: {edge_state.confidence_score:.2f} < {self.min_confidence_score:.2f}"
            )
            self._stats["confidence_failures"] += 1
        
        # Check conversion rate
        if edge_state.conversion_rate is None or edge_state.conversion_rate <= 0:
            result.conversion_rate_valid = False
            result.is_valid = False
            result.errors.append("Invalid or missing conversion rate")
        
        # Check gas cost
        if edge_state.gas_cost_usd is not None:
            result.estimated_gas_cost_usd = edge_state.gas_cost_usd
            if edge_state.gas_cost_usd > self.max_gas_cost_usd:
                result.gas_cost_reasonable = False
                result.warnings.append(
                    f"High gas cost: ${edge_state.gas_cost_usd:.2f}"
                )
        
        # Edge type specific validation
        self._validate_edge_type_specific(edge, edge_state, result)
        
        # Optional simulation validation
        if use_simulation and input_amount and result.is_valid:
            sim_result = await self._validate_with_simulation(
                edge, edge_state, input_amount
            )
            if not sim_result.success:
                result.is_valid = False
                result.errors.append(
                    f"Simulation failed: {sim_result.revert_reason}"
                )
                self._stats["simulation_failures"] += 1
        
        # Update statistics
        if result.is_valid:
            self._stats["edges_valid"] += 1
        else:
            self._stats["edges_invalid"] += 1
        
        return result
    
    async def validate_path(
        self,
        edges: List[YieldGraphEdge],
        edge_states: List[EdgeState],
        initial_amount: Optional[float] = None
    ) -> List[EdgeValidationResult]:
        """
        Validate a complete path of edges.
        
        Args:
            edges: List of edges in the path
            edge_states: Corresponding edge states
            initial_amount: Optional initial amount for the path
            
        Returns:
            List of validation results for each edge
        """
        if len(edges) != len(edge_states):
            raise ValueError("Edges and states lists must have same length")
        
        results = []
        current_amount = initial_amount
        
        for edge, state in zip(edges, edge_states):
            # Validate each edge
            result = await self.validate_edge(
                edge, state, current_amount, use_simulation=False
            )
            results.append(result)
            
            # Update amount for next edge if we have conversion rate
            if current_amount and state.conversion_rate:
                current_amount = current_amount * state.conversion_rate
        
        return results
    
    def _validate_edge_type_specific(
        self,
        edge: YieldGraphEdge,
        edge_state: EdgeState,
        result: EdgeValidationResult
    ) -> None:
        """Perform edge-type specific validation."""
        if edge.edge_type == EdgeType.FLASH_LOAN:
            # Flash loans need to be executed synchronously
            if not edge.execution_properties.supports_synchronous:
                result.is_valid = False
                result.errors.append("Flash loan edge must support synchronous execution")
        
        elif edge.edge_type == EdgeType.BRIDGE:
            # Bridges typically have time delays
            if edge.execution_properties.requires_time_delay is None:
                result.warnings.append("Bridge edge missing time delay information")
        
        elif edge.edge_type == EdgeType.LEND or edge.edge_type == EdgeType.STAKE:
            # These typically require capital holding
            if not edge.execution_properties.requires_capital_holding:
                result.warnings.append(
                    f"{edge.edge_type} edge should require capital holding"
                )
        
        elif edge.edge_type == EdgeType.BACK_RUN:
            # Back-runs are MEV sensitive but in a different way
            if edge.execution_properties.mev_sensitivity > 0.1:
                result.warnings.append(
                    "Back-run edge should have low MEV sensitivity"
                )
    
    async def _validate_with_simulation(
        self,
        edge: YieldGraphEdge,
        edge_state: EdgeState,
        input_amount: float
    ) -> SimulationResult:
        """Validate edge using simulation."""
        # Try Tenderly first if available
        if self.tenderly_simulator:
            try:
                return await self.tenderly_simulator.validate_edge(
                    edge, input_amount, SimulationMode.TENDERLY
                )
            except Exception as e:
                logger.warning(f"Tenderly simulation failed: {e}")
        
        # Fallback to local simulation
        if self.local_simulator:
            try:
                return await self.local_simulator.validate_edge(
                    edge, input_amount, SimulationMode.LOCAL
                )
            except Exception as e:
                logger.warning(f"Local simulation failed: {e}")
        
        # No simulation available
        return SimulationResult(
            success=False,
            simulation_mode="none",
            revert_reason="No simulation available for validation"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self._stats.copy()
        
        # Calculate percentages
        if stats["edges_validated"] > 0:
            stats["valid_percentage"] = (
                stats["edges_valid"] / stats["edges_validated"] * 100
            )
            stats["liquidity_failure_rate"] = (
                stats["liquidity_failures"] / stats["edges_validated"] * 100
            )
            stats["stale_data_rate"] = (
                stats["stale_data_failures"] / stats["edges_validated"] * 100
            )
        
        return stats
    
    async def validate_edge_viability(
        self,
        edge: YieldGraphEdge,
        current_state: Dict[str, Any]
    ) -> SimulationResult:
        """
        Validate edge viability (compatibility method).
        
        This method provides compatibility with the Tenderly integration
        interface from the original Task 13.
        """
        # Extract edge state from current state
        edge_state = current_state.get("edge_state")
        if not edge_state:
            return SimulationResult(
                success=False,
                simulation_mode="validation",
                revert_reason="No edge state provided"
            )
        
        # Validate the edge
        input_amount = current_state.get("input_amount", 1.0)
        validation_result = await self.validate_edge(
            edge, edge_state, input_amount, use_simulation=True
        )
        
        # Convert to SimulationResult
        return SimulationResult(
            success=validation_result.is_valid,
            simulation_mode="validation",
            warnings=validation_result.warnings,
            revert_reason=(
                validation_result.errors[0] if validation_result.errors else None
            ),
            gas_cost_usd=validation_result.estimated_gas_cost_usd
        )