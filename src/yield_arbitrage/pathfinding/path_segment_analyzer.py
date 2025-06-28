"""
Path Segment Analyzer for identifying batchable segments in arbitrage paths.

This module analyzes complete paths and identifies segments that can be executed
atomically in a single transaction, breaking paths at boundaries where atomic
execution is not possible (bridges, time delays, non-synchronous operations).
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeExecutionProperties

logger = logging.getLogger(__name__)


class SegmentType(str, Enum):
    """Types of path segments based on execution requirements."""
    ATOMIC = "atomic"                    # Can be executed in single transaction
    FLASH_LOAN_ATOMIC = "flash_loan"    # Atomic with flash loan
    TIME_DELAYED = "time_delayed"        # Requires waiting period
    BRIDGED = "bridged"                  # Requires cross-chain bridge
    CAPITAL_HOLDING = "capital_holding"  # Requires holding capital over time


@dataclass
class SegmentBoundary:
    """Represents a boundary between two segments."""
    boundary_type: str
    reason: str
    edge_index: int
    edge_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathSegment:
    """Represents a segment of edges that can be executed together."""
    segment_id: str
    segment_type: SegmentType
    edges: List[YieldGraphEdge]
    start_index: int
    end_index: int
    
    # Execution requirements
    requires_flash_loan: bool = False
    flash_loan_amount: Optional[float] = None
    flash_loan_asset: Optional[str] = None
    
    # Constraints
    max_gas_estimate: int = 0
    total_mev_sensitivity: float = 0.0
    min_liquidity_required: float = 0.0
    
    # Timing
    requires_delay_seconds: Optional[int] = None
    
    # Cross-chain
    source_chain: Optional[str] = None
    target_chain: Optional[str] = None
    
    @property
    def is_atomic(self) -> bool:
        """Check if segment can be executed atomically."""
        return self.segment_type in [SegmentType.ATOMIC, SegmentType.FLASH_LOAN_ATOMIC]
    
    @property
    def edge_count(self) -> int:
        """Number of edges in this segment."""
        return len(self.edges)
    
    def get_input_asset(self) -> Optional[str]:
        """Get the input asset for this segment."""
        return self.edges[0].source_asset_id if self.edges else None
    
    def get_output_asset(self) -> Optional[str]:
        """Get the output asset for this segment."""
        return self.edges[-1].target_asset_id if self.edges else None


class PathSegmentAnalyzer:
    """
    Analyzes arbitrage paths to identify executable segments.
    
    This analyzer examines edge execution properties to determine which
    consecutive edges can be batched together for atomic execution.
    """
    
    def __init__(self, 
                 max_segment_gas: int = 8_000_000,
                 max_mev_sensitivity: float = 0.8):
        """
        Initialize the path segment analyzer.
        
        Args:
            max_segment_gas: Maximum gas allowed per segment
            max_mev_sensitivity: Maximum cumulative MEV sensitivity per segment
        """
        self.max_segment_gas = max_segment_gas
        self.max_mev_sensitivity = max_mev_sensitivity
        
        # Statistics
        self.stats = {
            "paths_analyzed": 0,
            "segments_created": 0,
            "atomic_segments": 0,
            "delayed_segments": 0,
            "bridged_segments": 0,
            "flash_loan_segments": 0
        }
    
    def analyze_path(self, edges: List[YieldGraphEdge]) -> List[PathSegment]:
        """
        Analyze a complete path and identify executable segments.
        
        Args:
            edges: List of edges forming the complete path
            
        Returns:
            List of path segments that can be executed independently
        """
        if not edges:
            return []
        
        self.stats["paths_analyzed"] += 1
        
        # Identify segment boundaries
        boundaries = self._identify_boundaries(edges)
        
        # Create segments from boundaries
        segments = self._create_segments(edges, boundaries)
        
        # Analyze each segment for execution requirements
        for segment in segments:
            self._analyze_segment_requirements(segment)
        
        self.stats["segments_created"] += len(segments)
        
        logger.info(f"Analyzed path with {len(edges)} edges into {len(segments)} segments")
        
        return segments
    
    def _identify_boundaries(self, edges: List[YieldGraphEdge]) -> List[SegmentBoundary]:
        """Identify boundaries where segments must be split."""
        boundaries = []
        cumulative_gas = 0
        cumulative_mev = 0.0
        current_chain = edges[0].chain_name if edges else None
        
        for i, edge in enumerate(edges):
            props = edge.execution_properties
            
            # Check for hard boundaries (must split)
            
            # 1. Time delay required
            if props.requires_time_delay and props.requires_time_delay > 0:
                boundaries.append(SegmentBoundary(
                    boundary_type="time_delay",
                    reason=f"Edge requires {props.requires_time_delay}s delay",
                    edge_index=i,
                    edge_id=edge.edge_id,
                    metadata={"delay_seconds": props.requires_time_delay}
                ))
            
            # 2. Bridge required
            elif props.requires_bridge:
                boundaries.append(SegmentBoundary(
                    boundary_type="bridge",
                    reason="Edge requires cross-chain bridge",
                    edge_index=i,
                    edge_id=edge.edge_id,
                    metadata={"from_chain": current_chain, "to_chain": edge.chain_name}
                ))
                current_chain = edge.chain_name
            
            # 3. Not synchronous
            elif not props.supports_synchronous:
                boundaries.append(SegmentBoundary(
                    boundary_type="async_required",
                    reason="Edge does not support synchronous execution",
                    edge_index=i,
                    edge_id=edge.edge_id
                ))
            
            # 4. Capital holding required (e.g., lending positions)
            elif props.requires_capital_holding:
                boundaries.append(SegmentBoundary(
                    boundary_type="capital_holding",
                    reason="Edge requires holding capital over time",
                    edge_index=i,
                    edge_id=edge.edge_id
                ))
            
            # Check for soft boundaries (optimization splits)
            
            # 5. Gas limit exceeded
            cumulative_gas += props.gas_estimate
            if cumulative_gas > self.max_segment_gas:
                boundaries.append(SegmentBoundary(
                    boundary_type="gas_limit",
                    reason=f"Cumulative gas {cumulative_gas} exceeds limit",
                    edge_index=i,
                    edge_id=edge.edge_id,
                    metadata={"cumulative_gas": cumulative_gas}
                ))
                cumulative_gas = props.gas_estimate
            
            # 6. MEV sensitivity too high
            cumulative_mev = max(cumulative_mev, props.mev_sensitivity)
            if cumulative_mev > self.max_mev_sensitivity:
                boundaries.append(SegmentBoundary(
                    boundary_type="mev_sensitivity",
                    reason=f"Cumulative MEV sensitivity {cumulative_mev:.2f} too high",
                    edge_index=i,
                    edge_id=edge.edge_id,
                    metadata={"cumulative_mev": cumulative_mev}
                ))
                cumulative_mev = props.mev_sensitivity
            
            # 7. Chain change (even without bridge)
            if edge.chain_name != current_chain:
                boundaries.append(SegmentBoundary(
                    boundary_type="chain_change",
                    reason=f"Chain change from {current_chain} to {edge.chain_name}",
                    edge_index=i,
                    edge_id=edge.edge_id,
                    metadata={"from_chain": current_chain, "to_chain": edge.chain_name}
                ))
                current_chain = edge.chain_name
        
        return boundaries
    
    def _create_segments(self, edges: List[YieldGraphEdge], 
                        boundaries: List[SegmentBoundary]) -> List[PathSegment]:
        """Create segments from edges and boundaries."""
        segments = []
        start_idx = 0
        
        # Sort boundaries by edge index
        sorted_boundaries = sorted(boundaries, key=lambda b: b.edge_index)
        
        for i, boundary in enumerate(sorted_boundaries):
            # Create segment up to this boundary
            end_idx = boundary.edge_index
            
            if start_idx < end_idx:
                segment = self._create_segment(
                    edges[start_idx:end_idx],
                    start_idx,
                    end_idx - 1,
                    len(segments)
                )
                segments.append(segment)
            
            # Create segment for the boundary edge itself if needed
            if boundary.boundary_type in ["time_delay", "bridge", "capital_holding"]:
                segment = self._create_segment(
                    [edges[boundary.edge_index]],
                    boundary.edge_index,
                    boundary.edge_index,
                    len(segments),
                    boundary_type=boundary.boundary_type
                )
                segments.append(segment)
                start_idx = boundary.edge_index + 1
            else:
                # For other boundaries, include the edge in the next segment
                start_idx = boundary.edge_index
        
        # Create final segment if there are remaining edges
        if start_idx < len(edges):
            segment = self._create_segment(
                edges[start_idx:],
                start_idx,
                len(edges) - 1,
                len(segments)
            )
            segments.append(segment)
        
        return segments
    
    def _create_segment(self, edges: List[YieldGraphEdge], 
                       start_idx: int, end_idx: int, 
                       segment_number: int,
                       boundary_type: Optional[str] = None) -> PathSegment:
        """Create a single segment from edges."""
        # Determine segment type
        if boundary_type == "time_delay":
            segment_type = SegmentType.TIME_DELAYED
        elif boundary_type == "bridge":
            segment_type = SegmentType.BRIDGED
        elif boundary_type == "capital_holding":
            segment_type = SegmentType.CAPITAL_HOLDING
        else:
            segment_type = SegmentType.ATOMIC
        
        segment = PathSegment(
            segment_id=f"seg_{segment_number}",
            segment_type=segment_type,
            edges=edges,
            start_index=start_idx,
            end_index=end_idx,
            source_chain=edges[0].chain_name if edges else None,
            target_chain=edges[-1].chain_name if edges else None
        )
        
        # Update statistics
        if segment_type == SegmentType.ATOMIC:
            self.stats["atomic_segments"] += 1
        elif segment_type == SegmentType.TIME_DELAYED:
            self.stats["delayed_segments"] += 1
        elif segment_type == SegmentType.BRIDGED:
            self.stats["bridged_segments"] += 1
        
        return segment
    
    def _analyze_segment_requirements(self, segment: PathSegment) -> None:
        """Analyze execution requirements for a segment."""
        # Calculate aggregate properties
        total_gas = 0
        max_mev = 0.0
        min_liquidity = 0.0
        
        for edge in segment.edges:
            props = edge.execution_properties
            total_gas += props.gas_estimate
            max_mev = max(max_mev, props.mev_sensitivity)
            min_liquidity = max(min_liquidity, props.min_liquidity_required)
            
            # Check for time delays
            if props.requires_time_delay:
                segment.requires_delay_seconds = props.requires_time_delay
        
        segment.max_gas_estimate = total_gas
        segment.total_mev_sensitivity = max_mev
        segment.min_liquidity_required = min_liquidity
        
        # Check if segment needs flash loan
        if segment.is_atomic and self._requires_flash_loan(segment):
            segment.requires_flash_loan = True
            segment.segment_type = SegmentType.FLASH_LOAN_ATOMIC
            self.stats["flash_loan_segments"] += 1
            
            # Determine flash loan parameters
            self._determine_flash_loan_params(segment)
    
    def _requires_flash_loan(self, segment: PathSegment) -> bool:
        """Determine if a segment requires a flash loan."""
        # Check if any edge in the segment is a flash loan edge
        for edge in segment.edges:
            if edge.edge_type == EdgeType.FLASH_LOAN:
                return True
        
        # Check if segment starts with high capital requirement
        # This is a simplified check - real implementation would be more sophisticated
        if segment.edges and segment.min_liquidity_required > 100_000:  # $100k threshold
            return True
        
        return False
    
    def _determine_flash_loan_params(self, segment: PathSegment) -> None:
        """Determine flash loan parameters for a segment."""
        # Find flash loan edge if present
        for edge in segment.edges:
            if edge.edge_type == EdgeType.FLASH_LOAN:
                # Extract loan parameters from edge metadata
                segment.flash_loan_asset = edge.source_asset_id
                # In real implementation, amount would come from path analysis
                segment.flash_loan_amount = 100_000  # Placeholder
                return
        
        # If no explicit flash loan edge, determine from capital requirements
        if segment.edges:
            segment.flash_loan_asset = segment.edges[0].source_asset_id
            segment.flash_loan_amount = segment.min_liquidity_required
    
    def get_segment_summary(self, segment: PathSegment) -> Dict[str, Any]:
        """Get a summary of segment properties."""
        return {
            "segment_id": segment.segment_id,
            "type": segment.segment_type,
            "edge_count": segment.edge_count,
            "is_atomic": segment.is_atomic,
            "edges": [e.edge_id for e in segment.edges],
            "input_asset": segment.get_input_asset(),
            "output_asset": segment.get_output_asset(),
            "gas_estimate": segment.max_gas_estimate,
            "mev_sensitivity": segment.total_mev_sensitivity,
            "requires_flash_loan": segment.requires_flash_loan,
            "flash_loan_amount": segment.flash_loan_amount,
            "source_chain": segment.source_chain,
            "target_chain": segment.target_chain,
            "delay_seconds": segment.requires_delay_seconds
        }
    
    def validate_segment_connectivity(self, segments: List[PathSegment]) -> bool:
        """Validate that segments connect properly."""
        if not segments:
            return True
        
        for i in range(len(segments) - 1):
            current_output = segments[i].get_output_asset()
            next_input = segments[i + 1].get_input_asset()
            
            if current_output != next_input:
                logger.error(f"Segment connectivity broken: {current_output} != {next_input}")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return self.stats.copy()