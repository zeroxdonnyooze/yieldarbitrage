"""
MEV Risk Assessment Module.

This module provides comprehensive MEV (Maximum Extractable Value) risk assessment
for individual edges and complete arbitrage paths. It calculates risk scores based
on various factors including edge types, protocols, liquidity, and market conditions.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from enum import Enum

from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType
from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegment

logger = logging.getLogger(__name__)


class MEVRiskLevel(str, Enum):
    """MEV risk levels for categorization."""
    MINIMAL = "minimal"      # < 0.2 - Safe for public mempool
    LOW = "low"             # 0.2-0.4 - Consider private relay
    MEDIUM = "medium"       # 0.4-0.6 - Recommend private relay
    HIGH = "high"           # 0.6-0.8 - Require private relay
    CRITICAL = "critical"   # > 0.8 - Flashbots/private only


@dataclass
class EdgeMEVAnalysis:
    """Detailed MEV analysis for a single edge."""
    edge_id: str
    base_risk_score: float
    
    # Risk factors
    sandwich_risk: float = 0.0
    frontrun_risk: float = 0.0
    backrun_opportunity: float = 0.0
    liquidity_risk: float = 0.0
    
    # Modifiers
    protocol_modifier: float = 1.0
    chain_modifier: float = 1.0
    size_modifier: float = 1.0
    
    # Final scores
    final_risk_score: float = 0.0
    risk_level: MEVRiskLevel = MEVRiskLevel.MINIMAL
    
    # Recommendations
    recommended_execution: str = "public"
    estimated_mev_loss_bps: float = 0.0  # Basis points
    
    # Details
    risk_factors: Dict[str, float] = field(default_factory=dict)
    mitigation_suggestions: List[str] = field(default_factory=list)


@dataclass
class PathMEVAnalysis:
    """Complete MEV analysis for an arbitrage path."""
    path_id: str
    total_edges: int
    
    # Aggregate scores
    max_edge_risk: float = 0.0
    average_edge_risk: float = 0.0
    compounded_risk: float = 0.0
    
    # Risk categorization
    overall_risk_level: MEVRiskLevel = MEVRiskLevel.MINIMAL
    critical_edges: List[str] = field(default_factory=list)
    
    # Execution recommendations
    recommended_execution_method: str = "public"
    requires_atomic_execution: bool = False
    estimated_total_mev_loss_bps: float = 0.0
    
    # Segment analysis
    segment_risks: Dict[str, float] = field(default_factory=dict)
    highest_risk_segment: Optional[str] = None
    
    # Detailed analysis
    edge_analyses: List[EdgeMEVAnalysis] = field(default_factory=list)
    execution_strategy: Dict[str, Any] = field(default_factory=dict)


class MEVRiskAssessor:
    """
    Comprehensive MEV risk assessment for arbitrage paths.
    
    This class analyzes individual edges and complete paths to determine
    MEV exposure and recommend appropriate execution strategies.
    """
    
    def __init__(self):
        """Initialize MEV risk assessor with risk parameters."""
        
        # Base risk scores by edge type
        self.edge_type_risks = {
            EdgeType.TRADE: 0.6,      # High risk - DEX trades are primary MEV targets
            EdgeType.SPLIT: 0.2,      # Low risk - Internal operation
            EdgeType.COMBINE: 0.2,    # Low risk - Internal operation
            EdgeType.BRIDGE: 0.4,     # Medium risk - Cross-chain has delays
            EdgeType.LEND: 0.3,       # Low-medium risk - Less MEV opportunity
            EdgeType.BORROW: 0.3,     # Low-medium risk - Less MEV opportunity
            EdgeType.STAKE: 0.2,      # Low risk - Usually not MEV target
            EdgeType.WAIT: 0.1,       # Minimal risk - Time delay
            EdgeType.SHORT: 0.5,      # Medium-high risk - Complex operation
            EdgeType.FLASH_LOAN: 0.7, # High risk - Capital intensive
            EdgeType.BACK_RUN: 0.8    # Very high risk - MEV operation itself
        }
        
        # Protocol risk modifiers
        self.protocol_risks = {
            # AMMs - High MEV risk
            "uniswap_v2": 1.2,
            "uniswap_v3": 1.1,  # Slightly lower due to concentrated liquidity
            "sushiswap": 1.2,
            "pancakeswap": 1.2,
            "balancer": 1.0,    # Different mechanism
            "curve": 0.9,       # Stable swaps less MEV
            
            # Lending - Lower MEV risk
            "aave": 0.8,
            "compound": 0.8,
            "maker": 0.7,
            
            # Aggregators - Variable risk
            "1inch": 1.0,
            "paraswap": 1.0,
            "0x": 0.9,
            
            # Unknown
            "unknown": 1.0
        }
        
        # Chain-specific modifiers
        self.chain_modifiers = {
            "ethereum": 1.2,    # Highest MEV activity
            "bsc": 1.1,        # High MEV activity
            "polygon": 0.9,    # Moderate MEV
            "arbitrum": 0.8,   # Lower MEV due to sequencer
            "optimism": 0.8,   # Lower MEV due to sequencer
            "avalanche": 0.9,  # Moderate MEV
            "fantom": 0.8,     # Lower MEV
            "base": 0.7,       # Newer chain, less MEV
            "zksync": 0.7,     # Different architecture
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            MEVRiskLevel.MINIMAL: 0.2,
            MEVRiskLevel.LOW: 0.4,
            MEVRiskLevel.MEDIUM: 0.6,
            MEVRiskLevel.HIGH: 0.8,
            MEVRiskLevel.CRITICAL: 1.0
        }
    
    def assess_edge_risk(
        self,
        edge: YieldGraphEdge,
        input_amount_usd: float,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> EdgeMEVAnalysis:
        """
        Assess MEV risk for a single edge.
        
        Args:
            edge: The edge to analyze
            input_amount_usd: Trade size in USD
            market_conditions: Optional market data (volatility, gas prices, etc.)
            
        Returns:
            Detailed MEV analysis for the edge
        """
        logger.debug(f"Assessing MEV risk for edge {edge.edge_id}")
        
        # Start with base risk from edge properties and type
        base_risk = edge.execution_properties.mev_sensitivity
        edge_type_risk = self.edge_type_risks.get(edge.edge_type, 0.5)
        
        # Calculate specific risk factors
        sandwich_risk = self._calculate_sandwich_risk(edge, input_amount_usd)
        frontrun_risk = self._calculate_frontrun_risk(edge, input_amount_usd)
        backrun_opportunity = self._calculate_backrun_opportunity(edge)
        liquidity_risk = self._calculate_liquidity_risk(edge, input_amount_usd)
        
        # Get modifiers
        protocol_modifier = self._get_protocol_modifier(edge.protocol_name)
        chain_modifier = self._get_chain_modifier(edge.chain_name)
        size_modifier = self._calculate_size_modifier(input_amount_usd)
        
        # Calculate final risk score
        risk_components = {
            "base_risk": base_risk * 0.2,
            "edge_type_risk": edge_type_risk * 0.2,
            "sandwich_risk": sandwich_risk * 0.2,
            "frontrun_risk": frontrun_risk * 0.2,
            "liquidity_risk": liquidity_risk * 0.1,
            "backrun_opportunity": backrun_opportunity * 0.1
        }
        
        weighted_risk = sum(risk_components.values())
        final_risk = min(weighted_risk * protocol_modifier * chain_modifier * size_modifier, 1.0)
        
        # Determine risk level
        risk_level = self._categorize_risk_level(final_risk)
        
        # Generate recommendations
        recommended_execution = self._recommend_execution_method(risk_level, edge)
        estimated_mev_loss = self._estimate_mev_loss(final_risk, input_amount_usd)
        
        # Create analysis
        analysis = EdgeMEVAnalysis(
            edge_id=edge.edge_id,
            base_risk_score=base_risk,
            sandwich_risk=sandwich_risk,
            frontrun_risk=frontrun_risk,
            backrun_opportunity=backrun_opportunity,
            liquidity_risk=liquidity_risk,
            protocol_modifier=protocol_modifier,
            chain_modifier=chain_modifier,
            size_modifier=size_modifier,
            final_risk_score=final_risk,
            risk_level=risk_level,
            recommended_execution=recommended_execution,
            estimated_mev_loss_bps=estimated_mev_loss,
            risk_factors=risk_components,
            mitigation_suggestions=self._generate_mitigation_suggestions(
                edge, risk_level, risk_components
            )
        )
        
        return analysis
    
    def assess_path_risk(
        self,
        path: List[YieldGraphEdge],
        input_amount_usd: float,
        segments: Optional[List[PathSegment]] = None
    ) -> PathMEVAnalysis:
        """
        Assess MEV risk for a complete arbitrage path.
        
        Args:
            path: List of edges in the arbitrage path
            input_amount_usd: Initial trade size in USD
            segments: Optional path segments for atomic analysis
            
        Returns:
            Complete MEV analysis for the path
        """
        logger.info(f"Assessing MEV risk for path with {len(path)} edges")
        
        # Analyze each edge
        edge_analyses = []
        current_amount = input_amount_usd
        
        for edge in path:
            edge_analysis = self.assess_edge_risk(edge, current_amount)
            edge_analyses.append(edge_analysis)
            
            # Estimate output amount for next edge (simplified)
            if edge.state.conversion_rate:
                current_amount *= edge.state.conversion_rate
        
        # Calculate aggregate metrics
        risk_scores = [analysis.final_risk_score for analysis in edge_analyses]
        max_risk = max(risk_scores) if risk_scores else 0.0
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        # Compound risk calculation (edges can amplify each other)
        compounded_risk = self._calculate_compounded_risk(edge_analyses)
        
        # Find critical edges
        critical_edges = [
            analysis.edge_id for analysis in edge_analyses
            if analysis.risk_level in [MEVRiskLevel.HIGH, MEVRiskLevel.CRITICAL]
        ]
        
        # Analyze segments if provided
        segment_risks = {}
        highest_risk_segment = None
        if segments:
            segment_risks, highest_risk_segment = self._analyze_segment_risks(
                segments, edge_analyses
            )
        
        # Determine overall risk level
        overall_risk_level = self._categorize_risk_level(compounded_risk)
        
        # Generate execution strategy
        execution_strategy = self._generate_execution_strategy(
            overall_risk_level,
            edge_analyses,
            segments
        )
        
        # Create path analysis
        path_analysis = PathMEVAnalysis(
            path_id=f"path_{len(path)}_{int(compounded_risk * 1000)}",
            total_edges=len(path),
            max_edge_risk=max_risk,
            average_edge_risk=avg_risk,
            compounded_risk=compounded_risk,
            overall_risk_level=overall_risk_level,
            critical_edges=critical_edges,
            recommended_execution_method=execution_strategy["method"],
            requires_atomic_execution=execution_strategy["atomic_required"],
            estimated_total_mev_loss_bps=sum(
                analysis.estimated_mev_loss_bps for analysis in edge_analyses
            ),
            segment_risks=segment_risks,
            highest_risk_segment=highest_risk_segment,
            edge_analyses=edge_analyses,
            execution_strategy=execution_strategy
        )
        
        logger.info(
            f"Path MEV assessment complete: {overall_risk_level.value} risk, "
            f"recommend {execution_strategy['method']} execution"
        )
        
        return path_analysis
    
    def _calculate_sandwich_risk(self, edge: YieldGraphEdge, amount_usd: float) -> float:
        """Calculate sandwich attack risk for an edge."""
        if edge.edge_type != EdgeType.TRADE:
            return 0.0
        
        # Factors that increase sandwich risk
        risk = 0.0
        
        # Large trades are more attractive
        if amount_usd > 100_000:
            risk += 0.3
        elif amount_usd > 50_000:
            risk += 0.2
        elif amount_usd > 10_000:
            risk += 0.1
        
        # Low liquidity increases risk
        if edge.state.liquidity_usd and edge.state.liquidity_usd < amount_usd * 10:
            risk += 0.3
        
        # High slippage tolerance increases risk
        if edge.execution_properties.max_impact_allowed > 0.02:
            risk += 0.2
        
        return min(risk, 1.0)
    
    def _calculate_frontrun_risk(self, edge: YieldGraphEdge, amount_usd: float) -> float:
        """Calculate frontrunning risk for an edge."""
        risk = 0.0
        
        # Trade edges are most susceptible
        if edge.edge_type == EdgeType.TRADE:
            risk += 0.3
        elif edge.edge_type in [EdgeType.FLASH_LOAN, EdgeType.SHORT]:
            risk += 0.2
        
        # Large amounts increase frontrun risk
        if amount_usd > 100_000:
            risk += 0.2
        
        # Protocol-specific risks
        if "uniswap" in edge.protocol_name.lower():
            risk += 0.1
        
        return min(risk, 1.0)
    
    def _calculate_backrun_opportunity(self, edge: YieldGraphEdge) -> float:
        """Calculate if this edge creates backrun opportunities."""
        opportunity = 0.0
        
        # Large trades create imbalances
        if edge.edge_type == EdgeType.TRADE:
            opportunity += 0.3
        
        # Liquidations and large borrows
        if edge.edge_type in [EdgeType.BORROW, EdgeType.SHORT]:
            opportunity += 0.2
        
        # Flash loans often create opportunities
        if edge.edge_type == EdgeType.FLASH_LOAN:
            opportunity += 0.4
        
        return min(opportunity, 1.0)
    
    def _calculate_liquidity_risk(self, edge: YieldGraphEdge, amount_usd: float) -> float:
        """Calculate MEV risk from liquidity constraints."""
        if not edge.state.liquidity_usd:
            return 0.5  # Unknown liquidity is risky
        
        # Calculate size relative to liquidity
        size_ratio = amount_usd / edge.state.liquidity_usd
        
        if size_ratio > 0.1:  # > 10% of liquidity
            return 0.8
        elif size_ratio > 0.05:  # > 5% of liquidity
            return 0.5
        elif size_ratio > 0.01:  # > 1% of liquidity
            return 0.2
        else:
            return 0.0
    
    def _get_protocol_modifier(self, protocol_name: str) -> float:
        """Get risk modifier for specific protocol."""
        return self.protocol_risks.get(protocol_name.lower(), 1.0)
    
    def _get_chain_modifier(self, chain_name: str) -> float:
        """Get risk modifier for specific chain."""
        return self.chain_modifiers.get(chain_name.lower(), 1.0)
    
    def _calculate_size_modifier(self, amount_usd: float) -> float:
        """Calculate risk modifier based on trade size."""
        if amount_usd > 1_000_000:
            return 1.5  # Very large trades attract more MEV
        elif amount_usd > 500_000:
            return 1.3
        elif amount_usd > 100_000:
            return 1.1
        elif amount_usd < 10_000:
            return 0.8  # Small trades less attractive
        else:
            return 1.0
    
    def _categorize_risk_level(self, risk_score: float) -> MEVRiskLevel:
        """Categorize risk score into risk level."""
        if risk_score < self.risk_thresholds[MEVRiskLevel.MINIMAL]:
            return MEVRiskLevel.MINIMAL
        elif risk_score < self.risk_thresholds[MEVRiskLevel.LOW]:
            return MEVRiskLevel.LOW
        elif risk_score < self.risk_thresholds[MEVRiskLevel.MEDIUM]:
            return MEVRiskLevel.MEDIUM
        elif risk_score < self.risk_thresholds[MEVRiskLevel.HIGH]:
            return MEVRiskLevel.HIGH
        else:
            return MEVRiskLevel.CRITICAL
    
    def _recommend_execution_method(self, risk_level: MEVRiskLevel, edge: YieldGraphEdge) -> str:
        """Recommend execution method based on risk level."""
        if not edge.execution_properties.supports_private_mempool:
            return "public"  # No choice
        
        if risk_level == MEVRiskLevel.MINIMAL:
            return "public"
        elif risk_level == MEVRiskLevel.LOW:
            return "public_with_protection"  # Use MEV protection but public
        elif risk_level == MEVRiskLevel.MEDIUM:
            return "private_relay"
        elif risk_level == MEVRiskLevel.HIGH:
            return "flashbots"  # Ethereum specific
        else:  # CRITICAL
            return "flashbots_bundle"  # Must use bundle
    
    def _estimate_mev_loss(self, risk_score: float, amount_usd: float) -> float:
        """Estimate potential MEV loss in basis points."""
        # Base loss estimation
        base_loss_bps = risk_score * 50  # Up to 50 bps for max risk
        
        # Size adjustment - larger trades have lower relative loss
        if amount_usd > 1_000_000:
            size_multiplier = 0.7
        elif amount_usd > 100_000:
            size_multiplier = 0.85
        else:
            size_multiplier = 1.0
        
        return base_loss_bps * size_multiplier
    
    def _generate_mitigation_suggestions(
        self,
        edge: YieldGraphEdge,
        risk_level: MEVRiskLevel,
        risk_factors: Dict[str, float]
    ) -> List[str]:
        """Generate specific mitigation suggestions."""
        suggestions = []
        
        if risk_level >= MEVRiskLevel.MEDIUM:
            suggestions.append("Use private mempool or Flashbots for execution")
        
        if risk_factors.get("sandwich_risk", 0) > 0.3:
            suggestions.append("Split trade into smaller chunks")
            suggestions.append("Use limit orders instead of market orders")
        
        if risk_factors.get("liquidity_risk", 0) > 0.3:
            suggestions.append("Wait for better liquidity conditions")
            suggestions.append("Use multiple liquidity sources")
        
        if edge.edge_type == EdgeType.TRADE and risk_level >= MEVRiskLevel.HIGH:
            suggestions.append("Consider using a DEX aggregator with MEV protection")
        
        return suggestions
    
    def _calculate_compounded_risk(self, edge_analyses: List[EdgeMEVAnalysis]) -> float:
        """Calculate compounded risk for multiple edges."""
        if not edge_analyses:
            return 0.0
        
        # Start with the highest single edge risk
        max_risk = max(analysis.final_risk_score for analysis in edge_analyses)
        
        # Add compounding factor for multiple high-risk edges
        high_risk_count = sum(
            1 for analysis in edge_analyses
            if analysis.risk_level >= MEVRiskLevel.HIGH
        )
        
        compounding_factor = 1.0 + (high_risk_count * 0.1)
        
        return min(max_risk * compounding_factor, 1.0)
    
    def _analyze_segment_risks(
        self,
        segments: List[PathSegment],
        edge_analyses: List[EdgeMEVAnalysis]
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """Analyze MEV risks by segment."""
        segment_risks = {}
        edge_lookup = {analysis.edge_id: analysis for analysis in edge_analyses}
        
        highest_risk_score = 0.0
        highest_risk_segment = None
        
        for segment in segments:
            # Calculate segment risk as max of its edges
            segment_edges_risks = []
            for edge in segment.edges:
                if edge.edge_id in edge_lookup:
                    segment_edges_risks.append(edge_lookup[edge.edge_id].final_risk_score)
            
            if segment_edges_risks:
                segment_risk = max(segment_edges_risks)
                segment_risks[segment.segment_id] = segment_risk
                
                if segment_risk > highest_risk_score:
                    highest_risk_score = segment_risk
                    highest_risk_segment = segment.segment_id
        
        return segment_risks, highest_risk_segment
    
    def _generate_execution_strategy(
        self,
        risk_level: MEVRiskLevel,
        edge_analyses: List[EdgeMEVAnalysis],
        segments: Optional[List[PathSegment]] = None
    ) -> Dict[str, Any]:
        """Generate complete execution strategy based on risk analysis."""
        strategy = {
            "method": "public",
            "atomic_required": False,
            "use_flashbots": False,
            "use_private_relay": False,
            "split_execution": False,
            "priority_fee_multiplier": 1.0,
            "max_base_fee_gwei": 100,
            "slippage_buffer": 1.0
        }
        
        # Determine primary execution method
        if risk_level == MEVRiskLevel.MINIMAL:
            strategy["method"] = "public"
        elif risk_level == MEVRiskLevel.LOW:
            strategy["method"] = "public_protected"
            strategy["priority_fee_multiplier"] = 1.2
        elif risk_level == MEVRiskLevel.MEDIUM:
            strategy["method"] = "private_relay"
            strategy["use_private_relay"] = True
            strategy["priority_fee_multiplier"] = 1.5
        elif risk_level >= MEVRiskLevel.HIGH:
            strategy["method"] = "flashbots"
            strategy["use_flashbots"] = True
            strategy["atomic_required"] = True
            strategy["priority_fee_multiplier"] = 2.0
        
        # Check if any edge requires atomic execution
        if any(analysis.risk_level >= MEVRiskLevel.HIGH for analysis in edge_analyses):
            strategy["atomic_required"] = True
        
        # Check if we should split execution
        if segments and len(segments) > 1:
            # If segments have very different risk levels, consider splitting
            if any(s.segment_type.value == "non_atomic" for s in segments):
                strategy["split_execution"] = True
        
        # Adjust slippage buffer based on risk
        if risk_level >= MEVRiskLevel.HIGH:
            strategy["slippage_buffer"] = 0.5  # Tighter slippage for high MEV risk
        
        return strategy


# Convenience functions

def calculate_edge_mev_risk(
    edge: YieldGraphEdge,
    amount_usd: float = 10_000
) -> float:
    """
    Quick function to calculate MEV risk score for an edge.
    
    Args:
        edge: The edge to analyze
        amount_usd: Trade size in USD
        
    Returns:
        MEV risk score from 0 to 1
    """
    assessor = MEVRiskAssessor()
    analysis = assessor.assess_edge_risk(edge, amount_usd)
    return analysis.final_risk_score


def assess_path_mev_risk(
    path: List[YieldGraphEdge],
    amount_usd: float = 10_000
) -> PathMEVAnalysis:
    """
    Assess MEV risk for a complete path.
    
    Args:
        path: List of edges in the path
        amount_usd: Initial trade size
        
    Returns:
        Complete path MEV analysis
    """
    assessor = MEVRiskAssessor()
    return assessor.assess_path_risk(path, amount_usd)