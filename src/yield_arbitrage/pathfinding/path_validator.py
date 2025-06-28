"""Path validation algorithms for yield arbitrage pathfinding."""
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from ..graph_engine.models import UniversalYieldGraph, YieldGraphEdge, EdgeState, EdgeType
from .path_models import SearchPath, PathNode, PathStatus

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Validation result types."""
    VALID = "valid"
    INVALID_CYCLE = "invalid_cycle"
    INVALID_LENGTH = "invalid_length"
    INVALID_CONFIDENCE = "invalid_confidence"
    INVALID_LIQUIDITY = "invalid_liquidity"
    INVALID_CONSTRAINTS = "invalid_constraints"
    INVALID_GAS_COST = "invalid_gas_cost"
    INVALID_PROFIT = "invalid_profit"


@dataclass
class ValidationConfig:
    """Configuration for path validation."""
    # Basic path constraints
    max_path_length: int = 6
    min_path_length: int = 2
    min_confidence_threshold: float = 0.5
    min_cumulative_confidence: float = 0.3
    
    # Economic constraints
    min_profit_threshold: float = 0.001  # $0.001 minimum profit
    max_gas_to_profit_ratio: float = 0.5  # Gas cost ≤ 50% of profit
    min_liquidity_per_edge: float = 1000.0  # $1000 minimum per edge
    max_price_impact: float = 0.05  # 5% maximum price impact
    
    # Risk constraints
    max_slippage_tolerance: float = 0.02  # 2% maximum slippage
    max_execution_time_estimate: float = 300.0  # 5 minutes max execution
    min_success_probability: float = 0.7  # 70% minimum success probability
    
    # Cycle detection
    allow_immediate_cycles: bool = True   # Allow A->B->A paths (arbitrage)
    allow_complex_cycles: bool = False   # Allow A->B->C->A paths
    max_asset_revisits: int = 2  # Maximum times an asset can appear (for arbitrage)
    
    # Advanced validation
    validate_edge_compatibility: bool = True
    validate_temporal_consistency: bool = True
    validate_cross_chain_constraints: bool = True


@dataclass
class ValidationReport:
    """Detailed validation report for a path."""
    is_valid: bool
    result: ValidationResult
    errors: List[str]
    warnings: List[str]
    
    # Metrics
    path_score: float
    confidence_score: float
    liquidity_score: float
    risk_score: float
    
    # Detailed analysis
    cycle_analysis: Dict[str, Any]
    constraint_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class PathValidator:
    """
    Comprehensive path validation system for yield arbitrage paths.
    
    This class implements validation logic to ensure paths are:
    - Cycle-free (or controlled cycles)
    - Within length and confidence constraints
    - Economically viable
    - Risk-appropriate
    - Technically executable
    """
    
    def __init__(self, config: ValidationConfig = None):
        """
        Initialize the path validator.
        
        Args:
            config: Validation configuration parameters
        """
        self.config = config or ValidationConfig()
        
        # Validation statistics
        self._validation_stats = {
            "paths_validated": 0,
            "valid_paths": 0,
            "validation_results": {result.value: 0 for result in ValidationResult}
        }
    
    async def validate_path(
        self, 
        path: SearchPath, 
        graph: UniversalYieldGraph,
        target_asset_id: str,
        initial_amount: float = 1.0
    ) -> ValidationReport:
        """
        Perform comprehensive validation of a path.
        
        Args:
            path: The path to validate
            graph: The yield graph for context
            target_asset_id: Target asset for arbitrage completion
            initial_amount: Initial amount for economic calculations
            
        Returns:
            Detailed validation report
        """
        self._validation_stats["paths_validated"] += 1
        
        errors = []
        warnings = []
        
        # 1. Basic Structure Validation
        structure_result = self._validate_basic_structure(path)
        if structure_result["errors"]:
            errors.extend(structure_result["errors"])
        warnings.extend(structure_result["warnings"])
        
        # 2. Cycle Detection
        cycle_analysis = self._analyze_cycles(path)
        if cycle_analysis["has_invalid_cycles"]:
            errors.append(f"Invalid cycles detected: {cycle_analysis['cycle_description']}")
        
        # 3. Length Validation
        length_result = self._validate_path_length(path)
        if not length_result["valid"]:
            errors.append(length_result["error"])
        
        # 4. Confidence Validation
        confidence_result = self._validate_confidence_requirements(path)
        if not confidence_result["valid"]:
            errors.append(confidence_result["error"])
        warnings.extend(confidence_result["warnings"])
        
        # 5. Economic Validation
        economic_result = await self._validate_economic_constraints(
            path, target_asset_id, initial_amount
        )
        if not economic_result["valid"]:
            errors.extend(economic_result["errors"])
        warnings.extend(economic_result["warnings"])
        
        # 6. Liquidity and Market Impact Validation
        liquidity_result = await self._validate_liquidity_constraints(path, initial_amount)
        if not liquidity_result["valid"]:
            errors.extend(liquidity_result["errors"])
        warnings.extend(liquidity_result["warnings"])
        
        # 7. Risk Assessment
        risk_analysis = await self._assess_path_risks(path, graph)
        if risk_analysis["risk_score"] > 0.8:  # High risk threshold
            warnings.append(f"High risk path detected: {risk_analysis['risk_factors']}")
        
        # 8. Technical Validation
        if self.config.validate_edge_compatibility:
            tech_result = await self._validate_technical_constraints(path, graph)
            if not tech_result["valid"]:
                errors.extend(tech_result["errors"])
            warnings.extend(tech_result["warnings"])
        
        # Determine overall result
        if errors:
            is_valid = False
            # Determine primary failure reason
            if any("cycle" in error.lower() for error in errors):
                result = ValidationResult.INVALID_CYCLE
            elif any("length" in error.lower() for error in errors):
                result = ValidationResult.INVALID_LENGTH
            elif any("confidence" in error.lower() for error in errors):
                result = ValidationResult.INVALID_CONFIDENCE
            elif any("liquidity" in error.lower() for error in errors):
                result = ValidationResult.INVALID_LIQUIDITY
            elif any("gas" in error.lower() for error in errors):
                result = ValidationResult.INVALID_GAS_COST
            elif any("profit" in error.lower() for error in errors):
                result = ValidationResult.INVALID_PROFIT
            else:
                result = ValidationResult.INVALID_CONSTRAINTS
        else:
            is_valid = True
            result = ValidationResult.VALID
            self._validation_stats["valid_paths"] += 1
        
        # Update statistics
        self._validation_stats["validation_results"][result.value] += 1
        
        # Calculate scores
        confidence_score = confidence_result.get("score", 0.0)
        liquidity_score = liquidity_result.get("score", 0.0)
        risk_score = risk_analysis.get("risk_score", 1.0)
        
        # Overall path score (0-1, higher is better)
        path_score = self._calculate_overall_score(
            confidence_score, liquidity_score, risk_score, len(errors), len(warnings)
        )
        
        return ValidationReport(
            is_valid=is_valid,
            result=result,
            errors=errors,
            warnings=warnings,
            path_score=path_score,
            confidence_score=confidence_score,
            liquidity_score=liquidity_score,
            risk_score=risk_score,
            cycle_analysis=cycle_analysis,
            constraint_analysis={
                "length": length_result,
                "confidence": confidence_result,
                "economic": economic_result,
                "liquidity": liquidity_result
            },
            risk_analysis=risk_analysis,
            performance_metrics=self._calculate_performance_metrics(path)
        )
    
    def _validate_basic_structure(self, path: SearchPath) -> Dict[str, Any]:
        """Validate basic path structure."""
        errors = []
        warnings = []
        
        # Check if path has nodes
        if not path.nodes:
            errors.append("Path has no nodes")
            return {"errors": errors, "warnings": warnings}
        
        # Check node consistency
        for i, node in enumerate(path.nodes[1:], 1):
            prev_node = path.nodes[i-1]
            
            # Check edge path consistency
            if len(node.edge_path) != len(prev_node.edge_path) + 1:
                errors.append(f"Edge path inconsistency at node {i}")
            
            # Check amount positivity
            if node.amount <= 0:
                errors.append(f"Invalid amount at node {i}: {node.amount}")
            
            # Check confidence decay
            if node.confidence_accumulated > prev_node.confidence_accumulated:
                warnings.append(f"Confidence increased at node {i} (unusual)")
        
        return {"errors": errors, "warnings": warnings}
    
    def _analyze_cycles(self, path: SearchPath) -> Dict[str, Any]:
        """Analyze path for cycles and determine if they're valid."""
        asset_positions = {}
        cycles_found = []
        
        for i, node in enumerate(path.nodes):
            asset_id = node.asset_id
            
            if asset_id in asset_positions:
                # Found a cycle
                cycle_start = asset_positions[asset_id]
                cycle_length = i - cycle_start
                cycles_found.append({
                    "asset": asset_id,
                    "start_position": cycle_start,
                    "end_position": i,
                    "length": cycle_length
                })
            
            asset_positions[asset_id] = i
        
        # Analyze cycle validity
        has_invalid_cycles = False
        cycle_descriptions = []
        
        for cycle in cycles_found:
            is_immediate = cycle["length"] == 1
            is_simple = cycle["length"] == 2
            
            if is_immediate and not self.config.allow_immediate_cycles:
                has_invalid_cycles = True
                cycle_descriptions.append(f"Immediate cycle at {cycle['asset']}")
            elif not is_immediate and not is_simple and not self.config.allow_complex_cycles:
                has_invalid_cycles = True
                cycle_descriptions.append(f"Complex cycle at {cycle['asset']} (length {cycle['length']})")
        
        # Check asset revisit limits
        asset_counts = {}
        for node in path.nodes:
            asset_counts[node.asset_id] = asset_counts.get(node.asset_id, 0) + 1
        
        for asset, count in asset_counts.items():
            if count > self.config.max_asset_revisits:
                has_invalid_cycles = True
                cycle_descriptions.append(f"Asset {asset} visited {count} times (max {self.config.max_asset_revisits})")
        
        return {
            "has_cycles": len(cycles_found) > 0,
            "has_invalid_cycles": has_invalid_cycles,
            "cycles_found": cycles_found,
            "cycle_description": "; ".join(cycle_descriptions),
            "asset_visit_counts": asset_counts
        }
    
    def _validate_path_length(self, path: SearchPath) -> Dict[str, Any]:
        """Validate path length constraints."""
        length = path.path_length
        
        if length < self.config.min_path_length:
            return {
                "valid": False,
                "error": f"Path too short: {length} edges (minimum {self.config.min_path_length})"
            }
        
        if length > self.config.max_path_length:
            return {
                "valid": False,
                "error": f"Path too long: {length} edges (maximum {self.config.max_path_length})"
            }
        
        return {"valid": True}
    
    def _validate_confidence_requirements(self, path: SearchPath) -> Dict[str, Any]:
        """Validate confidence requirements."""
        warnings = []
        
        if not path.nodes:
            return {"valid": False, "error": "No nodes in path", "warnings": warnings}
        
        # Check minimum confidence per edge
        min_edge_confidence = float('inf')
        for node in path.nodes:
            if node.confidence_accumulated < min_edge_confidence:
                min_edge_confidence = node.confidence_accumulated
        
        if min_edge_confidence < self.config.min_confidence_threshold:
            return {
                "valid": False,
                "error": f"Edge confidence too low: {min_edge_confidence:.3f} (minimum {self.config.min_confidence_threshold})",
                "warnings": warnings
            }
        
        # Check cumulative confidence
        final_confidence = path.nodes[-1].confidence_accumulated
        if final_confidence < self.config.min_cumulative_confidence:
            return {
                "valid": False,
                "error": f"Cumulative confidence too low: {final_confidence:.3f} (minimum {self.config.min_cumulative_confidence})",
                "warnings": warnings
            }
        
        # Add warnings for borderline confidence
        if final_confidence < 0.7:
            warnings.append(f"Low cumulative confidence: {final_confidence:.3f}")
        
        if min_edge_confidence < 0.8:
            warnings.append(f"Low minimum edge confidence: {min_edge_confidence:.3f}")
        
        # Calculate confidence score
        confidence_score = (final_confidence + min_edge_confidence) / 2.0
        
        return {
            "valid": True,
            "warnings": warnings,
            "score": confidence_score,
            "final_confidence": final_confidence,
            "min_edge_confidence": min_edge_confidence
        }
    
    async def _validate_economic_constraints(
        self, 
        path: SearchPath, 
        target_asset_id: str, 
        initial_amount: float
    ) -> Dict[str, Any]:
        """Validate economic viability constraints."""
        errors = []
        warnings = []
        
        if not path.nodes:
            return {"valid": False, "errors": ["No nodes in path"], "warnings": warnings}
        
        # Calculate profit metrics
        final_amount = path.final_amount
        gas_cost = path.total_gas_cost
        
        if path.end_asset == target_asset_id:
            # Complete arbitrage path
            gross_profit = final_amount - initial_amount
            net_profit = gross_profit - gas_cost
        else:
            # Partial path - estimate completion
            estimated_return = final_amount * 0.95  # Conservative estimate
            gross_profit = estimated_return - initial_amount
            net_profit = gross_profit - gas_cost
        
        # Validate minimum profit
        if net_profit < self.config.min_profit_threshold:
            errors.append(f"Insufficient profit: {net_profit:.4f} (minimum {self.config.min_profit_threshold})")
        
        # Validate gas cost ratio
        if gross_profit > 0:
            gas_ratio = gas_cost / gross_profit
            if gas_ratio > self.config.max_gas_to_profit_ratio:
                errors.append(f"Gas cost too high: {gas_ratio:.2%} of profit (maximum {self.config.max_gas_to_profit_ratio:.2%})")
        else:
            # If no profit, any gas cost is too much
            if gas_cost > 0:
                errors.append("Gas cost present but no profit generated")
        
        # Add warnings for borderline cases
        if net_profit < self.config.min_profit_threshold * 2:
            warnings.append(f"Low profit margin: {net_profit:.4f}")
        
        profit_ratio = net_profit / initial_amount if initial_amount > 0 else 0
        if profit_ratio < 0.005:  # Less than 0.5% profit
            warnings.append(f"Low profit ratio: {profit_ratio:.2%}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "gas_cost": gas_cost,
            "profit_ratio": profit_ratio
        }
    
    async def _validate_liquidity_constraints(
        self, 
        path: SearchPath, 
        trade_amount: float
    ) -> Dict[str, Any]:
        """Validate liquidity and market impact constraints."""
        errors = []
        warnings = []
        
        total_liquidity = 0.0
        min_edge_liquidity = float('inf')
        max_price_impact = 0.0
        liquidity_scores = []
        
        for node in path.nodes[1:]:  # Skip first node
            # Mock liquidity data - in production would get from EdgeStateManager
            edge_liquidity = 1_000_000.0  # Default $1M liquidity
            
            total_liquidity += edge_liquidity
            min_edge_liquidity = min(min_edge_liquidity, edge_liquidity)
            
            # Estimate price impact
            if edge_liquidity > 0:
                impact = min(trade_amount / edge_liquidity, 0.5)  # Cap at 50%
                max_price_impact = max(max_price_impact, impact)
                
                # Liquidity score (0-1)
                score = math.exp(-trade_amount / (edge_liquidity * 0.01))  # 1% impact = ~0.37 score
                liquidity_scores.append(score)
            else:
                max_price_impact = 1.0  # 100% impact if no liquidity
                liquidity_scores.append(0.0)
        
        if min_edge_liquidity == float('inf'):
            min_edge_liquidity = 0.0
        
        # Validate minimum liquidity per edge
        if min_edge_liquidity < self.config.min_liquidity_per_edge:
            errors.append(f"Insufficient edge liquidity: {min_edge_liquidity:.0f} (minimum {self.config.min_liquidity_per_edge:.0f})")
        
        # Validate maximum price impact
        if max_price_impact > self.config.max_price_impact:
            errors.append(f"Price impact too high: {max_price_impact:.2%} (maximum {self.config.max_price_impact:.2%})")
        
        # Add warnings for borderline liquidity
        if min_edge_liquidity < self.config.min_liquidity_per_edge * 2:
            warnings.append(f"Low edge liquidity: {min_edge_liquidity:.0f}")
        
        if max_price_impact > self.config.max_price_impact * 0.5:
            warnings.append(f"High price impact: {max_price_impact:.2%}")
        
        # Calculate overall liquidity score
        if liquidity_scores:
            liquidity_score = sum(liquidity_scores) / len(liquidity_scores)
        else:
            liquidity_score = 0.0
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "score": liquidity_score,
            "total_liquidity": total_liquidity,
            "min_edge_liquidity": min_edge_liquidity,
            "max_price_impact": max_price_impact
        }
    
    async def _assess_path_risks(self, path: SearchPath, graph: UniversalYieldGraph) -> Dict[str, Any]:
        """Assess various risk factors for the path."""
        risk_factors = []
        risk_score = 0.0
        
        # Path length risk
        if path.path_length > 4:
            length_risk = (path.path_length - 4) * 0.1
            risk_score += length_risk
            risk_factors.append(f"Long path ({path.path_length} edges)")
        
        # Confidence risk
        final_confidence = path.nodes[-1].confidence_accumulated if path.nodes else 0
        if final_confidence < 0.8:
            confidence_risk = (0.8 - final_confidence) * 0.5
            risk_score += confidence_risk
            risk_factors.append(f"Low confidence ({final_confidence:.2f})")
        
        # Gas cost risk
        if path.nodes:
            gas_ratio = path.total_gas_cost / path.nodes[0].amount
            if gas_ratio > 0.02:  # Gas > 2% of initial amount
                gas_risk = gas_ratio * 2.0
                risk_score += gas_risk
                risk_factors.append(f"High gas cost ({gas_ratio:.2%} of amount)")
        
        # Execution complexity risk
        complexity_score = self._calculate_execution_complexity(path)
        if complexity_score > 0.7:
            complexity_risk = (complexity_score - 0.7) * 0.3
            risk_score += complexity_risk
            risk_factors.append(f"High execution complexity ({complexity_score:.2f})")
        
        # Time-based risk (older paths are riskier)
        if path.creation_time:
            age_minutes = (time.time() - path.creation_time) / 60.0
            if age_minutes > 5:  # Paths older than 5 minutes
                time_risk = min(age_minutes / 60.0, 0.5)  # Cap at 0.5
                risk_score += time_risk
                risk_factors.append(f"Stale path ({age_minutes:.1f} minutes old)")
        
        # Normalize risk score to 0-1
        risk_score = min(1.0, risk_score)
        
        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "complexity_score": complexity_score
        }
    
    async def _validate_technical_constraints(
        self, 
        path: SearchPath, 
        graph: UniversalYieldGraph
    ) -> Dict[str, Any]:
        """Validate technical execution constraints."""
        errors = []
        warnings = []
        
        # Cross-chain validation
        if self.config.validate_cross_chain_constraints:
            chain_result = self._validate_cross_chain_consistency(path, graph)
            errors.extend(chain_result["errors"])
            warnings.extend(chain_result["warnings"])
        
        # Temporal consistency
        if self.config.validate_temporal_consistency:
            temporal_result = self._validate_temporal_consistency(path)
            errors.extend(temporal_result["errors"])
            warnings.extend(temporal_result["warnings"])
        
        # Edge compatibility
        if self.config.validate_edge_compatibility:
            compatibility_result = self._validate_edge_compatibility(path, graph)
            errors.extend(compatibility_result["errors"])
            warnings.extend(compatibility_result["warnings"])
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_cross_chain_consistency(
        self, 
        path: SearchPath, 
        graph: UniversalYieldGraph
    ) -> Dict[str, Any]:
        """Validate cross-chain transaction consistency."""
        errors = []
        warnings = []
        
        # Track chains used in the path
        chains_encountered = set()
        chain_transitions = 0
        
        for node in path.nodes[1:]:
            if node.edge_path:
                edge_id = node.edge_path[-1]
                edge = graph.get_edge(edge_id)
                if edge:
                    current_chain = edge.chain_name
                    if current_chain not in chains_encountered:
                        if chains_encountered:  # Not the first chain
                            chain_transitions += 1
                        chains_encountered.add(current_chain)
        
        # Validate chain transition limits
        if chain_transitions > 2:  # More than 2 chain transitions
            warnings.append(f"Multiple chain transitions ({chain_transitions})")
        
        # Validate supported chains
        unsupported_chains = chains_encountered - {"ethereum", "polygon", "arbitrum", "optimism"}
        if unsupported_chains:
            errors.append(f"Unsupported chains: {unsupported_chains}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_temporal_consistency(self, path: SearchPath) -> Dict[str, Any]:
        """Validate temporal execution constraints."""
        errors = []
        warnings = []
        
        # Estimate execution time based on path complexity
        estimated_time = self._estimate_execution_time(path)
        
        if estimated_time > self.config.max_execution_time_estimate:
            errors.append(f"Estimated execution time too long: {estimated_time:.0f}s (max {self.config.max_execution_time_estimate:.0f}s)")
        
        if estimated_time > self.config.max_execution_time_estimate * 0.7:
            warnings.append(f"Long estimated execution time: {estimated_time:.0f}s")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_edge_compatibility(
        self, 
        path: SearchPath, 
        graph: UniversalYieldGraph
    ) -> Dict[str, Any]:
        """Validate edge compatibility within the path."""
        errors = []
        warnings = []
        
        # Check protocol compatibility
        protocols_used = set()
        
        for node in path.nodes[1:]:
            if node.edge_path:
                edge_id = node.edge_path[-1]
                edge = graph.get_edge(edge_id)
                if edge:
                    protocols_used.add(edge.protocol_name)
        
        # Some protocol combinations might be problematic
        if "uniswapv2" in protocols_used and "uniswapv3" in protocols_used:
            warnings.append("Mixed Uniswap V2/V3 protocols may have different behaviors")
        
        return {"errors": errors, "warnings": warnings}
    
    def _calculate_execution_complexity(self, path: SearchPath) -> float:
        """Calculate execution complexity score (0-1)."""
        complexity = 0.0
        
        # Length complexity
        complexity += min(path.path_length / 10.0, 0.5)
        
        # Gas cost complexity
        if path.nodes:
            gas_ratio = path.total_gas_cost / path.nodes[0].amount
            complexity += min(gas_ratio * 5.0, 0.3)
        
        # Confidence complexity (lower confidence = higher complexity)
        if path.nodes:
            final_confidence = path.nodes[-1].confidence_accumulated
            complexity += (1.0 - final_confidence) * 0.2
        
        return min(1.0, complexity)
    
    def _estimate_execution_time(self, path: SearchPath) -> float:
        """Estimate execution time in seconds."""
        base_time = 15.0  # Base 15 seconds
        
        # Add time per edge
        edge_time = path.path_length * 5.0
        
        # Add complexity penalty
        complexity = self._calculate_execution_complexity(path)
        complexity_time = complexity * 30.0
        
        return base_time + edge_time + complexity_time
    
    def _calculate_overall_score(
        self, 
        confidence_score: float, 
        liquidity_score: float, 
        risk_score: float,
        error_count: int,
        warning_count: int
    ) -> float:
        """Calculate overall path validation score."""
        if error_count > 0:
            return 0.0
        
        # Base score from metrics
        base_score = (confidence_score * 0.4 + liquidity_score * 0.4 + (1.0 - risk_score) * 0.2)
        
        # Penalty for warnings
        warning_penalty = min(warning_count * 0.05, 0.3)
        
        return max(0.0, base_score - warning_penalty)
    
    def _calculate_performance_metrics(self, path: SearchPath) -> Dict[str, Any]:
        """Calculate path performance metrics."""
        if not path.nodes:
            return {}
        
        initial_amount = path.nodes[0].amount
        final_amount = path.final_amount
        gas_cost = path.total_gas_cost
        
        return {
            "initial_amount": initial_amount,
            "final_amount": final_amount,
            "gross_return": final_amount - initial_amount,
            "net_return": final_amount - initial_amount - gas_cost,
            "return_ratio": (final_amount - initial_amount) / initial_amount if initial_amount > 0 else 0,
            "gas_efficiency": 1.0 / (1.0 + gas_cost / initial_amount) if initial_amount > 0 else 0,
            "confidence_preservation": path.nodes[-1].confidence_accumulated,
            "path_efficiency": initial_amount / max(1.0, path.path_length)
        }
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        total_validated = self._validation_stats["paths_validated"]
        valid_rate = (
            (self._validation_stats["valid_paths"] / total_validated * 100)
            if total_validated > 0 else 0.0
        )
        
        return {
            "paths_validated": total_validated,
            "valid_paths": self._validation_stats["valid_paths"],
            "validation_rate": f"{valid_rate:.1f}%",
            "result_distribution": self._validation_stats["validation_results"]
        }
    
    def clear_stats(self) -> None:
        """Clear validation statistics."""
        self._validation_stats = {
            "paths_validated": 0,
            "valid_paths": 0,
            "validation_results": {result.value: 0 for result in ValidationResult}
        }
        logger.info("Validation statistics cleared")