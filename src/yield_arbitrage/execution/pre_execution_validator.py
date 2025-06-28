"""
Pre-Execution Validation System using Tenderly Integration.

This module provides comprehensive validation of router contract execution
before mainnet deployment, including gas estimation, atomic execution validation,
and profitability analysis.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from enum import Enum

from yield_arbitrage.execution.router_simulator import (
    RouterSimulator, RouterSimulationParams, BatchSimulationResult,
    SimulationStatus, TenderlyNetworkId
)
from yield_arbitrage.pathfinding.path_segment_analyzer import (
    PathSegment, PathSegmentAnalyzer, SegmentType
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge
from yield_arbitrage.execution.calldata_generator import CalldataGenerator

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Result of pre-execution validation."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during analysis."""
    severity: str  # "error", "warning", "info"
    category: str  # "gas", "atomicity", "profitability", "safety"
    message: str
    segment_id: Optional[str] = None
    edge_id: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ExecutionValidationReport:
    """Comprehensive validation report for router execution."""
    
    # Overall result
    validation_result: ValidationResult
    total_segments: int
    valid_segments: int
    
    # Gas analysis
    estimated_gas_usage: int
    gas_cost_at_20_gwei: float
    max_gas_limit_required: int
    
    # Atomicity analysis
    atomic_segments: int
    non_atomic_segments: int
    atomicity_issues: List[str] = field(default_factory=list)
    
    # Profitability analysis
    estimated_profit_usd: float = 0.0
    break_even_gas_price_gwei: Optional[float] = None
    profit_margin_percentage: Optional[float] = None
    
    # Risk analysis
    mev_risk_score: float = 0.0
    slippage_risk_score: float = 0.0
    execution_complexity_score: float = 0.0
    
    # Issues and recommendations
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    
    # Performance metrics
    validation_time_ms: float = 0.0
    simulation_success_rate: float = 0.0
    
    # Raw simulation data
    batch_simulation_result: Optional[BatchSimulationResult] = None


class PreExecutionValidator:
    """
    Comprehensive pre-execution validation system for router contracts.
    
    This class analyzes complete arbitrage paths, validates atomic execution
    requirements, estimates gas costs and profitability, and provides
    detailed recommendations before mainnet deployment.
    """
    
    def __init__(
        self,
        router_simulator: RouterSimulator,
        path_analyzer: PathSegmentAnalyzer,
        min_profit_usd: float = 50.0,
        max_gas_price_gwei: float = 100.0,
        max_segment_gas: int = 8_000_000
    ):
        """
        Initialize the pre-execution validator.
        
        Args:
            router_simulator: Router contract simulator
            path_analyzer: Path segment analyzer
            min_profit_usd: Minimum profit threshold in USD
            max_gas_price_gwei: Maximum acceptable gas price
            max_segment_gas: Maximum gas per segment
        """
        self.router_simulator = router_simulator
        self.path_analyzer = path_analyzer
        self.min_profit_usd = min_profit_usd
        self.max_gas_price_gwei = max_gas_price_gwei
        self.max_segment_gas = max_segment_gas
        
        # Validation thresholds
        self.thresholds = {
            "max_mev_risk": 0.8,
            "max_slippage_risk": 0.5,
            "min_success_rate": 90.0,  # percentage
            "max_complexity_score": 0.7,
        }
    
    async def validate_execution_plan(
        self,
        edges: List[YieldGraphEdge],
        input_amount: Decimal,
        simulation_params: RouterSimulationParams
    ) -> ExecutionValidationReport:
        """
        Validate a complete execution plan for router deployment.
        
        Args:
            edges: List of graph edges representing the arbitrage path
            input_amount: Initial input amount for the arbitrage
            simulation_params: Parameters for Tenderly simulation
            
        Returns:
            ExecutionValidationReport with comprehensive analysis
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Starting validation for {len(edges)} edges with input ${input_amount}")
        
        # Analyze path segments
        segments = self.path_analyzer.analyze_path(edges)
        
        logger.info(f"Path analyzed into {len(segments)} segments")
        
        # Initialize report
        report = ExecutionValidationReport(
            validation_result=ValidationResult.VALID,
            total_segments=len(segments),
            valid_segments=0,
            estimated_gas_usage=0,
            gas_cost_at_20_gwei=0.0,
            max_gas_limit_required=0
        )
        
        # Validate each segment type
        atomicity_validation = await self._validate_atomicity(segments, report)
        
        # Simulate execution
        if atomicity_validation:
            simulation_result = await self._simulate_execution(
                segments, simulation_params, report
            )
            
            # Analyze profitability
            await self._analyze_profitability(
                segments, input_amount, simulation_params, report
            )
            
            # Analyze risks
            await self._analyze_risks(segments, report)
        
        # Determine final validation result
        report.validation_result = self._determine_final_result(report)
        
        # Calculate performance metrics
        report.validation_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.info(
            f"Validation complete: {report.validation_result.value} "
            f"({len(report.validation_issues)} issues found)"
        )
        
        return report
    
    async def validate_gas_efficiency(
        self,
        segments: List[PathSegment],
        simulation_params: RouterSimulationParams
    ) -> Dict[str, Any]:
        """
        Validate gas efficiency of execution plan.
        
        Args:
            segments: Path segments to validate
            simulation_params: Simulation parameters
            
        Returns:
            Dictionary with gas efficiency analysis
        """
        # Simulate at different gas prices
        gas_prices = [10, 20, 50, 100, 200]  # gwei
        efficiency_data = []
        
        for gas_price in gas_prices:
            params = RouterSimulationParams(
                router_contract_address=simulation_params.router_contract_address,
                network_id=simulation_params.network_id,
                gas_price_gwei=gas_price,
                executor_address=simulation_params.executor_address
            )
            
            batch_result = await self.router_simulator.simulate_batch_execution(
                segments, params
            )
            
            efficiency_data.append({
                "gas_price_gwei": gas_price,
                "total_gas_used": batch_result.total_gas_used,
                "gas_cost_usd": batch_result.total_gas_cost_usd,
                "success_rate": batch_result.success_rate,
                "estimated_profit": float(batch_result.estimated_profit)
            })
        
        return {
            "segments_analyzed": len(segments),
            "gas_efficiency_curve": efficiency_data,
            "optimal_gas_price": self._find_optimal_gas_price(efficiency_data),
            "gas_elasticity": self._calculate_gas_elasticity(efficiency_data)
        }
    
    async def validate_flash_loan_requirements(
        self,
        segments: List[PathSegment]
    ) -> Dict[str, Any]:
        """
        Validate flash loan requirements and constraints.
        
        Args:
            segments: Path segments to analyze
            
        Returns:
            Dictionary with flash loan validation results
        """
        flash_loan_segments = [s for s in segments if s.requires_flash_loan]
        
        validation_results = {
            "total_segments": len(segments),
            "flash_loan_segments": len(flash_loan_segments),
            "flash_loan_requirements": [],
            "validation_issues": []
        }
        
        for segment in flash_loan_segments:
            requirement = {
                "segment_id": segment.segment_id,
                "asset": segment.flash_loan_asset,
                "amount": float(segment.flash_loan_amount or 0),
                "estimated_fee": self._estimate_flash_loan_fee(segment)
            }
            
            # Validate flash loan constraints
            if not segment.is_atomic:
                validation_results["validation_issues"].append(
                    f"Flash loan segment {segment.segment_id} is not atomic"
                )
            
            if segment.flash_loan_amount and segment.flash_loan_amount <= 0:
                validation_results["validation_issues"].append(
                    f"Invalid flash loan amount for segment {segment.segment_id}"
                )
            
            validation_results["flash_loan_requirements"].append(requirement)
        
        return validation_results
    
    async def _validate_atomicity(
        self,
        segments: List[PathSegment],
        report: ExecutionValidationReport
    ) -> bool:
        """Validate atomicity requirements."""
        
        atomic_count = 0
        non_atomic_count = 0
        
        for segment in segments:
            if segment.is_atomic:
                atomic_count += 1
            else:
                non_atomic_count += 1
                report.atomicity_issues.append(
                    f"Segment {segment.segment_id} is not atomic "
                    f"(type: {segment.segment_type.value})"
                )
                
                # Add validation issue
                report.validation_issues.append(ValidationIssue(
                    severity="error",
                    category="atomicity",
                    message=f"Non-atomic segment detected: {segment.segment_id}",
                    segment_id=segment.segment_id,
                    suggested_fix="Consider splitting path or using flash loans"
                ))
        
        report.atomic_segments = atomic_count
        report.non_atomic_segments = non_atomic_count
        
        return non_atomic_count == 0
    
    async def _simulate_execution(
        self,
        segments: List[PathSegment],
        simulation_params: RouterSimulationParams,
        report: ExecutionValidationReport
    ) -> bool:
        """Simulate execution and update report."""
        
        try:
            batch_result = await self.router_simulator.simulate_batch_execution(
                segments, simulation_params
            )
            
            report.batch_simulation_result = batch_result
            report.estimated_gas_usage = batch_result.total_gas_used
            report.gas_cost_at_20_gwei = self._calculate_gas_cost(
                batch_result.total_gas_used, 20.0
            )
            report.simulation_success_rate = batch_result.success_rate
            report.valid_segments = batch_result.successful_segments
            
            # Check for gas issues
            if batch_result.total_gas_used > self.max_segment_gas:
                report.validation_issues.append(ValidationIssue(
                    severity="error",
                    category="gas",
                    message=f"Total gas usage ({batch_result.total_gas_used:,}) exceeds limit",
                    suggested_fix=f"Consider splitting into smaller segments"
                ))
            
            # Check success rate
            if batch_result.success_rate < self.thresholds["min_success_rate"]:
                report.validation_issues.append(ValidationIssue(
                    severity="warning",
                    category="reliability",
                    message=f"Low success rate: {batch_result.success_rate:.1f}%",
                    suggested_fix="Review failed segments and optimize parameters"
                ))
            
            return batch_result.success_rate > 0
            
        except Exception as e:
            report.validation_issues.append(ValidationIssue(
                severity="error",
                category="simulation",
                message=f"Simulation failed: {str(e)}",
                suggested_fix="Check Tenderly configuration and network connectivity"
            ))
            return False
    
    async def _analyze_profitability(
        self,
        segments: List[PathSegment],
        input_amount: Decimal,
        simulation_params: RouterSimulationParams,
        report: ExecutionValidationReport
    ):
        """Analyze profitability metrics."""
        
        # Get profitability analysis for the first segment (representative)
        if segments:
            profitability = await self.router_simulator.estimate_profitability(
                segments[0], simulation_params, input_amount
            )
            
            # Extract metrics from analysis
            if profitability["gas_price_analysis"]:
                analysis_20_gwei = next(
                    (a for a in profitability["gas_price_analysis"] if a["gas_price_gwei"] == 20.0),
                    profitability["gas_price_analysis"][0]
                )
                
                report.estimated_profit_usd = analysis_20_gwei["net_profit_usd"]
                
                # Find break-even gas price
                for analysis in profitability["gas_price_analysis"]:
                    if analysis["net_profit_usd"] <= 0:
                        report.break_even_gas_price_gwei = analysis["gas_price_gwei"]
                        break
                
                # Calculate profit margin
                if analysis_20_gwei["gas_cost_usd"] > 0:
                    gross_profit = analysis_20_gwei["net_profit_usd"] + analysis_20_gwei["gas_cost_usd"]
                    if gross_profit > 0:
                        report.profit_margin_percentage = (
                            analysis_20_gwei["net_profit_usd"] / gross_profit
                        ) * 100
        
        # Check profitability thresholds
        if report.estimated_profit_usd < self.min_profit_usd:
            report.validation_issues.append(ValidationIssue(
                severity="warning",
                category="profitability",
                message=f"Low estimated profit: ${report.estimated_profit_usd:.2f}",
                suggested_fix=f"Consider increasing input amount or finding better routes"
            ))
    
    async def _analyze_risks(
        self,
        segments: List[PathSegment],
        report: ExecutionValidationReport
    ):
        """Analyze execution risks."""
        
        total_mev_risk = 0.0
        total_slippage_risk = 0.0
        complexity_factors = []
        
        for segment in segments:
            # Calculate MEV risk based on segment properties
            mev_risk = segment.total_mev_sensitivity
            total_mev_risk += mev_risk
            
            # Calculate slippage risk (placeholder)
            slippage_risk = 0.1 * segment.edge_count  # Simplified
            total_slippage_risk += slippage_risk
            
            # Calculate complexity factors
            complexity_factors.append(segment.edge_count)
            
            if segment.requires_flash_loan:
                complexity_factors.append(2)  # Flash loans add complexity
        
        # Normalize risk scores
        report.mev_risk_score = min(total_mev_risk / len(segments), 1.0)
        report.slippage_risk_score = min(total_slippage_risk / len(segments), 1.0)
        report.execution_complexity_score = min(sum(complexity_factors) / (len(segments) * 3), 1.0)
        
        # Check risk thresholds
        if report.mev_risk_score > self.thresholds["max_mev_risk"]:
            report.validation_issues.append(ValidationIssue(
                severity="warning",
                category="mev",
                message=f"High MEV risk score: {report.mev_risk_score:.2f}",
                suggested_fix="Consider using private mempool or flashbots"
            ))
        
        if report.execution_complexity_score > self.thresholds["max_complexity_score"]:
            report.validation_issues.append(ValidationIssue(
                severity="info",
                category="complexity",
                message=f"High execution complexity: {report.execution_complexity_score:.2f}",
                suggested_fix="Consider simplifying the execution path"
            ))
    
    def _determine_final_result(self, report: ExecutionValidationReport) -> ValidationResult:
        """Determine the final validation result based on all factors."""
        
        # Count issues by severity
        errors = len([i for i in report.validation_issues if i.severity == "error"])
        warnings = len([i for i in report.validation_issues if i.severity == "warning"])
        
        if errors > 0:
            return ValidationResult.INVALID
        elif warnings > 3:  # Multiple warnings require review
            return ValidationResult.REQUIRES_REVIEW
        elif warnings > 0:
            return ValidationResult.WARNING
        else:
            return ValidationResult.VALID
    
    def _calculate_gas_cost(self, gas_used: int, gas_price_gwei: float) -> float:
        """Calculate gas cost in USD."""
        eth_price_usd = 2000.0  # Placeholder - would use real price feed
        gas_cost_eth = (gas_used * gas_price_gwei * 1e9) / 1e18
        return gas_cost_eth * eth_price_usd
    
    def _find_optimal_gas_price(self, efficiency_data: List[Dict]) -> float:
        """Find optimal gas price based on profit maximization."""
        best_profit = float('-inf')
        optimal_price = 20.0
        
        for data in efficiency_data:
            if data["estimated_profit"] > best_profit:
                best_profit = data["estimated_profit"]
                optimal_price = data["gas_price_gwei"]
        
        return optimal_price
    
    def _calculate_gas_elasticity(self, efficiency_data: List[Dict]) -> float:
        """Calculate gas price elasticity of profit."""
        # Simplified elasticity calculation
        if len(efficiency_data) < 2:
            return 0.0
        
        first = efficiency_data[0]
        last = efficiency_data[-1]
        
        price_change = (last["gas_price_gwei"] - first["gas_price_gwei"]) / first["gas_price_gwei"]
        profit_change = (last["estimated_profit"] - first["estimated_profit"]) / abs(first["estimated_profit"] + 1)
        
        return profit_change / price_change if price_change != 0 else 0.0
    
    def _estimate_flash_loan_fee(self, segment: PathSegment) -> float:
        """Estimate flash loan fee for a segment."""
        if not segment.requires_flash_loan or not segment.flash_loan_amount:
            return 0.0
        
        # Typical Aave flash loan fee is 0.09%
        fee_rate = 0.0009
        return float(segment.flash_loan_amount) * fee_rate