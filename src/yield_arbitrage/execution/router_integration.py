"""
Complete Router Integration Module.

This module provides a unified interface for router-based arbitrage execution,
combining all the components: path analysis, calldata generation, transaction building,
simulation, and validation.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from enum import Enum

from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegmentAnalyzer, PathSegment
from yield_arbitrage.execution.calldata_generator import CalldataGenerator
from yield_arbitrage.execution.enhanced_transaction_builder import (
    EnhancedTransactionBuilder, BatchExecutionPlan, RouterIntegrationMode
)
from yield_arbitrage.execution.router_simulator import RouterSimulator, RouterSimulationParams
from yield_arbitrage.execution.pre_execution_validator import PreExecutionValidator, ExecutionValidationReport
from yield_arbitrage.graph_engine.models import YieldGraphEdge

logger = logging.getLogger(__name__)


class ExecutionStrategy(str, Enum):
    """Available execution strategies."""
    CONSERVATIVE = "conservative"  # Maximize reliability
    AGGRESSIVE = "aggressive"     # Maximize profit
    BALANCED = "balanced"         # Balance profit and reliability
    GAS_OPTIMAL = "gas_optimal"   # Minimize gas costs


@dataclass
class RouterExecutionConfig:
    """Configuration for router execution."""
    router_contract_address: str
    executor_address: str
    recipient_address: Optional[str] = None
    
    # Execution preferences
    strategy: ExecutionStrategy = ExecutionStrategy.BALANCED
    integration_mode: RouterIntegrationMode = RouterIntegrationMode.HYBRID
    
    # Risk parameters
    max_gas_price_gwei: float = 100.0
    min_profit_threshold_usd: float = 50.0
    max_slippage_tolerance: float = 0.005  # 0.5%
    
    # Timing parameters
    execution_deadline_minutes: int = 30
    simulation_timeout_seconds: int = 60
    
    # Flash loan settings
    allow_flash_loans: bool = True
    max_flash_loan_amount_usd: float = 1_000_000.0
    
    # Safety settings
    enable_pre_execution_validation: bool = True
    require_simulation_success: bool = True
    dry_run_mode: bool = False


@dataclass
class ExecutionResult:
    """Result of router execution."""
    execution_id: str
    success: bool
    execution_plan: BatchExecutionPlan
    validation_report: Optional[ExecutionValidationReport] = None
    
    # Execution metrics
    total_gas_used: int = 0
    actual_gas_cost_usd: float = 0.0
    execution_time_seconds: float = 0.0
    
    # Financial results
    profit_loss_usd: float = 0.0
    profit_margin_percentage: float = 0.0
    
    # Transaction details
    transaction_hashes: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Performance metrics
    simulation_accuracy: float = 0.0  # How close simulation was to reality
    gas_efficiency: float = 0.0      # Actual vs estimated gas usage


class RouterExecutionEngine:
    """
    Complete execution engine for router-based arbitrage.
    
    This class orchestrates the entire execution pipeline from path analysis
    to transaction execution, providing a high-level interface for arbitrage operations.
    """
    
    def __init__(
        self,
        config: RouterExecutionConfig,
        tenderly_client=None,
        chain_id: int = 1
    ):
        """
        Initialize the router execution engine.
        
        Args:
            config: Execution configuration
            tenderly_client: Optional Tenderly client for simulation
            chain_id: Target blockchain chain ID
        """
        self.config = config
        self.chain_id = chain_id
        
        # Initialize core components
        self.path_analyzer = PathSegmentAnalyzer()
        self.calldata_generator = CalldataGenerator(chain_id)
        self.transaction_builder = EnhancedTransactionBuilder(
            config.router_contract_address,
            self.calldata_generator,
            chain_id
        )
        
        # Initialize simulation and validation if Tenderly client provided
        if tenderly_client:
            self.router_simulator = RouterSimulator(
                tenderly_client,
                self.calldata_generator,
                config.router_contract_address
            )
            self.pre_execution_validator = PreExecutionValidator(
                self.router_simulator,
                self.path_analyzer,
                config.min_profit_threshold_usd,
                config.max_gas_price_gwei
            )
        else:
            self.router_simulator = None
            self.pre_execution_validator = None
        
        # Execution tracking
        self.execution_history: Dict[str, ExecutionResult] = {}
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_profit_usd": 0.0,
            "average_gas_efficiency": 0.0
        }
    
    async def execute_arbitrage_path(
        self,
        edges: List[YieldGraphEdge],
        input_amount: Decimal,
        execution_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a complete arbitrage path through the router.
        
        Args:
            edges: List of yield graph edges forming the arbitrage path
            input_amount: Initial input amount for the arbitrage
            execution_id: Optional execution ID for tracking
            
        Returns:
            Complete execution result with all metrics
        """
        if execution_id is None:
            execution_id = f"exec_{int(asyncio.get_event_loop().time() * 1000)}"
        
        logger.info(f"Starting arbitrage execution {execution_id} with {len(edges)} edges")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Analyze path and create segments
            segments = await self._analyze_execution_path(edges)
            logger.info(f"Path analyzed into {len(segments)} segments")
            
            # Step 2: Build execution plan
            execution_plan = await self._build_execution_plan(segments)
            logger.info(f"Execution plan created with {execution_plan.total_transactions} transactions")
            
            # Step 3: Validate execution (if enabled)
            validation_report = None
            if self.config.enable_pre_execution_validation and self.pre_execution_validator:
                validation_report = await self._validate_execution_plan(edges, input_amount, execution_plan)
                logger.info(f"Validation result: {validation_report.validation_result.value}")
                
                # Check if validation passes
                if not self._is_validation_acceptable(validation_report):
                    return self._create_failed_result(
                        execution_id, execution_plan, validation_report,
                        "Execution failed validation checks"
                    )
            
            # Step 4: Execute (or simulate if dry run)
            if self.config.dry_run_mode:
                result = await self._simulate_execution(execution_id, execution_plan, validation_report)
            else:
                result = await self._execute_plan(execution_id, execution_plan, validation_report)
            
            # Step 5: Calculate final metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            result.execution_time_seconds = execution_time
            
            # Update performance tracking
            self._update_performance_metrics(result)
            
            # Store in history
            self.execution_history[execution_id] = result
            
            logger.info(
                f"Execution {execution_id} completed: "
                f"success={result.success}, profit=${result.profit_loss_usd:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Execution {execution_id} failed with error: {e}")
            
            # Create empty execution plan for error case
            empty_plan = BatchExecutionPlan(
                plan_id=execution_id,
                router_address=self.config.router_contract_address,
                executor_address=self.config.executor_address,
                segments=[]
            )
            
            return self._create_failed_result(
                execution_id, empty_plan, None, f"Execution error: {str(e)}"
            )
    
    async def simulate_arbitrage_path(
        self,
        edges: List[YieldGraphEdge],
        input_amount: Decimal
    ) -> Dict[str, Any]:
        """
        Simulate arbitrage path without execution.
        
        Args:
            edges: List of yield graph edges
            input_amount: Initial input amount
            
        Returns:
            Simulation results and analysis
        """
        logger.info(f"Simulating arbitrage path with {len(edges)} edges")
        
        # Temporarily enable dry run mode
        original_dry_run = self.config.dry_run_mode
        self.config.dry_run_mode = True
        
        try:
            result = await self.execute_arbitrage_path(edges, input_amount, "simulation")
            
            return {
                "simulation_success": result.success,
                "estimated_profit_usd": result.profit_loss_usd,
                "estimated_gas_cost_usd": result.actual_gas_cost_usd,
                "validation_result": result.validation_report.validation_result.value if result.validation_report else None,
                "execution_plan": {
                    "total_transactions": result.execution_plan.total_transactions,
                    "total_gas_estimate": result.execution_plan.total_gas_estimate,
                    "flash_loan_required": any(tx.requires_flash_loan for tx in result.execution_plan.segments)
                },
                "recommendations": self._generate_simulation_recommendations(result)
            }
            
        finally:
            # Restore original dry run setting
            self.config.dry_run_mode = original_dry_run
    
    async def _analyze_execution_path(self, edges: List[YieldGraphEdge]) -> List[PathSegment]:
        """Analyze arbitrage path and create executable segments."""
        # Use path analyzer to break down the arbitrage path
        segments = self.path_analyzer.analyze_path(edges)
        
        # Apply strategy-specific optimizations
        if self.config.strategy == ExecutionStrategy.GAS_OPTIMAL:
            segments = self._optimize_for_gas(segments)
        elif self.config.strategy == ExecutionStrategy.AGGRESSIVE:
            segments = self._optimize_for_profit(segments)
        elif self.config.strategy == ExecutionStrategy.CONSERVATIVE:
            segments = self._optimize_for_safety(segments)
        
        return segments
    
    async def _build_execution_plan(self, segments: List[PathSegment]) -> BatchExecutionPlan:
        """Build complete execution plan from segments."""
        recipient = self.config.recipient_address or self.config.executor_address
        
        # Choose integration mode based on strategy
        integration_mode = self.config.integration_mode
        if self.config.strategy == ExecutionStrategy.GAS_OPTIMAL:
            integration_mode = RouterIntegrationMode.BATCH
        elif self.config.strategy == ExecutionStrategy.AGGRESSIVE:
            integration_mode = RouterIntegrationMode.FLASH_LOAN
        
        # Build execution plan
        execution_plan = self.transaction_builder.build_batch_execution(
            segments,
            self.config.executor_address,
            integration_mode,
            recipient
        )
        
        # Optimize if needed
        if self.config.strategy in [ExecutionStrategy.GAS_OPTIMAL, ExecutionStrategy.BALANCED]:
            execution_plan = self.transaction_builder.optimize_execution_plan(execution_plan)
        
        return execution_plan
    
    async def _validate_execution_plan(
        self,
        edges: List[YieldGraphEdge],
        input_amount: Decimal,
        execution_plan: BatchExecutionPlan
    ) -> ExecutionValidationReport:
        """Validate execution plan using pre-execution validator."""
        if not self.pre_execution_validator:
            raise RuntimeError("Pre-execution validator not available")
        
        # Create simulation parameters
        simulation_params = RouterSimulationParams(
            router_contract_address=self.config.router_contract_address,
            executor_address=self.config.executor_address,
            gas_price_gwei=min(self.config.max_gas_price_gwei, 50.0)  # Use reasonable default
        )
        
        # Run validation
        validation_report = await self.pre_execution_validator.validate_execution_plan(
            edges, input_amount, simulation_params
        )
        
        return validation_report
    
    def _is_validation_acceptable(self, validation_report: ExecutionValidationReport) -> bool:
        """Check if validation results are acceptable for execution."""
        if validation_report.validation_result.value == "invalid":
            return False
        
        # Check profit threshold
        if validation_report.estimated_profit_usd < self.config.min_profit_threshold_usd:
            return False
        
        # Check gas costs
        if validation_report.gas_cost_at_20_gwei > validation_report.estimated_profit_usd * 0.5:
            # Gas costs too high relative to profit
            return False
        
        # Strategy-specific checks
        if self.config.strategy == ExecutionStrategy.CONSERVATIVE:
            # Require higher success rate for conservative strategy
            if validation_report.simulation_success_rate < 95.0:
                return False
        
        return True
    
    async def _simulate_execution(
        self,
        execution_id: str,
        execution_plan: BatchExecutionPlan,
        validation_report: Optional[ExecutionValidationReport]
    ) -> ExecutionResult:
        """Simulate execution without actual execution."""
        logger.info(f"Simulating execution plan {execution_id}")
        
        # Use validation report results if available
        if validation_report:
            profit_usd = validation_report.estimated_profit_usd
            gas_cost_usd = validation_report.gas_cost_at_20_gwei
            gas_used = validation_report.estimated_gas_usage
        else:
            # Estimate from execution plan
            profit_usd = float(execution_plan.expected_profit)
            gas_cost_usd = self._estimate_gas_cost(execution_plan.total_gas_estimate)
            gas_used = execution_plan.total_gas_estimate
        
        return ExecutionResult(
            execution_id=execution_id,
            success=True,
            execution_plan=execution_plan,
            validation_report=validation_report,
            total_gas_used=gas_used,
            actual_gas_cost_usd=gas_cost_usd,
            profit_loss_usd=profit_usd - gas_cost_usd,
            profit_margin_percentage=(profit_usd - gas_cost_usd) / profit_usd * 100 if profit_usd > 0 else 0,
            simulation_accuracy=100.0,  # Perfect accuracy in simulation
            gas_efficiency=100.0
        )
    
    async def _execute_plan(
        self,
        execution_id: str,
        execution_plan: BatchExecutionPlan,
        validation_report: Optional[ExecutionValidationReport]
    ) -> ExecutionResult:
        """Execute the actual execution plan (placeholder - would integrate with blockchain)."""
        logger.info(f"Executing plan {execution_id} - THIS IS A PLACEHOLDER")
        
        # In a real implementation, this would:
        # 1. Submit approval transactions first
        # 2. Execute router transactions in sequence
        # 3. Monitor execution and collect actual results
        # 4. Calculate actual profit/loss
        
        # For now, return simulated result
        return await self._simulate_execution(execution_id, execution_plan, validation_report)
    
    def _create_failed_result(
        self,
        execution_id: str,
        execution_plan: BatchExecutionPlan,
        validation_report: Optional[ExecutionValidationReport],
        error_message: str
    ) -> ExecutionResult:
        """Create a failed execution result."""
        return ExecutionResult(
            execution_id=execution_id,
            success=False,
            execution_plan=execution_plan,
            validation_report=validation_report,
            error_message=error_message,
            profit_loss_usd=0.0
        )
    
    def _optimize_for_gas(self, segments: List[PathSegment]) -> List[PathSegment]:
        """Optimize segments for minimum gas usage."""
        # Sort by gas efficiency, combine where possible
        return sorted(segments, key=lambda s: s.max_gas_estimate or 0)
    
    def _optimize_for_profit(self, segments: List[PathSegment]) -> List[PathSegment]:
        """Optimize segments for maximum profit."""
        # Prioritize high-profit operations, enable flash loans
        optimized = []
        for segment in segments:
            if not segment.requires_flash_loan and segment.can_use_flash_loan:
                # Enable flash loan for better capital efficiency
                segment.requires_flash_loan = True
                segment.flash_loan_asset = "USDC"  # Default
                segment.flash_loan_amount = 100000.0  # Default amount
            optimized.append(segment)
        return optimized
    
    def _optimize_for_safety(self, segments: List[PathSegment]) -> List[PathSegment]:
        """Optimize segments for maximum safety."""
        # Prefer simple operations, avoid complex flash loans
        safe_segments = []
        for segment in segments:
            if segment.edge_count <= 3:  # Limit complexity
                safe_segments.append(segment)
        return safe_segments
    
    def _estimate_gas_cost(self, gas_amount: int, gas_price_gwei: float = 20.0) -> float:
        """Estimate gas cost in USD."""
        eth_price_usd = 2000.0  # Placeholder - would use real price feed
        gas_cost_eth = (gas_amount * gas_price_gwei * 1e9) / 1e18
        return gas_cost_eth * eth_price_usd
    
    def _generate_simulation_recommendations(self, result: ExecutionResult) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        if result.profit_loss_usd < self.config.min_profit_threshold_usd:
            recommendations.append("Consider increasing input amount or finding better routes")
        
        if result.execution_plan.total_gas_estimate > 6_000_000:
            recommendations.append("High gas usage - consider splitting into smaller transactions")
        
        if result.actual_gas_cost_usd > result.profit_loss_usd * 0.3:
            recommendations.append("Gas costs are high relative to profit - wait for lower gas prices")
        
        flash_loan_count = len([tx for tx in result.execution_plan.segments if tx.requires_flash_loan])
        if flash_loan_count > 2:
            recommendations.append("Multiple flash loans required - verify all fees are accounted for")
        
        return recommendations
    
    def _update_performance_metrics(self, result: ExecutionResult):
        """Update internal performance tracking."""
        self.performance_metrics["total_executions"] += 1
        
        if result.success:
            self.performance_metrics["successful_executions"] += 1
            self.performance_metrics["total_profit_usd"] += result.profit_loss_usd
        
        # Update average gas efficiency
        if result.gas_efficiency > 0:
            current_avg = self.performance_metrics["average_gas_efficiency"]
            total_execs = self.performance_metrics["total_executions"]
            self.performance_metrics["average_gas_efficiency"] = (
                (current_avg * (total_execs - 1) + result.gas_efficiency) / total_execs
            )
    
    def get_execution_history(self) -> Dict[str, ExecutionResult]:
        """Get complete execution history."""
        return self.execution_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and metrics."""
        metrics = self.performance_metrics.copy()
        
        # Calculate additional metrics
        if metrics["total_executions"] > 0:
            metrics["success_rate"] = (
                metrics["successful_executions"] / metrics["total_executions"] * 100
            )
            metrics["average_profit_per_execution"] = (
                metrics["total_profit_usd"] / metrics["total_executions"]
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["average_profit_per_execution"] = 0.0
        
        return metrics
    
    def clear_history(self):
        """Clear execution history and reset metrics."""
        self.execution_history.clear()
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_profit_usd": 0.0,
            "average_gas_efficiency": 0.0
        }


# Convenience functions

async def execute_simple_arbitrage(
    edges: List[YieldGraphEdge],
    input_amount: Decimal,
    router_address: str,
    executor_address: str,
    tenderly_client=None
) -> ExecutionResult:
    """
    Execute a simple arbitrage path with default settings.
    
    Args:
        edges: Arbitrage path edges
        input_amount: Input amount
        router_address: Router contract address
        executor_address: Executor address
        tenderly_client: Optional Tenderly client
        
    Returns:
        Execution result
    """
    config = RouterExecutionConfig(
        router_contract_address=router_address,
        executor_address=executor_address,
        strategy=ExecutionStrategy.BALANCED
    )
    
    engine = RouterExecutionEngine(config, tenderly_client)
    return await engine.execute_arbitrage_path(edges, input_amount)


async def simulate_arbitrage_profitability(
    edges: List[YieldGraphEdge],
    input_amount: Decimal,
    router_address: str,
    executor_address: str,
    tenderly_client=None
) -> Dict[str, Any]:
    """
    Simulate arbitrage profitability without execution.
    
    Args:
        edges: Arbitrage path edges
        input_amount: Input amount
        router_address: Router contract address
        executor_address: Executor address
        tenderly_client: Optional Tenderly client
        
    Returns:
        Profitability analysis
    """
    config = RouterExecutionConfig(
        router_contract_address=router_address,
        executor_address=executor_address,
        dry_run_mode=True
    )
    
    engine = RouterExecutionEngine(config, tenderly_client)
    return await engine.simulate_arbitrage_path(edges, input_amount)