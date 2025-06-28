"""
Logged Execution Engine with PostgreSQL Integration.

This module extends the ExecutionEngine with comprehensive PostgreSQL logging
for all execution attempts, simulation results, and performance metrics.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

from .execution_engine import (
    ExecutionEngine, ExecutionContext, ExecutionResult, ExecutionStatus, PreFlightCheck
)
from ..database.execution_logger import get_execution_logger, ExecutionLogger
from ..pathfinding.path_models import YieldPath
from ..execution.hybrid_simulator import SimulationResult

logger = logging.getLogger(__name__)


class LoggedExecutionEngine(ExecutionEngine):
    """
    ExecutionEngine with comprehensive PostgreSQL logging integration.
    
    Extends the base ExecutionEngine to automatically log all execution attempts,
    simulation results, pre-flight checks, and performance metrics to PostgreSQL
    for analysis and monitoring.
    """
    
    def __init__(
        self,
        simulator,
        transaction_builder,
        mev_router,
        delta_tracker,
        mev_assessor,
        asset_oracle,
        router_address: str,
        chain_id: int = 1,
        enable_logging: bool = True
    ):
        """
        Initialize logged execution engine.
        
        Args:
            enable_logging: Whether to enable PostgreSQL logging
            All other arguments are passed to the base ExecutionEngine
        """
        super().__init__(
            simulator=simulator,
            transaction_builder=transaction_builder,
            mev_router=mev_router,
            delta_tracker=delta_tracker,
            mev_assessor=mev_assessor,
            asset_oracle=asset_oracle,
            router_address=router_address,
            chain_id=chain_id
        )
        
        self.enable_logging = enable_logging
        self.execution_logger: Optional[ExecutionLogger] = None
        
        if self.enable_logging:
            self.execution_logger = get_execution_logger()
        
        # Additional logging stats
        self.logging_stats = {
            "logs_written": 0,
            "log_failures": 0,
            "last_log_time": None
        }
        
        logger.info(
            f"LoggedExecutionEngine initialized with logging {'enabled' if enable_logging else 'disabled'}"
        )
    
    async def execute_path(
        self,
        path: YieldPath,
        initial_amount: float,
        execution_context: Optional[ExecutionContext] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        request_source: Optional[str] = "api"
    ) -> ExecutionResult:
        """
        Execute a yield arbitrage path with comprehensive logging.
        
        Args:
            path: Complete yield path to execute
            initial_amount: Starting amount in path's start asset
            execution_context: Optional execution context with parameters
            session_id: Optional session identifier for grouping
            user_id: Optional user identifier
            api_key_id: Optional API key identifier
            request_source: Source of the request (api, ui, automated)
            
        Returns:
            ExecutionResult with complete execution details
        """
        execution_id = f"exec_{int(time.time() * 1000)}_{hash(str(path.edges)) % 10000}"
        
        # Create execution context if not provided
        if execution_context is None:
            execution_context = ExecutionContext(
                execution_id=execution_id,
                path=path,
                initial_amount=initial_amount,
                start_asset_id=path.edges[0].source_asset_id,
                chain_id=self.chain_id
            )
        else:
            execution_context.execution_id = execution_id
        
        # Log execution start
        if self.enable_logging and self.execution_logger:
            log_success = await self.execution_logger.log_execution_start(
                context=execution_context,
                session_id=session_id,
                user_id=user_id,
                api_key_id=api_key_id,
                request_source=request_source
            )
            
            if log_success:
                self.logging_stats["logs_written"] += 1
                self.logging_stats["last_log_time"] = time.time()
            else:
                self.logging_stats["log_failures"] += 1
        
        # Track active execution
        self.active_executions[execution_id] = execution_context
        self.stats["executions_attempted"] += 1
        
        logger.info(
            f"Starting logged execution {execution_id}: "
            f"{len(path.edges)} edges, ${initial_amount:.2f} start amount"
        )
        
        try:
            # Phase 1: Pre-flight checks with timing
            execution_context.status = ExecutionStatus.PRE_FLIGHT_CHECK
            await self._log_status_update(execution_id, ExecutionStatus.PRE_FLIGHT_CHECK)
            
            pre_flight_start = time.time()
            pre_flight_result = await self._run_pre_flight_checks(execution_context)
            pre_flight_time_ms = int((time.time() - pre_flight_start) * 1000)
            
            # Log pre-flight results
            if self.enable_logging and self.execution_logger:
                await self.execution_logger.log_pre_flight_results(
                    execution_id=execution_id,
                    pre_flight_checks=pre_flight_result,
                    pre_flight_time_ms=pre_flight_time_ms
                )
                self.logging_stats["logs_written"] += 1
            
            if any(check.result.value == "fail" and check.is_blocking 
                   for check in pre_flight_result):
                logger.warning(f"Pre-flight checks failed for {execution_id}")
                result = self._create_failure_result(
                    execution_context,
                    "Pre-flight checks failed",
                    ExecutionStatus.FAILED,
                    pre_flight_checks=pre_flight_result
                )
                await self._log_execution_completion(result)
                return result
            
            # Phase 2: Path simulation with market context
            execution_context.status = ExecutionStatus.SIMULATING
            await self._log_status_update(execution_id, ExecutionStatus.SIMULATING)
            
            simulation_result = await self._simulate_path(execution_context)
            
            # Gather market context for logging
            market_context = await self._gather_market_context()
            
            # Log simulation results
            if self.enable_logging and self.execution_logger:
                await self.execution_logger.log_simulation_results(
                    execution_id=execution_id,
                    simulation_result=simulation_result,
                    market_context=market_context
                )
                self.logging_stats["logs_written"] += 1
            
            if not simulation_result.success:
                logger.warning(f"Simulation failed for {execution_id}: {simulation_result.revert_reason}")
                result = self._create_failure_result(
                    execution_context,
                    f"Simulation failed: {simulation_result.revert_reason}",
                    ExecutionStatus.FAILED,
                    simulation_result=simulation_result,
                    pre_flight_checks=pre_flight_result
                )
                await self._log_execution_completion(result)
                return result
            
            # Phase 3: Build execution plan
            execution_context.status = ExecutionStatus.BUILDING_TRANSACTIONS
            await self._log_status_update(execution_id, ExecutionStatus.BUILDING_TRANSACTIONS)
            
            execution_plan = await self._build_execution_plan(execution_context, simulation_result)
            
            if not execution_plan:
                result = self._create_failure_result(
                    execution_context,
                    "Failed to build execution plan",
                    ExecutionStatus.FAILED,
                    simulation_result=simulation_result,
                    pre_flight_checks=pre_flight_result
                )
                await self._log_execution_completion(result)
                return result
            
            # Phase 4: MEV risk assessment and routing
            execution_context.status = ExecutionStatus.ROUTING
            await self._log_status_update(execution_id, ExecutionStatus.ROUTING)
            
            execution_route = await self._select_execution_route(execution_context, execution_plan)
            
            # Phase 5: Position tracking setup
            position_id = None
            if self.enable_position_tracking:
                position_id = await self._create_position_tracking(execution_context, simulation_result)
            
            # Phase 6: Execute transactions
            execution_context.status = ExecutionStatus.EXECUTING
            await self._log_status_update(execution_id, ExecutionStatus.EXECUTING)
            
            execution_success, tx_hashes, error_msg = await self._execute_transactions(
                execution_plan, execution_route
            )
            
            if not execution_success:
                result = self._create_failure_result(
                    execution_context,
                    f"Execution failed: {error_msg}",
                    ExecutionStatus.FAILED,
                    simulation_result=simulation_result,
                    execution_plan=execution_plan,
                    execution_route=execution_route,
                    pre_flight_checks=pre_flight_result,
                    position_id=position_id
                )
                await self._log_execution_completion(result)
                return result
            
            # Phase 7: Confirmation and finalization
            execution_context.status = ExecutionStatus.CONFIRMING
            await self._log_status_update(execution_id, ExecutionStatus.CONFIRMING)
            
            await self._confirm_execution(execution_context, tx_hashes)
            
            execution_context.status = ExecutionStatus.COMPLETED
            
            # Create success result
            result = ExecutionResult(
                execution_id=execution_id,
                success=True,
                status=ExecutionStatus.COMPLETED,
                simulation_result=simulation_result,
                execution_plan=execution_plan,
                execution_route=execution_route,
                transaction_hashes=tx_hashes,
                position_created=position_id,
                pre_flight_checks=pre_flight_result,
                execution_time_seconds=time.time() - execution_context.created_at
            )
            
            # Log completion
            await self._log_execution_completion(result)
            
            # Update statistics
            self.stats["executions_successful"] += 1
            if simulation_result.profit_usd:
                self.stats["total_profit_realized_usd"] += simulation_result.profit_usd
            if simulation_result.gas_cost_usd:
                self.stats["total_gas_spent_usd"] += simulation_result.gas_cost_usd
            
            self.completed_executions[execution_id] = result
            logger.info(f"Logged execution {execution_id} completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Execution {execution_id} failed with error: {e}", exc_info=True)
            result = self._create_failure_result(
                execution_context,
                f"Unexpected error: {str(e)}",
                ExecutionStatus.FAILED
            )
            await self._log_execution_completion(result)
            return result
        
        finally:
            # Clean up active execution tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _log_status_update(
        self,
        execution_id: str,
        status: ExecutionStatus,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log execution status update."""
        if self.enable_logging and self.execution_logger:
            await self.execution_logger.log_execution_update(
                execution_id=execution_id,
                status=status,
                additional_data=additional_data
            )
            self.logging_stats["logs_written"] += 1
    
    async def _log_execution_completion(self, result: ExecutionResult) -> None:
        """Log execution completion with delta exposure."""
        if not self.enable_logging or not self.execution_logger:
            return
        
        try:
            # Get delta exposure if position was created
            delta_exposure = None
            if result.position_created:
                position = self.delta_tracker.get_position(result.position_created)
                if position:
                    delta_exposure = {
                        asset_id: float(exposure.amount)
                        for asset_id, exposure in position.exposures.items()
                    }
            
            await self.execution_logger.log_execution_completion(
                execution_result=result,
                delta_exposure=delta_exposure
            )
            self.logging_stats["logs_written"] += 1
            
        except Exception as e:
            logger.error(f"Failed to log execution completion: {e}")
            self.logging_stats["log_failures"] += 1
    
    async def _gather_market_context(self) -> Dict[str, Any]:
        """Gather market context for logging."""
        context = {}
        
        try:
            # Get ETH price
            eth_price = await self.asset_oracle.get_price_usd("ETH_MAINNET_WETH")
            if eth_price:
                context["eth_price_usd"] = eth_price
            
            # Mock gas price (in production, would get from gas oracle)
            context["gas_price_gwei"] = 20.0
            
            # Mock block number (in production, would get from blockchain)
            context["block_number"] = 18500000
            
        except Exception as e:
            logger.warning(f"Failed to gather market context: {e}")
        
        return context
    
    async def get_execution_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get execution analytics from the database.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with analytics data
        """
        if not self.enable_logging or not self.execution_logger:
            return {"error": "Logging not enabled"}
        
        return await self.execution_logger.get_execution_analytics(hours)
    
    async def get_recent_executions(
        self,
        limit: int = 100,
        status_filter: Optional[str] = None,
        success_filter: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent execution records from the database.
        
        Args:
            limit: Maximum number of records to return
            status_filter: Optional status filter
            success_filter: Optional success filter
            
        Returns:
            List of execution record dictionaries
        """
        if not self.enable_logging or not self.execution_logger:
            return []
        
        try:
            records = await self.execution_logger.get_recent_executions(
                limit=limit,
                status_filter=status_filter,
                success_filter=success_filter
            )
            
            # Convert to dictionaries for serialization
            return [
                {
                    "execution_id": record.execution_id,
                    "path_id": record.path_id,
                    "status": record.status,
                    "success": record.success,
                    "chain_name": record.chain_name,
                    "initial_amount": float(record.initial_amount),
                    "predicted_profit_usd": float(record.predicted_profit_usd) if record.predicted_profit_usd else None,
                    "simulation_mode": record.simulation_mode,
                    "execution_method": record.execution_method,
                    "started_at": record.started_at.isoformat(),
                    "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                    "execution_time_seconds": float(record.execution_time_seconds) if record.execution_time_seconds else None,
                    "protocols": record.protocols,
                    "mev_protected": record.mev_protected,
                    "error_message": record.error_message
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Failed to get recent executions: {e}")
            return []
    
    def get_logging_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        base_stats = self.get_stats()
        return {
            **base_stats,
            "logging_enabled": self.enable_logging,
            **self.logging_stats,
            "last_log_time_iso": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.logging_stats["last_log_time"]))
                if self.logging_stats["last_log_time"] else None
            )
        }
    
    async def enable_database_logging(self) -> bool:
        """Enable database logging if it was disabled."""
        if not self.enable_logging:
            try:
                self.execution_logger = get_execution_logger()
                self.enable_logging = True
                logger.info("Database logging enabled")
                return True
            except Exception as e:
                logger.error(f"Failed to enable database logging: {e}")
                return False
        return True
    
    def disable_database_logging(self) -> None:
        """Disable database logging."""
        self.enable_logging = False
        self.execution_logger = None
        logger.info("Database logging disabled")


# Factory function for creating logged execution engine
def create_logged_execution_engine(
    simulator,
    transaction_builder,
    mev_router,
    delta_tracker,
    mev_assessor,
    asset_oracle,
    router_address: str,
    chain_id: int = 1,
    enable_logging: bool = True
) -> LoggedExecutionEngine:
    """
    Factory function to create a LoggedExecutionEngine.
    
    Args:
        All arguments are passed to LoggedExecutionEngine constructor
        
    Returns:
        Configured LoggedExecutionEngine instance
    """
    return LoggedExecutionEngine(
        simulator=simulator,
        transaction_builder=transaction_builder,
        mev_router=mev_router,
        delta_tracker=delta_tracker,
        mev_assessor=mev_assessor,
        asset_oracle=asset_oracle,
        router_address=router_address,
        chain_id=chain_id,
        enable_logging=enable_logging
    )