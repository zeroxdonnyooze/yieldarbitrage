"""
Execution Logger for PostgreSQL Integration.

This module provides comprehensive logging of simulated and actual execution attempts
to PostgreSQL for analysis, monitoring, and performance optimization.
"""
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from decimal import Decimal

from sqlalchemy import select, update, desc
from sqlalchemy.exc import IntegrityError

from .connection import get_session
from .models import SimulatedExecution
from ..execution.execution_engine import (
    ExecutionContext, ExecutionResult, ExecutionStatus, PreFlightCheck, PreFlightCheckResult
)
from ..execution.hybrid_simulator import SimulationResult

logger = logging.getLogger(__name__)


class ExecutionLogger:
    """
    Comprehensive execution logger that tracks all execution attempts and results.
    
    Integrates with ExecutionEngine to provide detailed logging and analytics
    for simulated execution attempts, actual executions, and performance metrics.
    """
    
    def __init__(self):
        """Initialize execution logger."""
        self.stats = {
            "records_created": 0,
            "records_updated": 0,
            "write_errors": 0,
            "last_write_time": None
        }
        
        logger.info("ExecutionLogger initialized")
    
    async def log_execution_start(
        self,
        context: ExecutionContext,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        request_source: Optional[str] = None
    ) -> bool:
        """
        Log the start of an execution attempt.
        
        Args:
            context: Execution context with all parameters
            session_id: Optional session identifier
            user_id: Optional user identifier
            api_key_id: Optional API key identifier
            request_source: Source of the request (api, ui, automated)
            
        Returns:
            True if logged successfully
        """
        try:
            async with get_session() as session:
                # Calculate path hash for deduplication
                path_hash = self._calculate_path_hash(context.path.edges)
                
                # Create new simulated execution record
                execution_record = SimulatedExecution(
                    execution_id=context.execution_id,
                    session_id=session_id,
                    path_id=context.path.path_id,
                    path_hash=path_hash,
                    chain_id=context.chain_id,
                    chain_name=self._get_chain_name(context.chain_id),
                    initial_amount=Decimal(str(context.initial_amount)),
                    start_asset_id=context.start_asset_id,
                    edge_ids=[edge.edge_id for edge in context.path.edges],
                    edge_types=[edge.edge_type.value for edge in context.path.edges],
                    protocols=[edge.protocol_name for edge in context.path.edges if edge.protocol_name],
                    status=context.status.value,
                    success=False,  # Will be updated later
                    max_slippage=Decimal(str(context.max_slippage)),
                    mev_protected=context.use_mev_protection,
                    user_id=user_id,
                    api_key_id=api_key_id,
                    request_source=request_source,
                    started_at=datetime.fromtimestamp(context.created_at, tz=timezone.utc)
                )
                
                session.add(execution_record)
                await session.commit()
                
                self.stats["records_created"] += 1
                self.stats["last_write_time"] = datetime.now(timezone.utc)
                
                logger.debug(f"Logged execution start for {context.execution_id}")
                return True
                
        except IntegrityError as e:
            logger.warning(f"Execution {context.execution_id} already logged: {e}")
            return True  # Already exists, that's fine
            
        except Exception as e:
            logger.error(f"Failed to log execution start for {context.execution_id}: {e}")
            self.stats["write_errors"] += 1
            return False
    
    async def log_pre_flight_results(
        self,
        execution_id: str,
        pre_flight_checks: List[PreFlightCheck],
        pre_flight_time_ms: Optional[int] = None
    ) -> bool:
        """
        Log pre-flight check results.
        
        Args:
            execution_id: Execution identifier
            pre_flight_checks: List of pre-flight check results
            pre_flight_time_ms: Time spent on pre-flight checks
            
        Returns:
            True if logged successfully
        """
        try:
            async with get_session() as session:
                # Count check results
                warnings = sum(1 for check in pre_flight_checks 
                              if check.result == PreFlightCheckResult.WARNING)
                failures = sum(1 for check in pre_flight_checks 
                              if check.result == PreFlightCheckResult.FAIL)
                passed = warnings == 0 and failures == 0
                
                # Serialize check details
                check_details = []
                for check in pre_flight_checks:
                    check_details.append({
                        "check_name": check.check_name,
                        "result": check.result.value,
                        "message": check.message,
                        "is_blocking": check.is_blocking,
                        "details": check.details
                    })
                
                # Update execution record
                stmt = (
                    update(SimulatedExecution)
                    .where(SimulatedExecution.execution_id == execution_id)
                    .values(
                        pre_flight_passed=passed,
                        pre_flight_warnings=warnings,
                        pre_flight_failures=failures,
                        pre_flight_details=check_details,
                        pre_flight_time_ms=pre_flight_time_ms
                    )
                )
                
                await session.execute(stmt)
                await session.commit()
                
                self.stats["records_updated"] += 1
                logger.debug(f"Logged pre-flight results for {execution_id}: {warnings}W, {failures}F")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log pre-flight results for {execution_id}: {e}")
            self.stats["write_errors"] += 1
            return False
    
    async def log_simulation_results(
        self,
        execution_id: str,
        simulation_result: SimulationResult,
        market_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log simulation results.
        
        Args:
            execution_id: Execution identifier
            simulation_result: Simulation result from HybridPathSimulator
            market_context: Optional market context (ETH price, gas price, etc.)
            
        Returns:
            True if logged successfully
        """
        try:
            async with get_session() as session:
                update_data = {
                    "simulation_mode": simulation_result.simulation_mode,
                    "simulation_success": simulation_result.success,
                    "simulation_time_ms": int(simulation_result.simulation_time_ms or 0),
                    "predicted_profit_usd": (
                        Decimal(str(simulation_result.profit_usd)) 
                        if simulation_result.profit_usd is not None else None
                    ),
                    "predicted_profit_percentage": (
                        Decimal(str(simulation_result.profit_percentage)) 
                        if simulation_result.profit_percentage is not None else None
                    ),
                    "estimated_gas_cost_usd": (
                        Decimal(str(simulation_result.gas_cost_usd))
                        if simulation_result.gas_cost_usd is not None else None
                    ),
                    "estimated_output_amount": (
                        Decimal(str(simulation_result.output_amount))
                        if simulation_result.output_amount is not None else None
                    ),
                    "revert_reason": simulation_result.revert_reason,
                    "warnings": simulation_result.warnings
                }
                
                # Add market context if provided
                if market_context:
                    if "eth_price_usd" in market_context:
                        update_data["eth_price_usd"] = Decimal(str(market_context["eth_price_usd"]))
                    if "gas_price_gwei" in market_context:
                        update_data["gas_price_gwei"] = Decimal(str(market_context["gas_price_gwei"]))
                    if "block_number" in market_context:
                        update_data["block_number"] = market_context["block_number"]
                
                # Update execution record
                stmt = (
                    update(SimulatedExecution)
                    .where(SimulatedExecution.execution_id == execution_id)
                    .values(**update_data)
                )
                
                await session.execute(stmt)
                await session.commit()
                
                self.stats["records_updated"] += 1
                logger.debug(
                    f"Logged simulation results for {execution_id}: "
                    f"{'SUCCESS' if simulation_result.success else 'FAILED'}, "
                    f"profit=${simulation_result.profit_usd or 0:.2f}"
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to log simulation results for {execution_id}: {e}")
            self.stats["write_errors"] += 1
            return False
    
    async def log_execution_completion(
        self,
        execution_result: ExecutionResult,
        delta_exposure: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Log execution completion with final results.
        
        Args:
            execution_result: Complete execution result
            delta_exposure: Optional market delta exposure
            
        Returns:
            True if logged successfully
        """
        try:
            async with get_session() as session:
                update_data = {
                    "status": execution_result.status.value,
                    "success": execution_result.success,
                    "completed_at": datetime.now(timezone.utc),
                    "execution_time_seconds": (
                        Decimal(str(execution_result.execution_time_seconds))
                        if execution_result.execution_time_seconds is not None else None
                    ),
                    "error_message": execution_result.error_message,
                    "failed_at_step": execution_result.failed_at_step,
                    "position_id": execution_result.position_created,
                    "warnings": execution_result.warnings
                }
                
                # Add MEV and routing information
                if execution_result.execution_route:
                    update_data["execution_method"] = execution_result.execution_route.method.value
                    update_data["mev_protected"] = execution_result.execution_route.method.value != "public"
                
                # Add transaction details if available
                if execution_result.transaction_hashes:
                    update_data["transaction_hashes"] = execution_result.transaction_hashes
                
                if execution_result.gas_used:
                    update_data["actual_gas_used"] = execution_result.gas_used
                
                if execution_result.gas_cost_usd:
                    update_data["actual_gas_cost_usd"] = Decimal(str(execution_result.gas_cost_usd))
                
                # Add delta exposure if provided
                if delta_exposure:
                    update_data["delta_exposure"] = delta_exposure
                
                # Update execution record
                stmt = (
                    update(SimulatedExecution)
                    .where(SimulatedExecution.execution_id == execution_result.execution_id)
                    .values(**update_data)
                )
                
                await session.execute(stmt)
                await session.commit()
                
                self.stats["records_updated"] += 1
                logger.info(
                    f"Logged execution completion for {execution_result.execution_id}: "
                    f"{'SUCCESS' if execution_result.success else 'FAILED'}"
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to log execution completion for {execution_result.execution_id}: {e}")
            self.stats["write_errors"] += 1
            return False
    
    async def log_execution_update(
        self,
        execution_id: str,
        status: ExecutionStatus,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log execution status update.
        
        Args:
            execution_id: Execution identifier
            status: New execution status
            additional_data: Optional additional data to update
            
        Returns:
            True if logged successfully
        """
        try:
            async with get_session() as session:
                update_data = {"status": status.value}
                
                if additional_data:
                    update_data.update(additional_data)
                
                stmt = (
                    update(SimulatedExecution)
                    .where(SimulatedExecution.execution_id == execution_id)
                    .values(**update_data)
                )
                
                await session.execute(stmt)
                await session.commit()
                
                self.stats["records_updated"] += 1
                logger.debug(f"Updated execution {execution_id} status to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update execution {execution_id}: {e}")
            self.stats["write_errors"] += 1
            return False
    
    async def get_execution_record(self, execution_id: str) -> Optional[SimulatedExecution]:
        """
        Get execution record by ID.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            SimulatedExecution record or None if not found
        """
        try:
            async with get_session() as session:
                stmt = select(SimulatedExecution).where(
                    SimulatedExecution.execution_id == execution_id
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
                
        except Exception as e:
            logger.error(f"Failed to get execution record {execution_id}: {e}")
            return None
    
    async def get_recent_executions(
        self,
        limit: int = 100,
        status_filter: Optional[str] = None,
        success_filter: Optional[bool] = None
    ) -> List[SimulatedExecution]:
        """
        Get recent execution records.
        
        Args:
            limit: Maximum number of records to return
            status_filter: Optional status filter
            success_filter: Optional success filter
            
        Returns:
            List of SimulatedExecution records
        """
        try:
            async with get_session() as session:
                stmt = select(SimulatedExecution).order_by(
                    desc(SimulatedExecution.started_at)
                ).limit(limit)
                
                if status_filter:
                    stmt = stmt.where(SimulatedExecution.status == status_filter)
                
                if success_filter is not None:
                    stmt = stmt.where(SimulatedExecution.success == success_filter)
                
                result = await session.execute(stmt)
                return result.scalars().all()
                
        except Exception as e:
            logger.error(f"Failed to get recent executions: {e}")
            return []
    
    async def get_execution_analytics(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get execution analytics for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            async with get_session() as session:
                # Get executions from the last N hours
                cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
                cutoff_datetime = datetime.fromtimestamp(cutoff_time, tz=timezone.utc)
                
                stmt = select(SimulatedExecution).where(
                    SimulatedExecution.started_at >= cutoff_datetime
                )
                
                result = await session.execute(stmt)
                executions = result.scalars().all()
                
                # Calculate analytics
                total_executions = len(executions)
                successful_executions = sum(1 for ex in executions if ex.success)
                failed_executions = total_executions - successful_executions
                
                # Profitability analysis
                profitable_executions = [
                    ex for ex in executions 
                    if ex.predicted_profit_usd and ex.predicted_profit_usd > 0
                ]
                
                total_predicted_profit = sum(
                    float(ex.predicted_profit_usd) for ex in profitable_executions
                    if ex.predicted_profit_usd
                )
                
                # Simulation analysis
                simulation_modes = {}
                avg_simulation_time = 0
                if executions:
                    for ex in executions:
                        if ex.simulation_mode:
                            simulation_modes[ex.simulation_mode] = simulation_modes.get(ex.simulation_mode, 0) + 1
                    
                    sim_times = [ex.simulation_time_ms for ex in executions if ex.simulation_time_ms]
                    if sim_times:
                        avg_simulation_time = sum(sim_times) / len(sim_times)
                
                # Pre-flight analysis
                pre_flight_failures = sum(1 for ex in executions if not ex.pre_flight_passed)
                avg_warnings = sum(ex.pre_flight_warnings for ex in executions) / max(total_executions, 1)
                
                return {
                    "time_period_hours": hours,
                    "total_executions": total_executions,
                    "successful_executions": successful_executions,
                    "failed_executions": failed_executions,
                    "success_rate": successful_executions / max(total_executions, 1),
                    "profitable_executions": len(profitable_executions),
                    "total_predicted_profit_usd": total_predicted_profit,
                    "avg_predicted_profit_usd": total_predicted_profit / max(len(profitable_executions), 1),
                    "simulation_modes": simulation_modes,
                    "avg_simulation_time_ms": avg_simulation_time,
                    "pre_flight_failures": pre_flight_failures,
                    "avg_pre_flight_warnings": avg_warnings,
                    "most_common_protocols": self._analyze_protocols(executions),
                    "chain_distribution": self._analyze_chains(executions)
                }
                
        except Exception as e:
            logger.error(f"Failed to get execution analytics: {e}")
            return {"error": str(e)}
    
    def _calculate_path_hash(self, edges) -> str:
        """Calculate SHA256 hash of path for deduplication."""
        path_string = "->".join([edge.edge_id for edge in edges])
        return hashlib.sha256(path_string.encode()).hexdigest()
    
    def _get_chain_name(self, chain_id: int) -> str:
        """Get chain name from chain ID."""
        chain_names = {
            1: "ethereum",
            56: "bsc",
            137: "polygon",
            42161: "arbitrum",
            10: "optimism",
            8453: "base"
        }
        return chain_names.get(chain_id, f"chain_{chain_id}")
    
    def _analyze_protocols(self, executions: List[SimulatedExecution]) -> Dict[str, int]:
        """Analyze protocol usage in executions."""
        protocol_counts = {}
        for execution in executions:
            for protocol in execution.protocols:
                if protocol:
                    protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        # Return top 10 protocols
        return dict(sorted(protocol_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_chains(self, executions: List[SimulatedExecution]) -> Dict[str, int]:
        """Analyze chain distribution in executions."""
        chain_counts = {}
        for execution in executions:
            chain = execution.chain_name
            chain_counts[chain] = chain_counts.get(chain, 0) + 1
        
        return chain_counts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            **self.stats,
            "last_write_time_iso": (
                self.stats["last_write_time"].isoformat() 
                if self.stats["last_write_time"] else None
            )
        }


# Global logger instance
_execution_logger: Optional[ExecutionLogger] = None


def get_execution_logger() -> ExecutionLogger:
    """Get or create global execution logger instance."""
    global _execution_logger
    if _execution_logger is None:
        _execution_logger = ExecutionLogger()
    return _execution_logger


# Convenience functions for direct integration

async def log_execution_start(
    context: ExecutionContext,
    **kwargs
) -> bool:
    """Convenience function to log execution start."""
    logger_instance = get_execution_logger()
    return await logger_instance.log_execution_start(context, **kwargs)


async def log_simulation_results(
    execution_id: str,
    simulation_result: SimulationResult,
    **kwargs
) -> bool:
    """Convenience function to log simulation results."""
    logger_instance = get_execution_logger()
    return await logger_instance.log_simulation_results(execution_id, simulation_result, **kwargs)


async def log_execution_completion(
    execution_result: ExecutionResult,
    **kwargs
) -> bool:
    """Convenience function to log execution completion."""
    logger_instance = get_execution_logger()
    return await logger_instance.log_execution_completion(execution_result, **kwargs)