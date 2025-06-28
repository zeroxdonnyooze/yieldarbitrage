"""Hybrid path simulator combining basic mathematical simulation with Tenderly validation."""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import aiohttp

from ..graph_engine.models import YieldGraphEdge, EdgeState, EdgeType

logger = logging.getLogger(__name__)


class SimulationMode(str, Enum):
    """Simulation mode selection."""
    BASIC = "basic"           # Fast mathematical simulation
    TENDERLY = "tenderly"     # On-chain simulation via Tenderly
    HYBRID = "hybrid"         # Basic first, then Tenderly validation
    LOCAL = "local"           # Local EVM simulation fallback


@dataclass
class SimulationResult:
    """Result of path simulation."""
    success: bool
    simulation_mode: str
    
    # Profitability metrics
    profit_usd: Optional[float] = None
    profit_amount_start_asset: Optional[float] = None
    profit_percentage: Optional[float] = None
    
    # Gas metrics
    gas_used: Optional[int] = None
    gas_cost_usd: Optional[float] = None
    gas_cost_start_asset: Optional[float] = None
    
    # Output metrics
    output_amount: Optional[float] = None
    final_amount_start_asset: Optional[float] = None
    
    # Failure information
    revert_reason: Optional[str] = None
    failed_at_step: Optional[int] = None
    
    # Market impact
    slippage_estimate: Optional[float] = None
    price_impact_percentage: Optional[float] = None
    
    # Warnings and metadata
    warnings: List[str] = field(default_factory=list)
    path_details: Optional[List[Dict[str, Any]]] = None
    tenderly_trace: Optional[Dict[str, Any]] = None
    tenderly_fork_id: Optional[str] = None
    
    # Timing
    simulation_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "simulation_mode": self.simulation_mode,
            "profit_usd": self.profit_usd,
            "profit_percentage": self.profit_percentage,
            "gas_cost_usd": self.gas_cost_usd,
            "output_amount": self.output_amount,
            "revert_reason": self.revert_reason,
            "slippage_estimate": self.slippage_estimate,
            "warnings": self.warnings,
            "simulation_time_ms": self.simulation_time_ms,
        }


@dataclass
class TenderlyConfig:
    """Configuration for Tenderly API access."""
    api_key: str
    project_slug: str
    username: str
    base_url: str = "https://api.tenderly.co/api/v1"
    timeout_seconds: float = 30.0
    max_retries: int = 3


@dataclass
class SimulatorConfig:
    """Configuration for the hybrid path simulator."""
    # Basic simulation settings
    default_slippage_factor: float = 0.05  # 5% default slippage
    min_liquidity_threshold: float = 1000.0  # Minimum $1k liquidity
    confidence_threshold: float = 0.5  # Minimum edge confidence
    
    # Hybrid mode settings
    tenderly_profit_threshold_usd: float = 10.0  # Use Tenderly if profit > $10
    tenderly_amount_threshold_usd: float = 1000.0  # Use Tenderly if trade > $1k
    
    # Gas estimation
    default_gas_price_gwei: float = 20.0
    eth_price_usd: float = 2000.0  # Default ETH price if oracle fails
    
    # Performance settings
    max_concurrent_simulations: int = 10
    cache_ttl_seconds: float = 60.0
    
    # Local simulation fallback
    local_rpc_url: str = "http://localhost:8545"
    anvil_fork_block_number: Optional[int] = None


class HybridPathSimulator:
    """
    Unified path simulator with multiple simulation modes.
    
    Provides fast basic mathematical simulation for initial filtering,
    detailed Tenderly on-chain simulation for validation, and a hybrid
    mode that intelligently combines both approaches.
    """
    
    def __init__(
        self,
        redis_client,
        asset_oracle,
        config: Optional[SimulatorConfig] = None,
        tenderly_config: Optional[TenderlyConfig] = None
    ):
        """
        Initialize the hybrid path simulator.
        
        Args:
            redis_client: Redis client for edge state caching
            asset_oracle: Asset price oracle for USD conversions
            config: Simulator configuration
            tenderly_config: Tenderly API configuration
        """
        self.redis_client = redis_client
        self.asset_oracle = asset_oracle
        self.config = config or SimulatorConfig()
        self.tenderly_config = tenderly_config
        
        # HTTP session for Tenderly API
        self.tenderly_session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.simulation_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_simulations
        )
        
        # Statistics
        self._stats = {
            "basic_simulations": 0,
            "tenderly_simulations": 0,
            "hybrid_simulations": 0,
            "local_simulations": 0,
            "simulation_errors": 0,
            "total_profit_found_usd": 0.0,
        }
        
        logger.info(
            f"Initialized HybridPathSimulator with modes: "
            f"{'TENDERLY ' if tenderly_config else ''}"
            f"BASIC HYBRID LOCAL"
        )
    
    async def initialize(self) -> None:
        """Initialize async resources."""
        if self.tenderly_config and not self.tenderly_session:
            self.tenderly_session = aiohttp.ClientSession(
                headers={
                    "X-Access-Key": self.tenderly_config.api_key,
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(
                    total=self.tenderly_config.timeout_seconds
                )
            )
            logger.info("Initialized Tenderly API session")
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.tenderly_session:
            await self.tenderly_session.close()
            self.tenderly_session = None
            logger.info("Closed Tenderly API session")
    
    async def simulate_path(
        self,
        path: List[YieldGraphEdge],
        initial_amount: float,
        start_asset_id: str,
        mode: SimulationMode = SimulationMode.HYBRID,
        block_number: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate a path execution with the specified mode.
        
        Args:
            path: List of edges forming the arbitrage path
            initial_amount: Starting amount in start_asset units
            start_asset_id: ID of the starting asset
            mode: Simulation mode to use
            block_number: Optional block number for simulation
            
        Returns:
            SimulationResult with profitability analysis
        """
        async with self.simulation_semaphore:
            try:
                import time
                start_time = time.time()
                
                # Validate inputs
                if not path:
                    return SimulationResult(
                        success=False,
                        simulation_mode=mode.value,
                        revert_reason="Empty path provided"
                    )
                
                if initial_amount <= 0:
                    return SimulationResult(
                        success=False,
                        simulation_mode=mode.value,
                        revert_reason="Initial amount must be positive"
                    )
                
                # Edge validation (for non-basic modes or if explicitly requested)
                validation_warnings = []
                if mode != SimulationMode.BASIC:
                    validation_result = await self._validate_path_edges(path)
                    
                    # If validation fails critically, abort simulation
                    if not validation_result["valid"]:
                        issues_str = "; ".join(validation_result["issues"][:3])  # First 3 issues
                        return SimulationResult(
                            success=False,
                            simulation_mode=mode.value,
                            revert_reason=f"Path validation failed: {issues_str}",
                            warnings=validation_result["warnings"]
                        )
                    
                    # Collect warnings for later
                    validation_warnings = validation_result.get("warnings", [])
                
                # Route to appropriate simulation method
                if mode == SimulationMode.BASIC:
                    result = await self._simulate_basic(
                        path, initial_amount, start_asset_id
                    )
                elif mode == SimulationMode.TENDERLY:
                    if not self.tenderly_config:
                        return SimulationResult(
                            success=False,
                            simulation_mode=mode.value,
                            revert_reason="Tenderly not configured"
                        )
                    result = await self._simulate_tenderly(
                        path, initial_amount, start_asset_id, block_number
                    )
                elif mode == SimulationMode.HYBRID:
                    result = await self._simulate_hybrid(
                        path, initial_amount, start_asset_id, block_number
                    )
                else:  # LOCAL mode
                    result = await self._simulate_local(
                        path, initial_amount, start_asset_id, block_number
                    )
                
                # Add timing information
                result.simulation_time_ms = (time.time() - start_time) * 1000
                
                # Add validation warnings to result
                if validation_warnings:
                    if result.warnings:
                        result.warnings.extend(validation_warnings)
                    else:
                        result.warnings = validation_warnings
                
                # Extract call graph from Tenderly trace (if available)
                if result.tenderly_trace:
                    call_graph = self._extract_call_graph_from_trace(result.tenderly_trace)
                    if call_graph and call_graph.get("total_calls", 0) > 0:
                        # Add call graph info to path_details
                        if not result.path_details:
                            result.path_details = []
                        result.path_details.append({
                            "type": "call_graph",
                            "data": call_graph
                        })
                        logger.debug(f"Extracted call graph: {call_graph['total_calls']} calls, "
                                   f"{len(call_graph.get('unique_contracts', []))} contracts")
                
                # Update statistics
                self._update_stats(result)
                
                return result
                
            except Exception as e:
                logger.error(f"Simulation error: {e}", exc_info=True)
                self._stats["simulation_errors"] += 1
                return SimulationResult(
                    success=False,
                    simulation_mode=mode.value,
                    revert_reason=f"Simulation error: {str(e)}"
                )
    
    async def _simulate_basic(
        self,
        path: List[YieldGraphEdge],
        initial_amount: float,
        start_asset_id: str
    ) -> SimulationResult:
        """Fast mathematical simulation using EdgeState from Redis."""
        import time
        start_time = time.time()
        
        current_asset_id = start_asset_id
        current_amount = initial_amount
        total_gas_usd_cost = 0.0
        path_details_log = []
        warnings = []
        
        logger.debug(f"Starting basic simulation: {start_asset_id} -> {initial_amount}")
        
        # Process each edge in the path
        for i, edge in enumerate(path):
            try:
                logger.debug(f"Processing edge {i+1}/{len(path)}: {edge.edge_id}")
                
                # Validate edge matches current position
                if edge.source_asset_id != current_asset_id:
                    return SimulationResult(
                        success=False,
                        simulation_mode=SimulationMode.BASIC.value,
                        revert_reason=f"Edge {edge.edge_id} source {edge.source_asset_id} != current asset {current_asset_id}",
                        failed_at_step=i + 1,
                        path_details=path_details_log
                    )
                
                # Get edge state from Redis
                edge_state = await self._get_edge_state(edge.edge_id)
                if not edge_state:
                    return SimulationResult(
                        success=False,
                        simulation_mode=SimulationMode.BASIC.value,
                        revert_reason=f"No state available for edge {edge.edge_id}",
                        failed_at_step=i + 1,
                        path_details=path_details_log
                    )
                
                # Check edge state confidence
                if edge_state.confidence_score < self.config.confidence_threshold:
                    warnings.append(f"Low confidence edge {edge.edge_id}: {edge_state.confidence_score:.2f}")
                    
                    # If confidence is very low, fail the simulation
                    if edge_state.confidence_score < self.config.confidence_threshold * 0.5:
                        return SimulationResult(
                            success=False,
                            simulation_mode=SimulationMode.BASIC.value,
                            revert_reason=f"Edge {edge.edge_id} confidence too low: {edge_state.confidence_score:.2f}",
                            failed_at_step=i + 1,
                            path_details=path_details_log,
                            warnings=warnings
                        )
                
                # Check for valid conversion rate
                if edge_state.conversion_rate is None or edge_state.conversion_rate <= 0:
                    return SimulationResult(
                        success=False,
                        simulation_mode=SimulationMode.BASIC.value,
                        revert_reason=f"Invalid conversion rate for edge {edge.edge_id}: {edge_state.conversion_rate}",
                        failed_at_step=i + 1,
                        path_details=path_details_log
                    )
                
                # Calculate USD value for slippage estimation
                current_asset_price_usd = await self.asset_oracle.get_price_usd(current_asset_id)
                if current_asset_price_usd is None:
                    # Use default ETH price if oracle fails
                    if "ETH" in current_asset_id or "WETH" in current_asset_id:
                        current_asset_price_usd = self.config.eth_price_usd
                        warnings.append(f"Using default ETH price for {current_asset_id}")
                    else:
                        return SimulationResult(
                            success=False,
                            simulation_mode=SimulationMode.BASIC.value,
                            revert_reason=f"Could not get price for {current_asset_id}",
                            failed_at_step=i + 1,
                            path_details=path_details_log
                        )
                
                trade_amount_usd = current_amount * current_asset_price_usd
                
                # Estimate slippage impact
                slippage_impact = await self._estimate_slippage_impact(
                    trade_amount_usd, edge_state.liquidity_usd
                )
                
                # Check if liquidity is sufficient
                if edge_state.liquidity_usd and trade_amount_usd > edge_state.liquidity_usd * 0.1:
                    warnings.append(
                        f"Large trade relative to liquidity: ${trade_amount_usd:.2f} vs ${edge_state.liquidity_usd:.2f}"
                    )
                
                # Apply conversion rate and slippage
                effective_conversion_rate = edge_state.conversion_rate * (1 - slippage_impact)
                output_amount_before_gas = current_amount * effective_conversion_rate
                
                # Calculate gas cost
                gas_cost_usd = edge_state.gas_cost_usd or self._estimate_gas_cost_usd(edge)
                total_gas_usd_cost += gas_cost_usd
                
                # Convert gas cost to target asset units
                target_asset_price_usd = await self.asset_oracle.get_price_usd(edge.target_asset_id)
                if target_asset_price_usd is None:
                    # Use default ETH price for ETH-based assets
                    if "ETH" in edge.target_asset_id or "WETH" in edge.target_asset_id:
                        target_asset_price_usd = self.config.eth_price_usd
                        warnings.append(f"Using default ETH price for {edge.target_asset_id}")
                    else:
                        # Assume 1:1 USD conversion for stablecoins
                        if any(stable in edge.target_asset_id for stable in ["USDC", "USDT", "DAI"]):
                            target_asset_price_usd = 1.0
                            warnings.append(f"Assuming $1 price for stablecoin {edge.target_asset_id}")
                        else:
                            return SimulationResult(
                                success=False,
                                simulation_mode=SimulationMode.BASIC.value,
                                revert_reason=f"Could not get price for target asset {edge.target_asset_id}",
                                failed_at_step=i + 1,
                                path_details=path_details_log
                            )
                
                gas_cost_in_target_asset = gas_cost_usd / target_asset_price_usd
                final_output_amount = output_amount_before_gas - gas_cost_in_target_asset
                
                # Check if step is profitable
                if final_output_amount <= 0:
                    return SimulationResult(
                        success=False,
                        simulation_mode=SimulationMode.BASIC.value,
                        revert_reason=f"Step {i+1} became unprofitable: {final_output_amount:.6f}",
                        failed_at_step=i + 1,
                        path_details=path_details_log,
                        warnings=warnings
                    )
                
                # Log step details
                step_detail = {
                    "step": i + 1,
                    "edge_id": edge.edge_id,
                    "input_asset": current_asset_id,
                    "input_amount": current_amount,
                    "output_asset": edge.target_asset_id,
                    "conversion_rate": edge_state.conversion_rate,
                    "effective_conversion_rate": effective_conversion_rate,
                    "slippage_impact": slippage_impact,
                    "output_before_gas": output_amount_before_gas,
                    "gas_cost_usd": gas_cost_usd,
                    "gas_cost_target_asset": gas_cost_in_target_asset,
                    "final_output": final_output_amount,
                    "confidence_score": edge_state.confidence_score,
                    "liquidity_usd": edge_state.liquidity_usd,
                    "trade_amount_usd": trade_amount_usd
                }
                path_details_log.append(step_detail)
                
                # Update for next iteration
                current_amount = final_output_amount
                current_asset_id = edge.target_asset_id
                
                self._stats["edges_evaluated"] = self._stats.get("edges_evaluated", 0) + 1
                
            except Exception as e:
                logger.error(f"Error in basic simulation step {i+1}: {e}")
                return SimulationResult(
                    success=False,
                    simulation_mode=SimulationMode.BASIC.value,
                    revert_reason=f"Simulation error at step {i+1}: {str(e)}",
                    failed_at_step=i + 1,
                    path_details=path_details_log
                )
        
        # Calculate final profit metrics
        if current_asset_id != start_asset_id:
            warnings.append(f"Path does not return to start asset: {current_asset_id} != {start_asset_id}")
            
            # Try to convert final amount to start asset for comparison
            final_asset_price = await self.asset_oracle.get_price_usd(current_asset_id)
            start_asset_price = await self.asset_oracle.get_price_usd(start_asset_id)
            
            if final_asset_price and start_asset_price:
                equivalent_start_amount = (current_amount * final_asset_price) / start_asset_price
                profit_amount_start_asset = equivalent_start_amount - initial_amount
                is_cycle = False
            else:
                profit_amount_start_asset = 0.0
                equivalent_start_amount = 0.0
                is_cycle = False
        else:
            # Perfect arbitrage cycle
            profit_amount_start_asset = current_amount - initial_amount
            equivalent_start_amount = current_amount
            is_cycle = True
        
        # Calculate profit in USD
        start_asset_price_usd = await self.asset_oracle.get_price_usd(start_asset_id)
        if start_asset_price_usd:
            profit_usd = profit_amount_start_asset * start_asset_price_usd
            profit_percentage = (profit_amount_start_asset / initial_amount) * 100
        else:
            profit_usd = 0.0
            profit_percentage = 0.0
        
        # Check if profitable after gas costs
        net_profit_usd = profit_usd - total_gas_usd_cost
        is_profitable = net_profit_usd > 0
        
        simulation_time_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self._stats["basic_simulations"] += 1
        if is_profitable:
            self._stats["profitable_simulations"] = self._stats.get("profitable_simulations", 0) + 1
        
        logger.debug(
            f"Basic simulation complete: profit=${net_profit_usd:.4f}, "
            f"gas=${total_gas_usd_cost:.4f}, time={simulation_time_ms:.1f}ms"
        )
        
        return SimulationResult(
            success=is_profitable,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=profit_usd,
            profit_amount_start_asset=profit_amount_start_asset,
            profit_percentage=profit_percentage,
            gas_cost_usd=total_gas_usd_cost,
            output_amount=current_amount,
            final_amount_start_asset=equivalent_start_amount,
            slippage_estimate=sum(detail.get("slippage_impact", 0) for detail in path_details_log) / len(path_details_log) if path_details_log else 0,
            warnings=warnings,
            path_details=path_details_log,
            simulation_time_ms=simulation_time_ms
        )
    
    async def _simulate_tenderly(
        self,
        path: List[YieldGraphEdge],
        initial_amount: float,
        start_asset_id: str,
        block_number: Optional[int] = None
    ) -> SimulationResult:
        """Detailed on-chain simulation using Tenderly API."""
        import time
        from .tenderly_client import TenderlyNetworkId
        from .transaction_builder import TransactionBuilder
        
        start_time = time.time()
        
        if not self.tenderly_config:
            return SimulationResult(
                success=False,
                simulation_mode=SimulationMode.TENDERLY.value,
                revert_reason="Tenderly not configured"
            )
        
        # Initialize Tenderly client if needed
        if not hasattr(self, '_tenderly_client'):
            from .tenderly_client import TenderlyClient
            self._tenderly_client = TenderlyClient(
                api_key=self.tenderly_config.api_key,
                username=self.tenderly_config.username,
                project_slug=self.tenderly_config.project_slug,
                base_url=self.tenderly_config.base_url,
                timeout_seconds=self.tenderly_config.timeout_seconds,
                max_retries=self.tenderly_config.max_retries
            )
            await self._tenderly_client.initialize()
        
        # Initialize transaction builder
        tx_builder = TransactionBuilder()
        
        try:
            # Create fork for simulation
            logger.info(f"Creating Tenderly fork for path simulation")
            fork = await self._tenderly_client.create_fork(
                network_id=TenderlyNetworkId.ETHEREUM,
                block_number=block_number,
                description=f"Path simulation: {len(path)} edges"
            )
            
            # Use a realistic test address for simulation
            simulation_address = "0x000000000000000000000000000000000000dead"
            
            try:
                # Build transactions for the complete path
                logger.debug(f"Building transactions for {len(path)} edges")
                transactions = tx_builder.build_path_transactions(
                    path=path,
                    initial_amount=initial_amount,
                    from_address=simulation_address,
                    current_block=block_number
                )
                
                if len(transactions) != len(path):
                    return SimulationResult(
                        success=False,
                        simulation_mode=SimulationMode.TENDERLY.value,
                        revert_reason=f"Transaction count mismatch: {len(transactions)} != {len(path)}"
                    )
                
                # Simulate each transaction in sequence on the fork
                simulation_results = []
                total_gas_used = 0
                path_details = []
                warnings = []
                
                logger.info(f"Simulating {len(transactions)} transactions on fork {fork.fork_id}")
                
                for i, (edge, transaction) in enumerate(zip(path, transactions)):
                    logger.debug(f"Simulating step {i+1}/{len(path)}: {edge.edge_id}")
                    
                    try:
                        # Simulate the transaction
                        sim_result = await self._tenderly_client.simulate_transaction(
                            transaction=transaction,
                            network_id=TenderlyNetworkId.ETHEREUM,
                            block_number=block_number,
                            fork_id=fork.fork_id,
                            save_if_fails=True
                        )
                        
                        simulation_results.append(sim_result)
                        total_gas_used += sim_result.gas_used
                        
                        # Log step details
                        step_detail = {
                            "step": i + 1,
                            "edge_id": edge.edge_id,
                            "transaction_hash": sim_result.transaction_hash,
                            "success": sim_result.success,
                            "gas_used": sim_result.gas_used,
                            "simulation_time_ms": sim_result.simulation_time_ms,
                            "from_address": transaction.from_address,
                            "to_address": transaction.to_address,
                            "input_data": transaction.data[:20] + "..." if len(transaction.data) > 20 else transaction.data
                        }
                        
                        if not sim_result.success:
                            step_detail.update({
                                "error_message": sim_result.error_message,
                                "revert_reason": sim_result.revert_reason
                            })
                        
                        path_details.append(step_detail)
                        
                        # If any step fails, stop simulation
                        if not sim_result.success:
                            revert_reason = sim_result.revert_reason or sim_result.error_message or f"Step {i+1} reverted"
                            logger.warning(f"Simulation failed at step {i+1}: {revert_reason}")
                            
                            return SimulationResult(
                                success=False,
                                simulation_mode=SimulationMode.TENDERLY.value,
                                gas_used=total_gas_used,
                                revert_reason=revert_reason,
                                failed_at_step=i + 1,
                                path_details=path_details,
                                tenderly_fork_id=fork.fork_id,
                                simulation_time_ms=(time.time() - start_time) * 1000
                            )
                        
                        # Add warnings for high gas usage
                        if sim_result.gas_used > 500_000:
                            warnings.append(f"High gas usage in step {i+1}: {sim_result.gas_used:,}")
                        
                    except Exception as e:
                        logger.error(f"Error simulating step {i+1}: {e}", exc_info=True)
                        return SimulationResult(
                            success=False,
                            simulation_mode=SimulationMode.TENDERLY.value,
                            revert_reason=f"Simulation error at step {i+1}: {str(e)}",
                            failed_at_step=i + 1,
                            path_details=path_details,
                            simulation_time_ms=(time.time() - start_time) * 1000
                        )
                
                # All steps completed successfully
                logger.info(f"Path simulation completed successfully. Total gas: {total_gas_used:,}")
                
                # Calculate gas costs in USD
                gas_cost_usd = await self._calculate_gas_cost_usd(total_gas_used)
                
                # For profit calculation, we need to analyze the final state
                # This is complex as we need to track token balances through the fork
                # For now, we'll mark as successful but indicate that profit calculation
                # requires additional state analysis
                
                final_trace = simulation_results[-1].trace if simulation_results else None
                
                # Add summary warnings
                if total_gas_used > 1_000_000:
                    warnings.append(f"Very high total gas usage: {total_gas_used:,}")
                
                avg_sim_time = sum(r.simulation_time_ms or 0 for r in simulation_results) / len(simulation_results)
                if avg_sim_time > 1000:  # > 1 second average
                    warnings.append(f"Slow simulation average: {avg_sim_time:.1f}ms per step")
                
                simulation_time_ms = (time.time() - start_time) * 1000
                
                return SimulationResult(
                    success=True,
                    simulation_mode=SimulationMode.TENDERLY.value,
                    gas_used=total_gas_used,
                    gas_cost_usd=gas_cost_usd,
                    warnings=warnings,
                    path_details=path_details,
                    tenderly_trace=final_trace,
                    tenderly_fork_id=fork.fork_id,
                    simulation_time_ms=simulation_time_ms
                )
                
            except Exception as e:
                logger.error(f"Error during path simulation: {e}", exc_info=True)
                return SimulationResult(
                    success=False,
                    simulation_mode=SimulationMode.TENDERLY.value,
                    revert_reason=f"Path simulation error: {str(e)}",
                    simulation_time_ms=(time.time() - start_time) * 1000
                )
            
            finally:
                # Clean up fork
                try:
                    await self._tenderly_client.delete_fork(fork.fork_id)
                    logger.debug(f"Cleaned up fork {fork.fork_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup fork {fork.fork_id}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to create Tenderly fork: {e}", exc_info=True)
            return SimulationResult(
                success=False,
                simulation_mode=SimulationMode.TENDERLY.value,
                revert_reason=f"Fork creation failed: {str(e)}",
                simulation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _simulate_hybrid(
        self,
        path: List[YieldGraphEdge],
        initial_amount: float,
        start_asset_id: str,
        block_number: Optional[int] = None
    ) -> SimulationResult:
        """
        Hybrid simulation: Basic simulation first for fast filtering,
        then Tenderly for validation if the path meets criteria.
        
        This provides the best of both worlds:
        - Fast mathematical simulation for initial feasibility check
        - Accurate on-chain simulation for promising paths
        """
        import time
        start_time = time.time()
        logger.info(f"Starting hybrid simulation for {len(path)}-step path")
        
        # Phase 1: Basic simulation for fast filtering
        logger.debug("Phase 1: Running basic simulation for initial filtering")
        basic_result = await self._simulate_basic(path, initial_amount, start_asset_id)
        
        # If basic simulation fails, no need to proceed to Tenderly
        if not basic_result.success:
            logger.info(f"Basic simulation failed: {basic_result.revert_reason}")
            return SimulationResult(
                success=False,
                simulation_mode=SimulationMode.HYBRID.value,
                revert_reason=f"Basic filter failed: {basic_result.revert_reason}",
                output_amount=basic_result.output_amount,
                gas_cost_usd=basic_result.gas_cost_usd,
                slippage_estimate=basic_result.slippage_estimate,
                warnings=basic_result.warnings + ["Failed basic simulation filter"],
                simulation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Phase 2: Analyze if this path warrants Tenderly validation
        should_use_tenderly = await self._should_use_tenderly_validation(
            path=path,
            initial_amount=initial_amount,
            basic_result=basic_result
        )
        
        if not should_use_tenderly:
            logger.info("Path does not meet Tenderly criteria, using basic result")
            return SimulationResult(
                success=basic_result.success,
                simulation_mode=SimulationMode.HYBRID.value,
                profit_usd=basic_result.profit_usd,
                profit_percentage=basic_result.profit_percentage,
                output_amount=basic_result.output_amount,
                gas_cost_usd=basic_result.gas_cost_usd,
                slippage_estimate=basic_result.slippage_estimate,
                warnings=basic_result.warnings + ["Used basic simulation only"],
                simulation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Phase 3: Tenderly validation for promising paths
        logger.info("Path meets Tenderly criteria, running on-chain validation")
        tenderly_result = await self._simulate_tenderly(path, initial_amount, start_asset_id, block_number)
        
        # Phase 4: If Tenderly fails, try local simulation as fallback
        if not tenderly_result.success and self.config.local_rpc_url:
            logger.warning(f"Tenderly failed: {tenderly_result.revert_reason}, trying local simulation fallback")
            local_result = await self._simulate_local(path, initial_amount, start_asset_id, block_number)
            
            if local_result.success:
                logger.info("Local simulation succeeded after Tenderly failure")
                # Use local result but note the fallback in warnings
                local_result.warnings = (local_result.warnings or []) + [
                    f"Tenderly failed: {tenderly_result.revert_reason}",
                    "Used local simulation as fallback"
                ]
                # Update simulation time for the entire hybrid operation
                local_result.simulation_time_ms = (time.time() - start_time) * 1000
                final_result = local_result
            else:
                logger.warning("Both Tenderly and local simulation failed")
                # Combine both failures
                final_result = self._combine_hybrid_results(basic_result, tenderly_result, start_time)
                final_result.warnings = (final_result.warnings or []) + [
                    f"Local fallback also failed: {local_result.revert_reason}"
                ]
        else:
            # Phase 5: Combine results intelligently (original logic)
            final_result = self._combine_hybrid_results(basic_result, tenderly_result, start_time)
        
        logger.info(f"Hybrid simulation complete: {final_result.simulation_mode} in {final_result.simulation_time_ms or 0:.1f}ms")
        return final_result
    
    
    async def _get_edge_state(self, edge_id: str) -> Optional[EdgeState]:
        """Get edge state from Redis cache."""
        try:
            state_json = await self.redis_client.get(f"edge_state:{edge_id}")
            if state_json:
                return EdgeState.model_validate_json(state_json)
            return None
        except Exception as e:
            logger.warning(f"Failed to get edge state for {edge_id}: {e}")
            return None
    
    async def _estimate_slippage_impact(
        self,
        trade_amount_usd: float,
        liquidity_usd: Optional[float]
    ) -> float:
        """
        Estimate slippage based on trade size and liquidity.
        
        Args:
            trade_amount_usd: Trade size in USD
            liquidity_usd: Available liquidity in USD
            
        Returns:
            Estimated slippage factor (0.0 to 1.0)
        """
        if liquidity_usd is None or liquidity_usd < self.config.min_liquidity_threshold:
            return self.config.default_slippage_factor
        
        # Simple square root model for price impact
        # slippage = k * sqrt(trade_size / liquidity)
        slippage_factor = 0.1  # k factor
        impact = slippage_factor * (trade_amount_usd / liquidity_usd) ** 0.5
        
        return min(impact, 0.99)  # Cap at 99%
    
    async def _should_use_tenderly_validation(
        self,
        path: List[YieldGraphEdge],
        initial_amount: float,
        basic_result: SimulationResult
    ) -> bool:
        """
        Determine if a path should use Tenderly validation based on:
        1. Potential profit threshold
        2. Trade amount threshold
        3. Path complexity
        4. Risk factors
        """
        # Check profit threshold
        if basic_result.profit_usd and basic_result.profit_usd >= self.config.tenderly_profit_threshold_usd:
            logger.debug(f"Profit threshold met: ${basic_result.profit_usd:.2f} >= ${self.config.tenderly_profit_threshold_usd}")
            return True
        
        # Check trade amount threshold
        try:
            start_asset_price = await self.asset_oracle.get_price_usd(path[0].source_asset_id)
            trade_amount_usd = initial_amount * (start_asset_price or self.config.eth_price_usd)
            
            if trade_amount_usd >= self.config.tenderly_amount_threshold_usd:
                logger.debug(f"Trade amount threshold met: ${trade_amount_usd:.2f} >= ${self.config.tenderly_amount_threshold_usd}")
                return True
        except Exception as e:
            logger.warning(f"Failed to get asset price for threshold check: {e}")
        
        # Check path complexity - complex paths need validation
        if len(path) >= 4:  # 4+ step paths are complex
            logger.debug(f"Complex path detected: {len(path)} steps")
            return True
        
        # Check for risky edge types that need validation
        risky_edges = [EdgeType.FLASH_LOAN, EdgeType.BRIDGE, EdgeType.BACK_RUN]
        if any(edge.edge_type in risky_edges for edge in path):
            logger.debug("Risky edge types detected, requiring Tenderly validation")
            return True
        
        # Check for high slippage estimate
        if basic_result.slippage_estimate and basic_result.slippage_estimate > 0.02:  # > 2%
            logger.debug(f"High slippage detected: {basic_result.slippage_estimate:.2%}")
            return True
        
        # Check for multiple protocols (cross-protocol paths are riskier)
        protocols = set(edge.protocol_name for edge in path)
        if len(protocols) >= 3:
            logger.debug(f"Multi-protocol path detected: {len(protocols)} protocols")
            return True
        
        logger.debug("Path does not meet Tenderly validation criteria")
        return False
    
    async def _simulate_local(
        self,
        path: List[YieldGraphEdge],
        initial_amount: float,
        start_asset_id: str,
        block_number: Optional[int] = None
    ) -> SimulationResult:
        """
        Local EVM simulation using Anvil or Hardhat as fallback.
        
        This provides on-chain accuracy when Tenderly is unavailable,
        using a local fork of the blockchain state.
        """
        import time
        import subprocess
        import json
        import tempfile
        import os
        from .transaction_builder import TransactionBuilder
        
        start_time = time.time()
        logger.info(f"Starting local simulation for {len(path)}-step path")
        
        # Initialize transaction builder
        tx_builder = TransactionBuilder()
        
        # Check if Anvil is available
        try:
            result = subprocess.run(["anvil", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return SimulationResult(
                    success=False,
                    simulation_mode=SimulationMode.LOCAL.value,
                    revert_reason="Anvil not available - install foundry",
                    simulation_time_ms=(time.time() - start_time) * 1000
                )
            logger.debug(f"Anvil version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return SimulationResult(
                success=False,
                simulation_mode=SimulationMode.LOCAL.value,
                revert_reason=f"Anvil not found: {str(e)}",
                simulation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Create temporary directory for local simulation
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Start Anvil fork
                fork_block = block_number or "latest"
                anvil_cmd = [
                    "anvil",
                    "--fork-url", "https://rpc.ankr.com/eth",  # Use public RPC
                    "--fork-block-number", str(fork_block) if fork_block != "latest" else fork_block,
                    "--port", "8545",
                    "--host", "127.0.0.1",
                    "--accounts", "10",
                    "--balance", "10000",  # 10,000 ETH per account
                    "--chain-id", "1"
                ]
                
                if self.config.anvil_fork_block_number:
                    anvil_cmd.extend(["--fork-block-number", str(self.config.anvil_fork_block_number)])
                
                logger.debug(f"Starting Anvil: {' '.join(anvil_cmd)}")
                
                # Start Anvil process
                anvil_process = subprocess.Popen(
                    anvil_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temp_dir
                )
                
                # Wait for Anvil to start (check if port is responding)
                import socket
                import time
                
                max_wait = 10  # seconds
                wait_start = time.time()
                anvil_ready = False
                
                while time.time() - wait_start < max_wait:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(1)
                        result = sock.connect_ex(("127.0.0.1", 8545))
                        sock.close()
                        
                        if result == 0:
                            anvil_ready = True
                            break
                    except Exception:
                        pass
                    
                    time.sleep(0.5)
                
                if not anvil_ready:
                    anvil_process.terminate()
                    return SimulationResult(
                        success=False,
                        simulation_mode=SimulationMode.LOCAL.value,
                        revert_reason="Anvil failed to start or become ready",
                        simulation_time_ms=(time.time() - start_time) * 1000
                    )
                
                logger.info("Anvil fork started successfully")
                
                try:
                    # Build transactions for the path
                    simulation_address = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"  # Anvil default account
                    
                    transactions = tx_builder.build_path_transactions(
                        path=path,
                        initial_amount=initial_amount,
                        from_address=simulation_address,
                        current_block=block_number
                    )
                    
                    # Simulate each transaction using cast (foundry's CLI tool)
                    simulation_results = []
                    total_gas_used = 0
                    path_details = []
                    warnings = []
                    
                    for i, (edge, transaction) in enumerate(zip(path, transactions)):
                        logger.debug(f"Simulating local step {i+1}/{len(path)}: {edge.edge_id}")
                        
                        try:
                            # Use cast to simulate the transaction
                            cast_cmd = [
                                "cast", "call",
                                "--rpc-url", "http://127.0.0.1:8545",
                                "--from", transaction.from_address,
                                transaction.to_address,
                                transaction.data,
                                "--value", str(transaction.value or 0)
                            ]
                            
                            # Simulate the call
                            sim_result = subprocess.run(
                                cast_cmd,
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            
                            if sim_result.returncode != 0:
                                revert_reason = sim_result.stderr.strip() or "Transaction would revert"
                                logger.warning(f"Local simulation failed at step {i+1}: {revert_reason}")
                                
                                return SimulationResult(
                                    success=False,
                                    simulation_mode=SimulationMode.LOCAL.value,
                                    revert_reason=f"Step {i+1} failed: {revert_reason}",
                                    failed_at_step=i + 1,
                                    path_details=path_details,
                                    simulation_time_ms=(time.time() - start_time) * 1000
                                )
                            
                            # Estimate gas for this transaction
                            gas_cmd = [
                                "cast", "estimate",
                                "--rpc-url", "http://127.0.0.1:8545", 
                                "--from", transaction.from_address,
                                transaction.to_address,
                                transaction.data,
                                "--value", str(transaction.value or 0)
                            ]
                            
                            gas_result = subprocess.run(
                                gas_cmd,
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            
                            step_gas = 200_000  # Default
                            if gas_result.returncode == 0:
                                try:
                                    step_gas = int(gas_result.stdout.strip())
                                except ValueError:
                                    warnings.append(f"Could not parse gas estimate for step {i+1}")
                            
                            total_gas_used += step_gas
                            
                            # Actually send the transaction to update state for next step
                            send_cmd = [
                                "cast", "send",
                                "--rpc-url", "http://127.0.0.1:8545",
                                "--private-key", "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # Anvil default private key
                                transaction.to_address,
                                transaction.data,
                                "--value", str(transaction.value or 0)
                            ]
                            
                            send_result = subprocess.run(
                                send_cmd,
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            
                            tx_hash = "unknown"
                            if send_result.returncode == 0:
                                tx_hash = send_result.stdout.strip()
                            
                            # Log step details
                            step_detail = {
                                "step": i + 1,
                                "edge_id": edge.edge_id,
                                "transaction_hash": tx_hash,
                                "success": True,
                                "gas_used": step_gas,
                                "from_address": transaction.from_address,
                                "to_address": transaction.to_address,
                                "simulation_method": "anvil_local"
                            }
                            
                            path_details.append(step_detail)
                            
                            # Add warnings for high gas usage
                            if step_gas > 500_000:
                                warnings.append(f"High gas usage in step {i+1}: {step_gas:,}")
                            
                        except subprocess.TimeoutExpired:
                            return SimulationResult(
                                success=False,
                                simulation_mode=SimulationMode.LOCAL.value,
                                revert_reason=f"Step {i+1} simulation timeout",
                                failed_at_step=i + 1,
                                path_details=path_details,
                                simulation_time_ms=(time.time() - start_time) * 1000
                            )
                        except Exception as e:
                            logger.error(f"Error in local simulation step {i+1}: {e}")
                            return SimulationResult(
                                success=False,
                                simulation_mode=SimulationMode.LOCAL.value,
                                revert_reason=f"Local simulation error at step {i+1}: {str(e)}",
                                failed_at_step=i + 1,
                                path_details=path_details,
                                simulation_time_ms=(time.time() - start_time) * 1000
                            )
                    
                    # Calculate gas costs
                    gas_cost_usd = await self._calculate_gas_cost_usd(total_gas_used)
                    
                    # Add summary warnings
                    if total_gas_used > 1_000_000:
                        warnings.append(f"Very high total gas usage: {total_gas_used:,}")
                    
                    simulation_time_ms = (time.time() - start_time) * 1000
                    
                    # Update statistics
                    self._stats["local_simulations"] += 1
                    
                    logger.info(f"Local simulation completed successfully. Total gas: {total_gas_used:,}")
                    
                    return SimulationResult(
                        success=True,
                        simulation_mode=SimulationMode.LOCAL.value,
                        gas_used=total_gas_used,
                        gas_cost_usd=gas_cost_usd,
                        warnings=warnings,
                        path_details=path_details,
                        simulation_time_ms=simulation_time_ms
                    )
                
                finally:
                    # Clean up Anvil process
                    try:
                        anvil_process.terminate()
                        anvil_process.wait(timeout=5)
                        logger.debug("Anvil process terminated")
                    except subprocess.TimeoutExpired:
                        anvil_process.kill()
                        logger.warning("Had to kill Anvil process")
                    except Exception as e:
                        logger.warning(f"Error cleaning up Anvil: {e}")
            
            except Exception as e:
                logger.error(f"Error in local simulation setup: {e}", exc_info=True)
                return SimulationResult(
                    success=False,
                    simulation_mode=SimulationMode.LOCAL.value,
                    revert_reason=f"Local simulation setup error: {str(e)}",
                    simulation_time_ms=(time.time() - start_time) * 1000
                )
    
    def _combine_hybrid_results(
        self,
        basic_result: SimulationResult,
        tenderly_result: SimulationResult,
        start_time: float
    ) -> SimulationResult:
        """
        Intelligently combine results from basic and Tenderly simulations.
        
        Priority order:
        1. If Tenderly succeeds, use its results (most accurate)
        2. If Tenderly fails but basic succeeds, use basic with warnings
        3. If both fail, combine error information
        """
        total_time_ms = (time.time() - start_time) * 1000
        
        # Case 1: Tenderly succeeded - use its results as authoritative
        if tenderly_result.success:
            logger.debug("Using Tenderly results as authoritative")
            
            # Compare with basic simulation for warnings
            warnings = list(tenderly_result.warnings or [])
            
            # Check for significant differences
            if (basic_result.profit_usd and tenderly_result.profit_usd and 
                abs(basic_result.profit_usd - tenderly_result.profit_usd) > 5.0):
                warnings.append(f"Profit discrepancy: Basic=${basic_result.profit_usd:.2f}, Tenderly=${tenderly_result.profit_usd:.2f}")
            
            if (basic_result.gas_cost_usd and tenderly_result.gas_cost_usd and
                abs(basic_result.gas_cost_usd - tenderly_result.gas_cost_usd) > 2.0):
                warnings.append(f"Gas cost discrepancy: Basic=${basic_result.gas_cost_usd:.2f}, Tenderly=${tenderly_result.gas_cost_usd:.2f}")
            
            return SimulationResult(
                success=True,
                simulation_mode=SimulationMode.HYBRID.value,
                profit_usd=tenderly_result.profit_usd,
                profit_percentage=tenderly_result.profit_percentage,
                output_amount=tenderly_result.output_amount,
                gas_used=tenderly_result.gas_used,
                gas_cost_usd=tenderly_result.gas_cost_usd,
                slippage_estimate=tenderly_result.slippage_estimate,
                warnings=warnings,
                path_details=tenderly_result.path_details,
                tenderly_trace=tenderly_result.tenderly_trace,
                tenderly_fork_id=tenderly_result.tenderly_fork_id,
                simulation_time_ms=total_time_ms
            )
        
        # Case 2: Tenderly failed but basic succeeded - use basic with warnings
        elif basic_result.success:
            logger.warning(f"Tenderly failed but basic succeeded: {tenderly_result.revert_reason}")
            
            warnings = list(basic_result.warnings or [])
            warnings.append(f"Tenderly validation failed: {tenderly_result.revert_reason}")
            warnings.append("Using basic simulation results - may be less accurate")
            
            return SimulationResult(
                success=True,
                simulation_mode=SimulationMode.HYBRID.value,
                profit_usd=basic_result.profit_usd,
                profit_percentage=basic_result.profit_percentage,
                output_amount=basic_result.output_amount,
                gas_cost_usd=basic_result.gas_cost_usd,
                slippage_estimate=basic_result.slippage_estimate,
                warnings=warnings,
                simulation_time_ms=total_time_ms
            )
        
        # Case 3: Both failed - combine error information
        else:
            logger.error("Both basic and Tenderly simulations failed")
            
            combined_reason = f"Basic: {basic_result.revert_reason}; Tenderly: {tenderly_result.revert_reason}"
            
            warnings = []
            if basic_result.warnings:
                warnings.extend(basic_result.warnings)
            if tenderly_result.warnings:
                warnings.extend(tenderly_result.warnings)
            warnings.append("Both simulation modes failed")
            
            return SimulationResult(
                success=False,
                simulation_mode=SimulationMode.HYBRID.value,
                revert_reason=combined_reason,
                warnings=warnings,
                simulation_time_ms=total_time_ms
            )
    
    def _estimate_gas_cost_usd(self, edge: YieldGraphEdge) -> float:
        """
        Estimate gas cost for an edge in USD.
        
        Args:
            edge: The edge to estimate gas cost for
            
        Returns:
            Estimated gas cost in USD
        """
        # Default gas estimates by edge type
        gas_estimates = {
            EdgeType.TRADE: 150_000,       # DEX swaps
            EdgeType.LEND: 200_000,        # Lending operations
            EdgeType.BORROW: 200_000,      # Borrowing operations
            EdgeType.STAKE: 180_000,       # Staking operations
            EdgeType.BRIDGE: 300_000,      # Cross-chain bridges
            EdgeType.FLASH_LOAN: 100_000,  # Flash loan initiation
            EdgeType.BACK_RUN: 120_000,    # MEV back-run
        }
        
        # Get base gas estimate
        estimated_gas = gas_estimates.get(edge.edge_type, 200_000)
        
        # Protocol-specific adjustments
        if edge.protocol_name:
            protocol_multipliers = {
                "uniswapv2": 1.0,
                "uniswapv3": 1.2,      # More complex
                "sushiswap": 1.0,
                "compound": 1.5,       # More complex lending
                "aave": 1.4,           # Complex lending
                "curve": 1.3,          # Complex AMM
                "balancer": 1.4,       # Complex AMM
                "1inch": 1.6,          # Aggregator overhead
                "0x": 1.5,             # DEX aggregator
            }
            
            multiplier = protocol_multipliers.get(edge.protocol_name.lower(), 1.2)
            estimated_gas = int(estimated_gas * multiplier)
        
        # Convert to USD
        gas_price_wei = self.config.default_gas_price_gwei * 1e9  # Convert gwei to wei
        gas_cost_eth = (estimated_gas * gas_price_wei) / 1e18     # Convert wei to ETH
        gas_cost_usd = gas_cost_eth * self.config.eth_price_usd
        
        return gas_cost_usd
    
    async def _calculate_gas_cost_usd(self, total_gas_used: int) -> float:
        """
        Calculate gas cost in USD based on current gas price and ETH price.
        
        Args:
            total_gas_used: Total gas units used
            
        Returns:
            Gas cost in USD
        """
        try:
            # Get current ETH price from oracle
            eth_price_usd = await self.asset_oracle.get_price_usd("ETH_MAINNET_WETH")
            if eth_price_usd is None:
                eth_price_usd = self.config.eth_price_usd  # Fallback to config default
            
            # Convert gas to ETH cost
            gas_price_wei = self.config.default_gas_price_gwei * 1e9  # Convert gwei to wei
            gas_cost_eth = (total_gas_used * gas_price_wei) / 1e18   # Convert wei to ETH
            
            # Convert to USD
            gas_cost_usd = gas_cost_eth * eth_price_usd
            
            return gas_cost_usd
            
        except Exception as e:
            logger.warning(f"Failed to calculate gas cost in USD: {e}")
            # Fallback calculation
            return total_gas_used * 0.00002  # Rough estimate: $0.00002 per gas unit
    
    def _update_stats(self, result: SimulationResult) -> None:
        """Update internal statistics."""
        mode_key = f"{result.simulation_mode}_simulations"
        if mode_key in self._stats:
            self._stats[mode_key] += 1
        
        if result.success and result.profit_usd:
            self._stats["total_profit_found_usd"] += result.profit_usd
    
    async def _validate_path_edges(self, path: List[YieldGraphEdge]) -> Dict[str, Any]:
        """
        Validate all edges in a path before simulation.
        
        Args:
            path: List of edges to validate
            
        Returns:
            Validation result with details
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "edge_states": {},
            "missing_states": [],
            "stale_states": [],
            "low_confidence": []
        }
        
        logger.debug(f"Validating {len(path)} edges")
        
        for i, edge in enumerate(path):
            edge_validation = await self._validate_single_edge(edge, i)
            
            # Collect edge state
            if edge_validation["state"]:
                validation_result["edge_states"][edge.edge_id] = edge_validation["state"]
            
            # Collect issues
            if edge_validation["issues"]:
                validation_result["issues"].extend(edge_validation["issues"])
                validation_result["valid"] = False
            
            # Collect warnings
            if edge_validation["warnings"]:
                validation_result["warnings"].extend(edge_validation["warnings"])
            
            # Track specific validation categories
            if edge_validation["missing_state"]:
                validation_result["missing_states"].append(edge.edge_id)
            
            if edge_validation["stale_state"]:
                validation_result["stale_states"].append(edge.edge_id)
            
            if edge_validation["low_confidence"]:
                validation_result["low_confidence"].append(edge.edge_id)
        
        # Path-level validations
        await self._validate_path_coherence(path, validation_result)
        
        logger.info(f"Path validation: {'✅ PASS' if validation_result['valid'] else '❌ FAIL'}")
        if validation_result["issues"]:
            logger.warning(f"Validation issues: {len(validation_result['issues'])}")
        if validation_result["warnings"]:
            logger.info(f"Validation warnings: {len(validation_result['warnings'])}")
        
        return validation_result
    
    async def _validate_single_edge(self, edge: YieldGraphEdge, position: int) -> Dict[str, Any]:
        """
        Validate a single edge and its state.
        
        Args:
            edge: Edge to validate
            position: Position in path (for context)
            
        Returns:
            Single edge validation result
        """
        result = {
            "issues": [],
            "warnings": [],
            "state": None,
            "missing_state": False,
            "stale_state": False,
            "low_confidence": False
        }
        
        # 1. Basic edge structure validation
        if not edge.edge_id:
            result["issues"].append(f"Edge {position}: Missing edge_id")
        
        if not edge.source_asset_id or not edge.target_asset_id:
            result["issues"].append(f"Edge {position}: Missing source/target asset IDs")
        
        if edge.source_asset_id == edge.target_asset_id:
            result["issues"].append(f"Edge {position}: Source and target assets are the same")
        
        if not edge.protocol_name:
            result["warnings"].append(f"Edge {position}: Missing protocol name")
        
        # 2. Edge state validation
        edge_state = await self._get_edge_state(edge.edge_id)
        
        if not edge_state:
            result["missing_state"] = True
            result["warnings"].append(f"Edge {position}: No state data available")
        else:
            result["state"] = edge_state
            
            # Check if state is stale
            if edge_state.is_stale(max_age_seconds=300):  # 5 minutes
                result["stale_state"] = True
                result["warnings"].append(f"Edge {position}: State data is stale")
            
            # Check confidence
            if edge_state.confidence_score < self.config.confidence_threshold:
                result["low_confidence"] = True
                result["warnings"].append(f"Edge {position}: Low confidence ({edge_state.confidence_score:.2f})")
            
            # Check liquidity
            if edge_state.liquidity_usd and edge_state.liquidity_usd < self.config.min_liquidity_threshold:
                result["warnings"].append(f"Edge {position}: Low liquidity (${edge_state.liquidity_usd:,.0f})")
            
            # Check conversion rate
            if edge_state.conversion_rate is None or edge_state.conversion_rate <= 0:
                result["issues"].append(f"Edge {position}: Invalid conversion rate")
        
        # 3. Protocol-specific validations
        protocol_issues = self._validate_protocol_specific(edge, position)
        result["issues"].extend(protocol_issues.get("issues", []))
        result["warnings"].extend(protocol_issues.get("warnings", []))
        
        return result
    
    def _validate_protocol_specific(self, edge: YieldGraphEdge, position: int) -> Dict[str, List[str]]:
        """
        Perform protocol-specific validations.
        
        Args:
            edge: Edge to validate
            position: Position in path
            
        Returns:
            Protocol-specific validation results
        """
        issues = []
        warnings = []
        
        protocol = edge.protocol_name.lower() if edge.protocol_name else ""
        
        # Uniswap validations
        if "uniswap" in protocol:
            if edge.edge_type != EdgeType.TRADE:
                issues.append(f"Edge {position}: Uniswap edges should be TRADE type")
            
            # Check for common Uniswap asset pairs
            if "WETH" in edge.source_asset_id and "USDC" in edge.target_asset_id:
                # This is a common pair, should have good liquidity
                pass
        
        # Aave validations
        elif "aave" in protocol:
            if edge.edge_type not in [EdgeType.LEND, EdgeType.BORROW, EdgeType.FLASH_LOAN]:
                warnings.append(f"Edge {position}: Unexpected edge type for Aave")
        
        # Compound validations
        elif "compound" in protocol:
            if edge.edge_type not in [EdgeType.LEND, EdgeType.BORROW]:
                warnings.append(f"Edge {position}: Unexpected edge type for Compound")
        
        # Bridge validations
        elif "bridge" in protocol or edge.edge_type == EdgeType.BRIDGE:
            if edge.chain_name and hasattr(edge, 'target_chain_name'):
                if edge.chain_name == getattr(edge, 'target_chain_name', None):
                    issues.append(f"Edge {position}: Bridge with same source/target chain")
            
            warnings.append(f"Edge {position}: Bridge operations have higher risk")
        
        # Flash loan validations
        if edge.edge_type == EdgeType.FLASH_LOAN:
            warnings.append(f"Edge {position}: Flash loan requires careful gas management")
        
        # MEV validations
        if edge.edge_type == EdgeType.BACK_RUN:
            warnings.append(f"Edge {position}: MEV operation requires timing considerations")
        
        return {"issues": issues, "warnings": warnings}
    
    async def _validate_path_coherence(self, path: List[YieldGraphEdge], validation_result: Dict[str, Any]) -> None:
        """
        Validate that the path edges connect properly.
        
        Args:
            path: Complete path to validate
            validation_result: Validation result to update
        """
        if len(path) < 2:
            return  # Single edge paths don't need coherence validation
        
        for i in range(len(path) - 1):
            current_edge = path[i]
            next_edge = path[i + 1]
            
            # Check that edges connect (target of current = source of next)
            if current_edge.target_asset_id != next_edge.source_asset_id:
                validation_result["issues"].append(
                    f"Path disconnect at step {i}->{i+1}: "
                    f"{current_edge.target_asset_id} != {next_edge.source_asset_id}"
                )
                validation_result["valid"] = False
            
            # Check for circular paths
            if current_edge.source_asset_id == next_edge.target_asset_id and len(path) == 2:
                validation_result["warnings"].append("Detected 2-step circular arbitrage")
            
            # Check chain consistency
            if current_edge.chain_name != next_edge.chain_name:
                if next_edge.edge_type != EdgeType.BRIDGE:
                    validation_result["warnings"].append(
                        f"Chain change without bridge at step {i}->{i+1}"
                    )
        
        # Check for path efficiency
        asset_sequence = [path[0].source_asset_id] + [edge.target_asset_id for edge in path]
        if len(set(asset_sequence)) < len(asset_sequence) - 1:  # Allow start/end to be same for arbitrage
            validation_result["warnings"].append("Path contains redundant asset conversions")
    
    def _extract_call_graph_from_trace(self, tenderly_trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract call graph information from Tenderly simulation trace.
        
        Args:
            tenderly_trace: Raw Tenderly trace data
            
        Returns:
            Extracted call graph information
        """
        call_graph = {
            "total_calls": 0,
            "unique_contracts": set(),
            "call_hierarchy": [],
            "gas_by_contract": {},
            "failed_calls": [],
            "external_calls": [],
            "storage_reads": 0,
            "storage_writes": 0,
            "events_emitted": [],
            "protocol_interactions": {}
        }
        
        if not tenderly_trace:
            return call_graph
        
        try:
            # Extract from transaction trace
            if "transaction" in tenderly_trace:
                tx_trace = tenderly_trace["transaction"]
                
                # Process call trace
                if "trace" in tx_trace:
                    self._process_trace_calls(tx_trace["trace"], call_graph, 0)
                
                # Extract events
                if "logs" in tx_trace:
                    call_graph["events_emitted"] = self._extract_events(tx_trace["logs"])
                
                # Extract gas information
                if "gas_used" in tx_trace:
                    call_graph["total_gas_used"] = tx_trace["gas_used"]
            
            # Process call hierarchy for protocol detection
            self._detect_protocol_interactions(call_graph)
            
            # Convert sets to lists for JSON serialization
            call_graph["unique_contracts"] = list(call_graph["unique_contracts"])
            
            logger.debug(f"Extracted call graph: {call_graph['total_calls']} calls, "
                        f"{len(call_graph['unique_contracts'])} contracts")
            
        except Exception as e:
            logger.warning(f"Failed to extract call graph: {e}")
            call_graph["extraction_error"] = str(e)
        
        return call_graph
    
    def _process_trace_calls(self, trace: Dict[str, Any], call_graph: Dict[str, Any], depth: int) -> None:
        """Process individual trace calls recursively."""
        if not isinstance(trace, dict):
            return
        
        call_graph["total_calls"] += 1
        
        # Extract contract address
        to_address = trace.get("to", "").lower()
        if to_address:
            call_graph["unique_contracts"].add(to_address)
            
            # Track gas usage by contract
            gas_used = trace.get("gasUsed", 0)
            # Handle string gas values
            if isinstance(gas_used, str):
                try:
                    gas_used = int(gas_used)
                except ValueError:
                    gas_used = 0
            
            if to_address not in call_graph["gas_by_contract"]:
                call_graph["gas_by_contract"][to_address] = 0
            call_graph["gas_by_contract"][to_address] += gas_used
        
        # Build call hierarchy
        call_info = {
            "depth": depth,
            "to": to_address,
            "input": trace.get("input", "")[:10],  # First 10 chars of input data
            "gas_used": gas_used,
            "call_type": trace.get("type", ""),
            "success": trace.get("error") is None
        }
        
        if not call_info["success"]:
            call_graph["failed_calls"].append(call_info)
        
        call_graph["call_hierarchy"].append(call_info)
        
        # Process subcalls
        if "calls" in trace and isinstance(trace["calls"], list):
            for subcall in trace["calls"]:
                self._process_trace_calls(subcall, call_graph, depth + 1)
    
    def _extract_events(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and categorize events from transaction logs."""
        events = []
        
        for log in logs:
            if not isinstance(log, dict):
                continue
                
            event_info = {
                "address": log.get("address", "").lower(),
                "topics": log.get("topics", []),
                "data": log.get("data", ""),
                "name": self._decode_event_name(log.get("topics", []))
            }
            
            events.append(event_info)
        
        return events
    
    def _decode_event_name(self, topics: List[str]) -> str:
        """Attempt to decode event name from topics."""
        if not topics or not topics[0]:
            return "unknown"
        
        # Common event signatures (topic0 -> name mapping)
        common_events = {
            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef": "Transfer",
            "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925": "Approval", 
            "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1": "Sync",
            "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822": "Swap",
            "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9": "PairCreated"
        }
        
        return common_events.get(topics[0].lower(), f"unknown_{topics[0][:10]}")
    
    def _detect_protocol_interactions(self, call_graph: Dict[str, Any]) -> None:
        """Detect which DeFi protocols were interacted with."""
        # Known protocol contract addresses (simplified)
        protocol_contracts = {
            "uniswap_v2": ["0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f"],
            "uniswap_v3": ["0x1f98431c8ad98523631ae4a59f267346ea31f984"],
            "sushiswap": ["0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac"],
            "aave": ["0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9"],
            "compound": ["0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b"],
            "curve": ["0x79a8c46dea5ada233abaffd40f3a0a2b1e5a4f27"]
        }
        
        detected_protocols = {}
        
        for contract in call_graph["unique_contracts"]:
            for protocol, addresses in protocol_contracts.items():
                if any(addr.lower() == contract for addr in addresses):
                    if protocol not in detected_protocols:
                        detected_protocols[protocol] = []
                    detected_protocols[protocol].append(contract)
        
        call_graph["protocol_interactions"] = detected_protocols
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return self._stats.copy()
    
    async def validate_edge(
        self,
        edge: YieldGraphEdge,
        input_amount: float,
        mode: SimulationMode = SimulationMode.BASIC
    ) -> SimulationResult:
        """
        Validate a single edge execution.
        
        Args:
            edge: Edge to validate
            input_amount: Input amount for the edge
            mode: Simulation mode to use
            
        Returns:
            SimulationResult for the single edge
        """
        # Create a single-edge path
        path = [edge]
        return await self.simulate_path(
            path,
            input_amount,
            edge.source_asset_id,
            mode
        )