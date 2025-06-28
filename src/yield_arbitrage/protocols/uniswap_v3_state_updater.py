"""Advanced state updater for Uniswap V3 edges with real-time market data."""
import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

from web3 import Web3

from yield_arbitrage.protocols.abis.uniswap_v3 import (
    UNISWAP_V3_POOL_ABI,
    UNISWAP_V3_QUOTER_ABI,
    ERC20_ABI
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState

logger = logging.getLogger(__name__)


@dataclass
class PoolStateSnapshot:
    """Comprehensive snapshot of pool state at a specific time."""
    pool_address: str
    block_number: int
    timestamp: datetime
    
    # Core pool data
    liquidity: int
    sqrt_price_x96: int
    tick: int
    fee_growth_global_0_x128: int
    fee_growth_global_1_x128: int
    protocol_fees_token0: int
    protocol_fees_token1: int
    
    # Token balances
    token0_balance: int
    token1_balance: int
    
    # Calculated values
    price_token0_per_token1: float
    price_token1_per_token0: float
    tvl_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    
    # Market indicators
    price_impact_1_percent: Optional[float] = None
    price_impact_5_percent: Optional[float] = None
    effective_liquidity: Optional[float] = None


@dataclass
class StateUpdateConfig:
    """Configuration for state update operations."""
    max_concurrent_updates: int = 20
    update_timeout_seconds: int = 30
    price_staleness_threshold_seconds: int = 300  # 5 minutes
    enable_price_impact_calculation: bool = True
    enable_volume_tracking: bool = True
    cache_pool_states: bool = True
    cache_ttl_seconds: int = 60  # 1 minute
    retry_failed_updates: bool = True
    max_retries: int = 2


class UniswapV3StateUpdater:
    """Advanced state updater for Uniswap V3 trading edges."""
    
    def __init__(self, provider, chain_name: str, config: Optional[StateUpdateConfig] = None):
        self.provider = provider
        self.chain_name = chain_name
        self.config = config or StateUpdateConfig()
        
        # Contract instances
        self.web3 = None
        self.quoter_contract = None
        
        # State caching
        self.pool_state_cache: Dict[str, PoolStateSnapshot] = {}
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}
        self.last_cache_cleanup = datetime.now(timezone.utc)
        
        # Performance tracking
        self.update_stats = {
            "updates_performed": 0,
            "updates_failed": 0,
            "cache_hits": 0,
            "avg_update_time_ms": 0.0,
            "last_update": None
        }
        
        # Rate limiting
        self.update_semaphore = asyncio.Semaphore(self.config.max_concurrent_updates)
        
    async def initialize(self, quoter_address: str) -> bool:
        """Initialize the state updater."""
        try:
            self.web3 = await self.provider.get_web3(self.chain_name)
            if not self.web3:
                logger.error(f"Failed to get Web3 instance for {self.chain_name}")
                return False
            
            # Initialize quoter contract
            self.quoter_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(quoter_address),
                abi=UNISWAP_V3_QUOTER_ABI
            )
            
            logger.info(f"UniswapV3StateUpdater initialized for {self.chain_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize state updater: {e}")
            return False
    
    async def update_edge_state(self, edge: YieldGraphEdge, metadata: Dict[str, Any]) -> EdgeState:
        """Update edge state with comprehensive market data."""
        start_time = datetime.now(timezone.utc)
        
        async with self.update_semaphore:
            try:
                # Extract pool information
                pool_address = metadata["pool_address"]
                token0_address = metadata["token0_address"]
                token1_address = metadata["token1_address"]
                fee_tier = metadata["fee_tier"]
                
                # Get comprehensive pool state
                pool_snapshot = await self._get_pool_state_snapshot(
                    pool_address, token0_address, token1_address, metadata
                )
                
                if not pool_snapshot:
                    self._update_performance_stats(start_time, success=False)
                    return await self._create_degraded_state(edge, "Failed to get pool snapshot")
                
                # Calculate conversion rates with multiple sample sizes
                conversion_rates = await self._calculate_conversion_rates(
                    token0_address, token1_address, fee_tier, metadata
                )
                
                # Calculate price impact for different trade sizes
                price_impacts = await self._calculate_price_impacts(
                    pool_address, token0_address, token1_address, fee_tier, metadata
                ) if self.config.enable_price_impact_calculation else {}
                
                # Get volume data if enabled
                volume_data = await self._get_volume_data(pool_address) if self.config.enable_volume_tracking else {}
                
                # Calculate delta exposure (market risk)
                delta_exposure = await self._calculate_delta_exposure(
                    pool_snapshot, metadata
                )
                
                # Create enhanced edge state
                new_state = EdgeState(
                    conversion_rate=conversion_rates.get("base_rate"),
                    liquidity_usd=pool_snapshot.tvl_usd,
                    gas_cost_usd=self._estimate_gas_cost(),
                    delta_exposure=delta_exposure,
                    last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                    confidence_score=self._calculate_confidence_score(pool_snapshot, conversion_rates, price_impacts)
                )
                
                # Add custom metadata to state for advanced features
                if hasattr(new_state, '_custom_data'):
                    new_state._custom_data = {
                        "pool_snapshot": pool_snapshot,
                        "conversion_rates": conversion_rates,
                        "price_impacts": price_impacts,
                        "volume_data": volume_data,
                        "update_latency_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    }
                
                # Update statistics
                self._update_performance_stats(start_time, success=True)
                
                return new_state
                
            except Exception as e:
                logger.error(f"Error updating edge state for {edge.edge_id}: {e}")
                self._update_performance_stats(start_time, success=False)
                return await self._create_degraded_state(edge, str(e))
    
    async def _get_pool_state_snapshot(self, pool_address: str, token0: str, token1: str, metadata: Dict) -> Optional[PoolStateSnapshot]:
        """Get comprehensive pool state snapshot."""
        try:
            # Check cache first
            if self.config.cache_pool_states:
                cached_snapshot = self._get_cached_pool_state(pool_address)
                if cached_snapshot:
                    self.update_stats["cache_hits"] += 1
                    return cached_snapshot
            
            # Get current block
            current_block = await self.web3.eth.get_block_number()
            
            # Create pool contract with checksum address
            pool_contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(pool_address),
                abi=UNISWAP_V3_POOL_ABI
            )
            
            # Create token contracts with checksum addresses
            token0_contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token0),
                abi=ERC20_ABI
            )
            token1_contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token1),
                abi=ERC20_ABI
            )
            
            # Batch call for efficiency
            pool_calls = await asyncio.gather(
                pool_contract.functions.liquidity().call(),
                pool_contract.functions.slot0().call(),
                pool_contract.functions.feeGrowthGlobal0X128().call(),
                pool_contract.functions.feeGrowthGlobal1X128().call(),
                pool_contract.functions.protocolFees().call(),
                token0_contract.functions.balanceOf(Web3.to_checksum_address(pool_address)).call(),
                token1_contract.functions.balanceOf(Web3.to_checksum_address(pool_address)).call(),
                return_exceptions=True
            )
            
            # Check for errors
            for call_result in pool_calls:
                if isinstance(call_result, Exception):
                    logger.warning(f"Pool state call failed: {call_result}")
                    return None
            
            (liquidity, slot0, fee_growth_0, fee_growth_1, 
             protocol_fees, token0_balance, token1_balance) = pool_calls
            
            # Parse slot0 data
            sqrt_price_x96, tick, observation_index, observation_cardinality, observation_cardinality_next, fee_protocol, unlocked = slot0
            
            # Calculate prices
            decimals0 = metadata.get("token0_decimals", 18)
            decimals1 = metadata.get("token1_decimals", 18)
            
            price_token0_per_token1, price_token1_per_token0 = self._calculate_prices_from_sqrt(
                sqrt_price_x96, decimals0, decimals1
            )
            
            # Estimate TVL
            tvl_usd = await self._estimate_pool_tvl_usd(
                token0_balance, token1_balance, decimals0, decimals1,
                price_token0_per_token1, metadata
            )
            
            # Parse protocol fees (returns tuple of (token0_fees, token1_fees))
            if isinstance(protocol_fees, (list, tuple)) and len(protocol_fees) >= 2:
                protocol_fees_token0 = protocol_fees[0]
                protocol_fees_token1 = protocol_fees[1]
            else:
                # Fallback if protocol_fees is an integer (shouldn't happen but handle gracefully)
                protocol_fees_token0 = 0
                protocol_fees_token1 = 0
            
            # Create snapshot
            snapshot = PoolStateSnapshot(
                pool_address=pool_address,
                block_number=current_block,
                timestamp=datetime.now(timezone.utc),
                liquidity=liquidity,
                sqrt_price_x96=sqrt_price_x96,
                tick=tick,
                fee_growth_global_0_x128=fee_growth_0,
                fee_growth_global_1_x128=fee_growth_1,
                protocol_fees_token0=protocol_fees_token0,
                protocol_fees_token1=protocol_fees_token1,
                token0_balance=token0_balance,
                token1_balance=token1_balance,
                price_token0_per_token1=price_token0_per_token1,
                price_token1_per_token0=price_token1_per_token0,
                tvl_usd=tvl_usd
            )
            
            # Cache the snapshot
            if self.config.cache_pool_states:
                self.pool_state_cache[pool_address] = snapshot
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error getting pool state snapshot for {pool_address}: {e}")
            return None
    
    async def _calculate_conversion_rates(self, token0: str, token1: str, fee_tier: int, metadata: Dict) -> Dict[str, float]:
        """Calculate conversion rates for different trade sizes."""
        try:
            rates = {}
            
            # Get token decimals
            decimals0 = metadata.get("token0_decimals", 18)
            decimals1 = metadata.get("token1_decimals", 18)
            
            # Sample trade sizes (in normalized units)
            sample_sizes = [
                1,        # 1 unit
                100,      # 100 units  
                1000,     # 1K units
                10000,    # 10K units
                100000    # 100K units
            ]
            
            # Calculate rates for each sample size
            for size in sample_sizes:
                try:
                    # Scale to token decimals
                    scaled_amount = int(size * (10 ** decimals0))
                    
                    # Get quote from contract
                    quote_result = await self.quoter_contract.functions.quoteExactInputSingle(
                        self.web3.to_checksum_address(token0),
                        self.web3.to_checksum_address(token1),
                        fee_tier,
                        scaled_amount,
                        0  # sqrtPriceLimitX96 (0 = no limit)
                    ).call()
                    
                    # Extract amount out
                    amount_out = quote_result if isinstance(quote_result, int) else quote_result[0]
                    
                    # Calculate normalized rate
                    normalized_out = amount_out / (10 ** decimals1)
                    rate = normalized_out / size if size > 0 else 0
                    
                    rates[f"rate_{size}"] = rate
                    
                except Exception as e:
                    logger.debug(f"Failed to get rate for size {size}: {e}")
                    rates[f"rate_{size}"] = None
            
            # Calculate base rate (using smallest successful sample)
            for size in sample_sizes:
                if rates.get(f"rate_{size}") is not None:
                    rates["base_rate"] = rates[f"rate_{size}"]
                    break
            
            return rates
            
        except Exception as e:
            logger.error(f"Error calculating conversion rates: {e}")
            return {"base_rate": None}
    
    async def _calculate_price_impacts(self, pool_address: str, token0: str, token1: str, fee_tier: int, metadata: Dict) -> Dict[str, float]:
        """Calculate price impact for different trade sizes."""
        try:
            impacts = {}
            
            # Get pool's current sqrt price
            pool_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(pool_address),
                abi=UNISWAP_V3_POOL_ABI
            )
            
            slot0 = await pool_contract.functions.slot0().call()
            current_sqrt_price = slot0[0]
            
            decimals0 = metadata.get("token0_decimals", 18)
            decimals1 = metadata.get("token1_decimals", 18)
            
            # Calculate current price
            current_price = self._sqrt_price_to_price(current_sqrt_price, decimals0, decimals1)
            
            # Test different trade sizes as percentage of liquidity
            test_percentages = [0.01, 0.05, 0.1]  # 1%, 5%, 10%
            
            for pct in test_percentages:
                try:
                    # Estimate trade size based on pool liquidity
                    # This is a simplified calculation
                    estimated_trade_size = int(1000000 * (10 ** decimals0) * pct)  # Rough estimate
                    
                    # Get quote for this trade size
                    quote_result = await self.quoter_contract.functions.quoteExactInputSingle(
                        self.web3.to_checksum_address(token0),
                        self.web3.to_checksum_address(token1),
                        fee_tier,
                        estimated_trade_size,
                        0
                    ).call()
                    
                    amount_out = quote_result if isinstance(quote_result, int) else quote_result[0]
                    
                    # Calculate effective price
                    effective_price = (amount_out / (10 ** decimals1)) / (estimated_trade_size / (10 ** decimals0))
                    
                    # Calculate price impact
                    price_impact = abs(effective_price - current_price) / current_price if current_price > 0 else 0
                    impacts[f"impact_{int(pct*100)}pct"] = price_impact
                    
                except Exception as e:
                    logger.debug(f"Failed to calculate price impact for {pct*100}%: {e}")
                    impacts[f"impact_{int(pct*100)}pct"] = None
            
            return impacts
            
        except Exception as e:
            logger.error(f"Error calculating price impacts: {e}")
            return {}
    
    async def _get_volume_data(self, pool_address: str) -> Dict[str, float]:
        """Get volume data for the pool (simplified implementation)."""
        try:
            # This is a placeholder implementation
            # In production, you'd integrate with indexing services or event logs
            return {
                "volume_24h_usd": None,
                "trades_24h": None,
                "volume_7d_usd": None
            }
            
        except Exception as e:
            logger.error(f"Error getting volume data: {e}")
            return {}
    
    async def _calculate_delta_exposure(self, snapshot: PoolStateSnapshot, metadata: Dict) -> Dict[str, float]:
        """Calculate delta exposure for risk assessment."""
        try:
            # Calculate exposure based on pool composition
            token0_symbol = metadata.get("token0_symbol", "TOKEN0")
            token1_symbol = metadata.get("token1_symbol", "TOKEN1")
            
            # Simple exposure calculation based on price ratio
            total_value = snapshot.token0_balance + (snapshot.token1_balance * snapshot.price_token1_per_token0)
            
            if total_value > 0:
                token0_exposure = snapshot.token0_balance / total_value
                token1_exposure = (snapshot.token1_balance * snapshot.price_token1_per_token0) / total_value
                
                return {
                    token0_symbol: token0_exposure,
                    token1_symbol: token1_exposure
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating delta exposure: {e}")
            return {}
    
    def _calculate_confidence_score(self, snapshot: PoolStateSnapshot, rates: Dict, impacts: Dict) -> float:
        """Calculate confidence score based on data quality."""
        try:
            score = 1.0
            
            # Reduce score for stale data
            age_seconds = (datetime.now(timezone.utc) - snapshot.timestamp).total_seconds()
            if age_seconds > self.config.price_staleness_threshold_seconds:
                score *= 0.8
            
            # Reduce score if conversion rate failed
            if not rates.get("base_rate"):
                score *= 0.3
            
            # Reduce score for low liquidity
            if snapshot.liquidity < 1000:
                score *= 0.5
            
            # Boost score for recent, high-liquidity pools
            if age_seconds < 60 and snapshot.liquidity > 1000000:
                score = min(1.0, score * 1.1)
            
            return max(0.1, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    async def _estimate_pool_tvl_usd(self, balance0: int, balance1: int, decimals0: int, decimals1: int, 
                                   price_ratio: float, metadata: Dict) -> Optional[float]:
        """Estimate pool TVL in USD."""
        try:
            # Normalize balances
            norm_balance0 = balance0 / (10 ** decimals0)
            norm_balance1 = balance1 / (10 ** decimals1)
            
            # Simple estimation assuming one token has known USD value
            # In production, integrate with price oracles (Chainlink, etc.)
            
            # Rough heuristic based on token symbols
            token0_symbol = metadata.get("token0_symbol", "").upper()
            token1_symbol = metadata.get("token1_symbol", "").upper()
            
            estimated_usd_value = 0
            
            if "USDC" in token0_symbol or "USDT" in token0_symbol or "DAI" in token0_symbol:
                # Token0 is USD stablecoin
                estimated_usd_value = norm_balance0 * 2  # Double for both sides
            elif "USDC" in token1_symbol or "USDT" in token1_symbol or "DAI" in token1_symbol:
                # Token1 is USD stablecoin  
                estimated_usd_value = norm_balance1 * 2  # Double for both sides
            elif "WETH" in token0_symbol or "ETH" in token0_symbol:
                # Token0 is ETH, assume ~$2000
                estimated_usd_value = norm_balance0 * 2000 * 2
            elif "WETH" in token1_symbol or "ETH" in token1_symbol:
                # Token1 is ETH, assume ~$2000
                estimated_usd_value = norm_balance1 * 2000 * 2
            else:
                # Generic estimation
                estimated_usd_value = (norm_balance0 + norm_balance1 * price_ratio) * 100
            
            return max(0, estimated_usd_value)
            
        except Exception as e:
            logger.debug(f"Error estimating TVL: {e}")
            return None
    
    def _calculate_prices_from_sqrt(self, sqrt_price_x96: int, decimals0: int, decimals1: int) -> Tuple[float, float]:
        """Calculate token prices from sqrt price."""
        try:
            # Convert sqrt price to actual price
            price_raw = (sqrt_price_x96 / (2 ** 96)) ** 2
            
            # Adjust for decimals
            price_token0_per_token1 = price_raw * (10 ** (decimals0 - decimals1))
            price_token1_per_token0 = 1 / price_token0_per_token1 if price_token0_per_token1 > 0 else 0
            
            return price_token0_per_token1, price_token1_per_token0
            
        except Exception as e:
            logger.error(f"Error calculating prices from sqrt: {e}")
            return 0.0, 0.0
    
    def _sqrt_price_to_price(self, sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
        """Convert sqrt price to regular price."""
        try:
            price_raw = (sqrt_price_x96 / (2 ** 96)) ** 2
            return price_raw * (10 ** (decimals0 - decimals1))
        except:
            return 0.0
    
    def _get_cached_pool_state(self, pool_address: str) -> Optional[PoolStateSnapshot]:
        """Get cached pool state if still valid."""
        if pool_address not in self.pool_state_cache:
            return None
        
        snapshot = self.pool_state_cache[pool_address]
        age_seconds = (datetime.now(timezone.utc) - snapshot.timestamp).total_seconds()
        
        if age_seconds > self.config.cache_ttl_seconds:
            del self.pool_state_cache[pool_address]
            return None
        
        return snapshot
    
    async def _create_degraded_state(self, edge: YieldGraphEdge, error_msg: str) -> EdgeState:
        """Create degraded state when update fails."""
        logger.warning(f"Creating degraded state for {edge.edge_id}: {error_msg}")
        
        # Create new degraded state based on existing state or defaults
        if edge.state:
            return EdgeState(
                conversion_rate=edge.state.conversion_rate,
                liquidity_usd=edge.state.liquidity_usd,
                gas_cost_usd=edge.state.gas_cost_usd,
                delta_exposure=edge.state.delta_exposure,
                last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                confidence_score=max(0.1, edge.state.confidence_score * 0.5)
            )
        else:
            # Create minimal state if no existing state
            return EdgeState(
                conversion_rate=0.0,
                liquidity_usd=0.0,
                gas_cost_usd=self._estimate_gas_cost(),
                delta_exposure={},
                last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                confidence_score=0.1
            )
    
    def _estimate_gas_cost(self) -> float:
        """Estimate gas cost for swap operation."""
        # Chain-specific gas estimates for Uniswap V3 swaps
        gas_estimates = {
            "ethereum": 15.0,
            "arbitrum": 2.0,
            "base": 1.5,
            "polygon": 0.5,
            "sonic": 0.1,
            "berachain": 0.1
        }
        
        return gas_estimates.get(self.chain_name, 5.0)
    
    def _update_performance_stats(self, start_time: datetime, success: bool) -> None:
        """Update performance statistics."""
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        if success:
            self.update_stats["updates_performed"] += 1
            # Rolling average of update times
            current_avg = self.update_stats["avg_update_time_ms"]
            total_updates = self.update_stats["updates_performed"]
            self.update_stats["avg_update_time_ms"] = (current_avg * (total_updates - 1) + duration_ms) / total_updates
        else:
            self.update_stats["updates_failed"] += 1
        
        self.update_stats["last_update"] = datetime.now(timezone.utc)
    
    def cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = datetime.now(timezone.utc)
        
        # Only cleanup every 5 minutes
        if (current_time - self.last_cache_cleanup).total_seconds() < 300:
            return
        
        expired_pools = []
        for pool_address, snapshot in self.pool_state_cache.items():
            if (current_time - snapshot.timestamp).total_seconds() > self.config.cache_ttl_seconds:
                expired_pools.append(pool_address)
        
        for pool_address in expired_pools:
            del self.pool_state_cache[pool_address]
        
        self.last_cache_cleanup = current_time
        
        if expired_pools:
            logger.info(f"Cleaned up {len(expired_pools)} expired pool state cache entries")
    
    def get_update_stats(self) -> Dict[str, Any]:
        """Get update performance statistics."""
        return {
            **self.update_stats,
            "cache_size": len(self.pool_state_cache),
            "cache_hit_rate": self.update_stats["cache_hits"] / max(1, self.update_stats["updates_performed"]) * 100,
            "success_rate": self.update_stats["updates_performed"] / max(1, self.update_stats["updates_performed"] + self.update_stats["updates_failed"]) * 100,
            "config": {
                "max_concurrent_updates": self.config.max_concurrent_updates,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
                "price_staleness_threshold": self.config.price_staleness_threshold_seconds
            }
        }
    
    async def batch_update_edges(self, edges_with_metadata: List[Tuple[YieldGraphEdge, Dict]]) -> Dict[str, EdgeState]:
        """Update multiple edges in batch for better performance."""
        if not edges_with_metadata:
            return {}
        
        # Create tasks for all updates
        tasks = []
        edge_ids = []
        
        for edge, metadata in edges_with_metadata:
            task = self.update_edge_state(edge, metadata)
            tasks.append(task)
            edge_ids.append(edge.edge_id)
        
        # Execute all updates concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        updated_states = {}
        for edge_id, result in zip(edge_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Batch update failed for {edge_id}: {result}")
            else:
                updated_states[edge_id] = result
        
        logger.info(f"Batch updated {len(updated_states)}/{len(edges_with_metadata)} edges")
        return updated_states