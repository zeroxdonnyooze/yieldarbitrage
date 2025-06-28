"""Advanced Uniswap V3 pool discovery logic for live blockchain integration."""
import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass

from yield_arbitrage.protocols.abis.uniswap_v3 import (
    UNISWAP_V3_FACTORY_ABI,
    UNISWAP_V3_POOL_ABI,
    ERC20_ABI
)
from yield_arbitrage.protocols.contracts import UNISWAP_V3_CONTRACTS

logger = logging.getLogger(__name__)


@dataclass
class PoolDiscoveryConfig:
    """Configuration for pool discovery process."""
    max_pools_per_batch: int = 50
    discovery_timeout_seconds: int = 300  # 5 minutes
    min_liquidity_threshold: float = 10000.0  # $10k minimum TVL
    max_gas_price_gwei: int = 50  # Skip discovery if gas too high
    enable_event_scanning: bool = True
    event_scan_blocks: int = 10000  # Blocks to scan for pool creation events
    retry_failed_pools: bool = True
    max_retries: int = 3


@dataclass
class PoolInfo:
    """Comprehensive pool information."""
    pool_address: str
    token0_address: str
    token1_address: str
    token0_symbol: str
    token1_symbol: str
    token0_decimals: int
    token1_decimals: int
    fee_tier: int
    liquidity: int
    sqrt_price_x96: int
    tick: int
    tick_spacing: int
    protocol_fee: int
    tvl_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    created_block: Optional[int] = None
    created_timestamp: Optional[datetime] = None
    is_active: bool = True


class UniswapV3PoolDiscovery:
    """Advanced pool discovery system for Uniswap V3."""
    
    def __init__(self, provider, chain_name: str = "ethereum", config: Optional[PoolDiscoveryConfig] = None):
        self.provider = provider
        self.chain_name = chain_name
        self.config = config or PoolDiscoveryConfig()
        
        # Contract instances
        self.factory_contract = None
        self.web3 = None
        
        # Discovery state
        self.discovered_pools: Dict[str, PoolInfo] = {}
        self.failed_pools: Set[str] = set()
        self.discovery_stats = {
            "pools_discovered": 0,
            "pools_failed": 0,
            "blocks_scanned": 0,
            "discovery_time": 0.0,
            "last_discovery": None
        }
        
        # Rate limiting
        self.discovery_semaphore = asyncio.Semaphore(self.config.max_pools_per_batch)
        
    async def initialize(self) -> bool:
        """Initialize the pool discovery system."""
        try:
            # Get contract addresses
            contracts = UNISWAP_V3_CONTRACTS.get(self.chain_name)
            if not contracts:
                logger.error(f"No Uniswap V3 contracts configured for {self.chain_name}")
                return False
            
            # Get Web3 instance
            self.web3 = await self.provider.get_web3(self.chain_name)
            if not self.web3:
                logger.error(f"Failed to get Web3 instance for {self.chain_name}")
                return False
            
            # Initialize factory contract
            self.factory_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contracts["factory"]),
                abi=UNISWAP_V3_FACTORY_ABI
            )
            
            logger.info(f"UniswapV3PoolDiscovery initialized for {self.chain_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pool discovery: {e}")
            return False
    
    async def discover_pools_by_tokens(self, token_addresses: List[str]) -> List[PoolInfo]:
        """Discover pools for given token pairs across all fee tiers."""
        if not await self._check_gas_conditions():
            logger.warning("Gas conditions unfavorable for pool discovery")
            return []
        
        start_time = datetime.now(timezone.utc)
        all_pools = []
        
        try:
            # Generate all possible token pairs
            token_pairs = []
            for i, token0 in enumerate(token_addresses):
                for token1 in token_addresses[i + 1:]:
                    token_pairs.append((token0, token1))
            
            logger.info(f"Discovering pools for {len(token_pairs)} token pairs")
            
            # Process in batches
            batch_size = self.config.max_pools_per_batch // 4  # 4 fee tiers per pair
            for i in range(0, len(token_pairs), batch_size):
                batch = token_pairs[i:i + batch_size]
                
                # Create tasks for all fee tiers in this batch
                tasks = []
                for token0, token1 in batch:
                    for fee_tier in [100, 500, 3000, 10000]:  # Standard fee tiers
                        tasks.append(self._discover_single_pool(token0, token1, fee_tier))
                
                # Execute batch
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"Pool discovery error: {result}")
                        self.discovery_stats["pools_failed"] += 1
                    elif result:
                        all_pools.append(result)
                        self.discovered_pools[result.pool_address] = result
                        self.discovery_stats["pools_discovered"] += 1
                
                # Brief pause between batches
                if i + batch_size < len(token_pairs):
                    await asyncio.sleep(0.5)
            
            # Update stats
            discovery_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.discovery_stats["discovery_time"] = discovery_time
            self.discovery_stats["last_discovery"] = start_time
            
            logger.info(f"Discovered {len(all_pools)} pools in {discovery_time:.2f}s")
            return all_pools
            
        except Exception as e:
            logger.error(f"Pool discovery failed: {e}")
            return []
    
    async def discover_pools_by_events(self, from_block: Optional[int] = None) -> List[PoolInfo]:
        """Discover pools by scanning PoolCreated events."""
        if not self.config.enable_event_scanning:
            return []
        
        try:
            # Determine block range
            if from_block is None:
                current_block = await self.web3.eth.get_block_number()
                from_block = max(0, current_block - self.config.event_scan_blocks)
            
            to_block = await self.web3.eth.get_block_number()
            
            logger.info(f"Scanning for PoolCreated events from block {from_block} to {to_block}")
            
            # Get PoolCreated events
            pool_created_filter = self.factory_contract.events.PoolCreated.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            events = await pool_created_filter.get_all_entries()
            logger.info(f"Found {len(events)} PoolCreated events")
            
            # Process events in batches
            pools = []
            batch_size = 20
            
            for i in range(0, len(events), batch_size):
                batch_events = events[i:i + batch_size]
                tasks = [self._process_pool_created_event(event) for event in batch_events]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"Event processing error: {result}")
                    elif result:
                        pools.append(result)
                        self.discovered_pools[result.pool_address] = result
            
            self.discovery_stats["blocks_scanned"] = to_block - from_block
            return pools
            
        except Exception as e:
            logger.error(f"Event-based pool discovery failed: {e}")
            return []
    
    async def _discover_single_pool(self, token0: str, token1: str, fee_tier: int) -> Optional[PoolInfo]:
        """Discover a single pool for specific tokens and fee tier."""
        async with self.discovery_semaphore:
            try:
                # Get pool address from factory
                pool_address = await self.factory_contract.functions.getPool(
                    self.web3.to_checksum_address(token0),
                    self.web3.to_checksum_address(token1),
                    fee_tier
                ).call()
                
                # Check if pool exists
                if pool_address == "0x0000000000000000000000000000000000000000":
                    return None
                
                # Skip if already discovered or failed
                if (pool_address in self.discovered_pools or 
                    pool_address in self.failed_pools):
                    return self.discovered_pools.get(pool_address)
                
                # Get detailed pool information
                pool_info = await self._get_detailed_pool_info(pool_address, token0, token1, fee_tier)
                
                if pool_info and pool_info.is_active:
                    return pool_info
                else:
                    self.failed_pools.add(pool_address)
                    return None
                    
            except Exception as e:
                logger.debug(f"Error discovering pool {token0}/{token1} fee {fee_tier}: {e}")
                return None
    
    async def _get_detailed_pool_info(self, pool_address: str, token0: str, token1: str, fee_tier: int) -> Optional[PoolInfo]:
        """Get comprehensive information about a pool."""
        try:
            pool_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(pool_address),
                abi=UNISWAP_V3_POOL_ABI
            )
            
            # Batch call for efficiency
            calls = await asyncio.gather(
                pool_contract.functions.liquidity().call(),
                pool_contract.functions.slot0().call(),
                pool_contract.functions.tickSpacing().call(),
                self._get_token_info(token0),
                self._get_token_info(token1),
                return_exceptions=True
            )
            
            # Check for errors
            for call_result in calls:
                if isinstance(call_result, Exception):
                    logger.debug(f"Pool info call failed: {call_result}")
                    return None
            
            liquidity, slot0, tick_spacing, token0_info, token1_info = calls
            
            # Parse slot0 data
            sqrt_price_x96, tick, observation_index, observation_cardinality, observation_cardinality_next, fee_protocol, unlocked = slot0
            
            # Check minimum liquidity
            if liquidity < 1000:  # Very low liquidity threshold
                return None
            
            # Estimate TVL (simplified)
            tvl_usd = await self._estimate_pool_tvl(
                pool_address, token0, token1, 
                token0_info["decimals"], token1_info["decimals"],
                liquidity, sqrt_price_x96
            )
            
            # Check TVL threshold
            if tvl_usd and tvl_usd < self.config.min_liquidity_threshold:
                return None
            
            return PoolInfo(
                pool_address=pool_address,
                token0_address=token0.lower(),
                token1_address=token1.lower(),
                token0_symbol=token0_info["symbol"],
                token1_symbol=token1_info["symbol"],
                token0_decimals=token0_info["decimals"],
                token1_decimals=token1_info["decimals"],
                fee_tier=fee_tier,
                liquidity=liquidity,
                sqrt_price_x96=sqrt_price_x96,
                tick=tick,
                tick_spacing=tick_spacing,
                protocol_fee=fee_protocol,
                tvl_usd=tvl_usd,
                is_active=liquidity > 0 and unlocked
            )
            
        except Exception as e:
            logger.debug(f"Error getting pool info for {pool_address}: {e}")
            return None
    
    async def _get_token_info(self, token_address: str) -> Dict[str, Any]:
        """Get token information (symbol, decimals)."""
        try:
            token_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(token_address),
                abi=ERC20_ABI
            )
            
            # Batch call for token info
            symbol, decimals = await asyncio.gather(
                token_contract.functions.symbol().call(),
                token_contract.functions.decimals().call()
            )
            
            return {
                "symbol": symbol,
                "decimals": decimals
            }
            
        except Exception as e:
            logger.debug(f"Error getting token info for {token_address}: {e}")
            return {
                "symbol": "UNKNOWN",
                "decimals": 18
            }
    
    async def _estimate_pool_tvl(self, pool_address: str, token0: str, token1: str,
                                decimals0: int, decimals1: int, liquidity: int, sqrt_price_x96: int) -> Optional[float]:
        """Estimate pool TVL in USD (simplified calculation)."""
        try:
            # This is a simplified TVL estimation
            # In production, you'd want to integrate with price oracles
            
            # Get token balances
            token0_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(token0),
                abi=ERC20_ABI
            )
            token1_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(token1),
                abi=ERC20_ABI
            )
            
            balance0, balance1 = await asyncio.gather(
                token0_contract.functions.balanceOf(pool_address).call(),
                token1_contract.functions.balanceOf(pool_address).call()
            )
            
            # Simple heuristic: assume one token is worth ~$1-$1000
            # This is very rough and should be replaced with real price feeds
            normalized_balance0 = balance0 / (10 ** decimals0)
            normalized_balance1 = balance1 / (10 ** decimals1)
            
            # Rough estimate assuming average token value
            estimated_tvl = (normalized_balance0 + normalized_balance1) * 100  # Very rough estimate
            
            return max(estimated_tvl, 0.0)
            
        except Exception as e:
            logger.debug(f"Error estimating TVL for pool {pool_address}: {e}")
            return None
    
    async def _process_pool_created_event(self, event) -> Optional[PoolInfo]:
        """Process a PoolCreated event to extract pool information."""
        try:
            args = event.args
            pool_address = args.pool
            token0 = args.token0
            token1 = args.token1
            fee = args.fee
            
            # Get pool creation block and timestamp
            block = await self.web3.eth.get_block(event.blockNumber)
            created_timestamp = datetime.fromtimestamp(block.timestamp, tz=timezone.utc)
            
            # Get detailed pool info
            pool_info = await self._get_detailed_pool_info(pool_address, token0, token1, fee)
            
            if pool_info:
                pool_info.created_block = event.blockNumber
                pool_info.created_timestamp = created_timestamp
            
            return pool_info
            
        except Exception as e:
            logger.debug(f"Error processing PoolCreated event: {e}")
            return None
    
    async def _check_gas_conditions(self) -> bool:
        """Check if gas conditions are favorable for discovery."""
        try:
            gas_price = await self.web3.eth.gas_price
            gas_price_gwei = gas_price / 10**9
            
            if gas_price_gwei > self.config.max_gas_price_gwei:
                logger.warning(f"Gas price too high: {gas_price_gwei:.1f} gwei > {self.config.max_gas_price_gwei} gwei")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not check gas conditions: {e}")
            return True  # Proceed anyway
    
    async def refresh_pool_data(self, pool_addresses: List[str]) -> Dict[str, PoolInfo]:
        """Refresh data for existing pools."""
        refreshed_pools = {}
        
        batch_size = 10
        for i in range(0, len(pool_addresses), batch_size):
            batch = pool_addresses[i:i + batch_size]
            tasks = [self._refresh_single_pool(addr) for addr in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for addr, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Pool refresh failed for {addr}: {result}")
                elif result:
                    refreshed_pools[addr] = result
                    self.discovered_pools[addr] = result
        
        return refreshed_pools
    
    async def _refresh_single_pool(self, pool_address: str) -> Optional[PoolInfo]:
        """Refresh data for a single pool."""
        try:
            if pool_address not in self.discovered_pools:
                return None
            
            old_pool = self.discovered_pools[pool_address]
            
            # Get updated pool info
            updated_pool = await self._get_detailed_pool_info(
                pool_address,
                old_pool.token0_address,
                old_pool.token1_address,
                old_pool.fee_tier
            )
            
            if updated_pool:
                # Preserve creation data
                updated_pool.created_block = old_pool.created_block
                updated_pool.created_timestamp = old_pool.created_timestamp
            
            return updated_pool
            
        except Exception as e:
            logger.debug(f"Error refreshing pool {pool_address}: {e}")
            return None
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            **self.discovery_stats,
            "total_pools_cached": len(self.discovered_pools),
            "failed_pools": len(self.failed_pools),
            "config": {
                "max_pools_per_batch": self.config.max_pools_per_batch,
                "min_liquidity_threshold": self.config.min_liquidity_threshold,
                "event_scanning_enabled": self.config.enable_event_scanning
            }
        }
    
    def get_pools_by_token(self, token_address: str) -> List[PoolInfo]:
        """Get all discovered pools containing a specific token."""
        token_lower = token_address.lower()
        return [
            pool for pool in self.discovered_pools.values()
            if pool.token0_address == token_lower or pool.token1_address == token_lower
        ]
    
    def get_pools_by_fee_tier(self, fee_tier: int) -> List[PoolInfo]:
        """Get all discovered pools with a specific fee tier."""
        return [
            pool for pool in self.discovered_pools.values()
            if pool.fee_tier == fee_tier
        ]
    
    def clear_failed_pools(self) -> None:
        """Clear the failed pools cache to retry discovery."""
        self.failed_pools.clear()
        logger.info("Failed pools cache cleared")