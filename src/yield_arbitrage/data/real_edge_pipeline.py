"""Production real edge state pipeline for live DeFi protocol data collection."""
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum

from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType
from yield_arbitrage.protocols.production_registry import production_registry, ProtocolCategory
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.protocols.uniswap_v3_state_updater import UniswapV3StateUpdater, StateUpdateConfig

logger = logging.getLogger(__name__)


class EdgePriority(str, Enum):
    """Priority levels for edge state updates."""
    CRITICAL = "critical"    # High-volume, mission-critical edges (update every block)
    HIGH = "high"           # Important edges (update every 2-3 blocks)
    MEDIUM = "medium"       # Standard edges (update every 5-10 blocks)
    LOW = "low"            # Background edges (update every 30+ blocks)


@dataclass
class EdgeUpdateConfig:
    """Configuration for edge state updates."""
    edge_id: str
    protocol_id: str
    chain_name: str
    priority: EdgePriority
    update_interval_seconds: int
    contract_address: Optional[str] = None
    pool_metadata: Optional[Dict[str, Any]] = None
    last_updated: Optional[float] = None
    consecutive_failures: int = 0
    is_active: bool = True


class RealEdgeStatePipeline:
    """
    Production pipeline for collecting real edge states from live DeFi protocols.
    
    This pipeline:
    - Discovers real pools/edges from production protocols
    - Collects live state data from blockchain contracts
    - Updates edge states with real conversion rates, liquidity, gas costs
    - Manages update priorities and intervals
    - Provides real-time data for arbitrage execution
    """
    
    def __init__(
        self,
        blockchain_provider,
        oracle: OnChainPriceOracle,
        redis_client,
        max_concurrent_updates: int = 10
    ):
        """
        Initialize the real edge state pipeline.
        
        Args:
            blockchain_provider: Blockchain connection provider
            oracle: On-chain price oracle for asset prices
            redis_client: Redis client for caching
            max_concurrent_updates: Max concurrent edge updates
        """
        self.blockchain_provider = blockchain_provider
        self.oracle = oracle
        self.redis_client = redis_client
        self.max_concurrent_updates = max_concurrent_updates
        
        # Edge management
        self.edge_configs: Dict[str, EdgeUpdateConfig] = {}
        self.active_edges: Dict[str, YieldGraphEdge] = {}
        self.edge_states: Dict[str, EdgeState] = {}
        
        # Protocol adapters for state collection
        self.protocol_adapters: Dict[str, Any] = {}
        
        # Update scheduling
        self.update_scheduler_running = False
        self.update_semaphore = asyncio.Semaphore(max_concurrent_updates)
        
        # Statistics
        self.stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "average_update_time_ms": 0.0,
            "edges_discovered": 0,
            "active_edge_count": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the pipeline and discover production edges."""
        logger.info("Initializing real edge state pipeline...")
        
        # Initialize protocol adapters
        await self._initialize_protocol_adapters()
        
        # Discover production edges from real protocols
        await self._discover_production_edges()
        
        # Start the update scheduler
        asyncio.create_task(self._run_update_scheduler())
        
        logger.info(f"Pipeline initialized with {len(self.edge_configs)} edges")
    
    async def discover_edges(self) -> List[YieldGraphEdge]:
        """Public method to discover edges."""
        return list(self.active_edges.values())
    
    async def update_edge_state(self, edge: YieldGraphEdge) -> Optional[YieldGraphEdge]:
        """Public method to update edge state."""
        success = await self._update_edge_state(edge.edge_id)
        if success:
            # Return updated edge
            updated_edge = self.active_edges.get(edge.edge_id)
            if updated_edge:
                return updated_edge
        return None
    
    async def _initialize_protocol_adapters(self) -> None:
        """Initialize adapters for supported protocols."""
        supported_protocols = ["uniswap_v3"]  # Start with Uniswap V3
        
        for protocol_id in supported_protocols:
            protocol_config = production_registry.get_protocol(protocol_id)
            if not protocol_config or not protocol_config.is_enabled:
                continue
            
            try:
                if protocol_id == "uniswap_v3":
                    # Initialize Uniswap V3 state updater for Ethereum
                    web3 = await self.blockchain_provider.get_web3("ethereum")
                    if web3:
                        config = StateUpdateConfig(
                            cache_pool_states=True,
                            enable_price_impact_calculation=True
                        )
                        
                        adapter = UniswapV3StateUpdater(
                            provider=self.blockchain_provider,
                            chain_name="ethereum",
                            config=config
                        )
                        
                        # Initialize the adapter with quoter address
                        quoter_address = production_registry.get_contract_address("uniswap_v3", "ethereum", "quoter")
                        if quoter_address:
                            await adapter.initialize(quoter_address)
                            self.protocol_adapters[protocol_id] = adapter
                            logger.info(f"Initialized {protocol_id} adapter with quoter {quoter_address}")
                        else:
                            logger.error(f"Quoter address not found for {protocol_id}")
                    else:
                        logger.error(f"Web3 not available for {protocol_id} on ethereum")
                
            except Exception as e:
                logger.error(f"Failed to initialize {protocol_id} adapter: {e}")
    
    async def _discover_production_edges(self) -> None:
        """Discover real edges from production protocols."""
        logger.info("Discovering production edges...")
        
        # Get enabled protocols that support edge discovery
        enabled_protocols = production_registry.get_enabled_protocols()
        
        discovered_count = 0
        
        for protocol in enabled_protocols:
            if protocol.category != ProtocolCategory.DEX_SPOT:
                continue  # Focus on DEX protocols for now
            
            try:
                # Discover edges for each supported chain
                for chain_name in protocol.supported_chains:
                    if chain_name == "ethereum":  # Start with Ethereum mainnet
                        edges = await self._discover_protocol_edges(protocol.protocol_id, chain_name)
                        discovered_count += len(edges)
                        
                        logger.info(f"Discovered {len(edges)} edges for {protocol.name} on {chain_name}")
                
            except Exception as e:
                logger.error(f"Failed to discover edges for {protocol.protocol_id}: {e}")
        
        self.stats["edges_discovered"] = discovered_count
        logger.info(f"Total edges discovered: {discovered_count}")
    
    async def _discover_protocol_edges(self, protocol_id: str, chain_name: str) -> List[YieldGraphEdge]:
        """Discover edges for a specific protocol and chain."""
        edges = []
        
        if protocol_id == "uniswap_v3" and chain_name == "ethereum":
            # Discover high-volume Uniswap V3 pools
            edges = await self._discover_uniswap_v3_edges()
        
        # Register discovered edges
        for edge in edges:
            await self._register_edge(edge, protocol_id, chain_name)
        
        return edges
    
    async def _discover_uniswap_v3_edges(self) -> List[YieldGraphEdge]:
        """Discover high-volume Uniswap V3 edges."""
        edges = []
        
        # Define high-priority trading pairs with real pool addresses
        high_priority_pools = [
            {
                "pool_address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
                "token0": "0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                "token1": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "fee": 500,  # 0.05%
                "symbol": "USDC/WETH"
            },
            {
                "pool_address": "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",
                "token0": "0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                "token1": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "fee": 3000,  # 0.3%
                "symbol": "USDC/WETH"
            },
            {
                "pool_address": "0x11b815efB8f581194ae79006d24E0d814B7697F6",
                "token0": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "token1": "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "fee": 3000,  # 0.3%
                "symbol": "WETH/USDT"
            },
            {
                "pool_address": "0x60594a405d53811d3BC4766596EFD80fd545A270",
                "token0": "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
                "token1": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "fee": 3000,  # 0.3%
                "symbol": "DAI/WETH"
            }
        ]
        
        for pool_info in high_priority_pools:
            try:
                # Create bidirectional edges for each pool
                
                # Edge 1: token0 -> token1
                edge_id_0_1 = f"uniswap_v3_{pool_info['pool_address'].lower()}_{pool_info['token0'].lower()}_{pool_info['token1'].lower()}"
                edge_0_1 = YieldGraphEdge(
                    edge_id=edge_id_0_1,
                    source_asset_id=f"ETH_MAINNET_{self._get_token_symbol(pool_info['token0'])}",
                    target_asset_id=f"ETH_MAINNET_{self._get_token_symbol(pool_info['token1'])}",
                    edge_type=EdgeType.TRADE,
                    protocol_name="uniswap_v3",
                    chain_name="ethereum"
                )
                edges.append(edge_0_1)
                
                # Edge 2: token1 -> token0
                edge_id_1_0 = f"uniswap_v3_{pool_info['pool_address'].lower()}_{pool_info['token1'].lower()}_{pool_info['token0'].lower()}"
                edge_1_0 = YieldGraphEdge(
                    edge_id=edge_id_1_0,
                    source_asset_id=f"ETH_MAINNET_{self._get_token_symbol(pool_info['token1'])}",
                    target_asset_id=f"ETH_MAINNET_{self._get_token_symbol(pool_info['token0'])}",
                    edge_type=EdgeType.TRADE,
                    protocol_name="uniswap_v3",
                    chain_name="ethereum"
                )
                edges.append(edge_1_0)
                
            except Exception as e:
                logger.error(f"Failed to create edges for pool {pool_info['pool_address']}: {e}")
        
        logger.info(f"Created {len(edges)} Uniswap V3 edges from {len(high_priority_pools)} pools")
        return edges
    
    def _get_token_symbol(self, token_address: str) -> str:
        """Get token symbol from address."""
        token_map = {
            "0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b": "USDC",
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "WETH",
            "0xdAC17F958D2ee523a2206206994597C13D831ec7": "USDT",
            "0x6B175474E89094C44Da98b954EedeAC495271d0F": "DAI",
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": "WBTC"
        }
        return token_map.get(token_address, f"TOKEN_{token_address[-6:].upper()}")
    
    async def _register_edge(self, edge: YieldGraphEdge, protocol_id: str, chain_name: str) -> None:
        """Register an edge for state updates."""
        # Determine priority based on expected volume/importance
        priority = self._determine_edge_priority(edge, protocol_id)
        
        # Set update interval based on priority
        update_intervals = {
            EdgePriority.CRITICAL: 12,   # ~1 block
            EdgePriority.HIGH: 36,       # ~3 blocks  
            EdgePriority.MEDIUM: 120,    # ~10 blocks
            EdgePriority.LOW: 600        # ~50 blocks
        }
        
        config = EdgeUpdateConfig(
            edge_id=edge.edge_id,
            protocol_id=protocol_id,
            chain_name=chain_name,
            priority=priority,
            update_interval_seconds=update_intervals[priority],
            pool_metadata=self._extract_pool_metadata(edge)
        )
        
        self.edge_configs[edge.edge_id] = config
        self.active_edges[edge.edge_id] = edge
        
        # Initialize with empty state (will be updated by scheduler)
        self.edge_states[edge.edge_id] = EdgeState()
        
        logger.debug(f"Registered edge {edge.edge_id} with {priority.value} priority")
    
    def _determine_edge_priority(self, edge: YieldGraphEdge, protocol_id: str) -> EdgePriority:
        """Determine update priority for an edge."""
        # High priority for major stablecoin and ETH pairs
        if any(token in edge.source_asset_id or token in edge.target_asset_id 
               for token in ["WETH", "USDC", "USDT"]):
            return EdgePriority.HIGH
        
        # Medium priority for other major tokens
        if any(token in edge.source_asset_id or token in edge.target_asset_id 
               for token in ["DAI", "WBTC"]):
            return EdgePriority.MEDIUM
        
        return EdgePriority.LOW
    
    def _extract_pool_metadata(self, edge: YieldGraphEdge) -> Dict[str, Any]:
        """Extract pool metadata from edge for state updates."""
        if edge.protocol_name == "uniswap_v3":
            # Extract pool address from edge_id
            parts = edge.edge_id.split("_")
            if len(parts) >= 3:
                pool_address = parts[2]  # Should be the pool address
                return {
                    "pool_address": pool_address,
                    "protocol": "uniswap_v3",
                    "source_token": edge.source_asset_id.split("_")[-1],
                    "target_token": edge.target_asset_id.split("_")[-1]
                }
        
        return {}
    
    async def _run_update_scheduler(self) -> None:
        """Run the edge state update scheduler."""
        self.update_scheduler_running = True
        logger.info("Started edge state update scheduler")
        
        while self.update_scheduler_running:
            try:
                # Get edges that need updates
                edges_to_update = self._get_edges_needing_update()
                
                if edges_to_update:
                    # Update edges in parallel (respecting concurrency limit)
                    tasks = [
                        self._update_edge_state(edge_id) 
                        for edge_id in edges_to_update[:self.max_concurrent_updates]
                    ]
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update statistics
                self.stats["active_edge_count"] = len([c for c in self.edge_configs.values() if c.is_active])
                
                # Sleep before next update cycle
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in update scheduler: {e}")
                await asyncio.sleep(10)  # Longer sleep on error
    
    def _get_edges_needing_update(self) -> List[str]:
        """Get list of edge IDs that need state updates."""
        current_time = time.time()
        edges_to_update = []
        
        for edge_id, config in self.edge_configs.items():
            if not config.is_active:
                continue
            
            # Check if enough time has passed since last update
            if (config.last_updated is None or 
                current_time - config.last_updated >= config.update_interval_seconds):
                edges_to_update.append(edge_id)
        
        # Sort by priority and staleness
        edges_to_update.sort(key=lambda eid: (
            self.edge_configs[eid].priority.value,
            self.edge_configs[eid].last_updated or 0
        ))
        
        return edges_to_update
    
    async def _update_edge_state(self, edge_id: str) -> bool:
        """Update state for a specific edge."""
        async with self.update_semaphore:
            config = self.edge_configs.get(edge_id)
            edge = self.active_edges.get(edge_id)
            
            if not config or not edge:
                return False
            
            start_time = time.time()
            
            try:
                # Update state based on protocol
                new_state = await self._collect_edge_state(edge, config)
                
                if new_state:
                    # Update stored state
                    self.edge_states[edge_id] = new_state
                    edge.state = new_state
                    
                    # Update config
                    config.last_updated = time.time()
                    config.consecutive_failures = 0
                    
                    # Update statistics
                    self.stats["successful_updates"] += 1
                    update_time = (time.time() - start_time) * 1000
                    self._update_average_time(update_time)
                    
                    logger.debug(f"Updated {edge_id}: rate={new_state.conversion_rate}, liq=${new_state.liquidity_usd:,.0f}")
                    return True
                else:
                    raise Exception("Failed to collect edge state")
                
            except Exception as e:
                config.consecutive_failures += 1
                config.last_updated = time.time()  # Still update to avoid spam
                
                # Disable edge if too many failures
                if config.consecutive_failures >= 5:
                    config.is_active = False
                    logger.warning(f"Disabled edge {edge_id} after {config.consecutive_failures} failures")
                
                self.stats["failed_updates"] += 1
                logger.error(f"Failed to update {edge_id}: {e}")
                return False
            
            finally:
                self.stats["total_updates"] += 1
    
    async def _collect_edge_state(self, edge: YieldGraphEdge, config: EdgeUpdateConfig) -> Optional[EdgeState]:
        """Collect real state data for an edge."""
        if edge.protocol_name == "uniswap_v3":
            adapter = self.protocol_adapters.get("uniswap_v3")
            if not adapter:
                return None
            
            try:
                # Get pool metadata
                metadata = config.pool_metadata or {}
                pool_address = metadata.get("pool_address")
                
                if not pool_address:
                    return None
                
                # Extract token addresses from the pool
                # For real implementation, we'd query the pool contract
                # For now, use mappings based on known pools
                token_info = self._get_pool_token_info(pool_address)
                if not token_info:
                    return None
                
                # Update edge state using the adapter
                metadata_with_tokens = {
                    **metadata,
                    **token_info
                }
                updated_state = await adapter.update_edge_state(edge, metadata_with_tokens)
                
                return updated_state
                
            except Exception as e:
                logger.error(f"Failed to collect Uniswap V3 state for {edge.edge_id}: {e}")
                logger.debug(f"Edge metadata: {metadata}")
                logger.debug(f"Pool address: {pool_address}")
                logger.debug(f"Token info: {token_info if 'token_info' in locals() else 'Not found'}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return None
        
        return None
    
    def _get_pool_token_info(self, pool_address: str) -> Optional[Dict[str, Any]]:
        """Get token info for a pool address."""
        # Normalize address to lowercase for lookup
        normalized_address = pool_address.lower()
        
        pool_info_map = {
            "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640": {
                "token0_address": "0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                "token1_address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "fee_tier": 500,
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
                "token0_decimals": 6,
                "token1_decimals": 18
            },
            "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8": {
                "token0_address": "0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                "token1_address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "fee_tier": 3000,
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
                "token0_decimals": 6,
                "token1_decimals": 18
            },
            "0x11b815efb8f581194ae79006d24e0d814b7697f6": {
                "token0_address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "token1_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "fee_tier": 3000,
                "token0_symbol": "WETH",
                "token1_symbol": "USDT",
                "token0_decimals": 18,
                "token1_decimals": 6
            },
            "0x60594a405d53811d3bc4766596efd80fd545a270": {
                "token0_address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
                "token1_address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "fee_tier": 3000,
                "token0_symbol": "DAI",
                "token1_symbol": "WETH",
                "token0_decimals": 18,
                "token1_decimals": 18
            }
        }
        
        return pool_info_map.get(normalized_address)
    
    def _update_average_time(self, update_time_ms: float) -> None:
        """Update average update time statistic."""
        if self.stats["successful_updates"] == 1:
            self.stats["average_update_time_ms"] = update_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["average_update_time_ms"] = (
                alpha * update_time_ms + 
                (1 - alpha) * self.stats["average_update_time_ms"]
            )
    
    async def get_edge_state(self, edge_id: str) -> Optional[EdgeState]:
        """Get current state for an edge."""
        return self.edge_states.get(edge_id)
    
    async def get_all_active_edges(self) -> Dict[str, YieldGraphEdge]:
        """Get all active edges."""
        return {
            eid: edge for eid, edge in self.active_edges.items()
            if self.edge_configs[eid].is_active
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "pipeline_uptime_seconds": time.time() - (self.stats.get("start_time", time.time())),
            "edge_priorities": {
                priority.value: len([c for c in self.edge_configs.values() if c.priority == priority])
                for priority in EdgePriority
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the pipeline."""
        logger.info("Shutting down real edge state pipeline...")
        self.update_scheduler_running = False
        
        # Close protocol adapters
        for adapter in self.protocol_adapters.values():
            if hasattr(adapter, 'close'):
                await adapter.close()
        
        logger.info("Pipeline shutdown complete")