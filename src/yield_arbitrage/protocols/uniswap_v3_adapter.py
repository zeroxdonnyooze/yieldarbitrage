"""Uniswap V3 protocol adapter implementation."""
import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from yield_arbitrage.protocols.base_adapter import ProtocolAdapter, ProtocolError, NetworkError
from yield_arbitrage.protocols.abis.uniswap_v3 import (
    UNISWAP_V3_FACTORY_ABI,
    UNISWAP_V3_QUOTER_ABI,
    UNISWAP_V3_POOL_ABI,
    ERC20_ABI
)
from yield_arbitrage.protocols.contracts import UNISWAP_V3_CONTRACTS, WELL_KNOWN_TOKENS
from yield_arbitrage.protocols.token_filter import TokenFilter, TokenCriteria, default_token_filter
from yield_arbitrage.protocols.uniswap_v3_pool_discovery import UniswapV3PoolDiscovery, PoolDiscoveryConfig
from yield_arbitrage.protocols.uniswap_v3_state_updater import UniswapV3StateUpdater, StateUpdateConfig
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType, EdgeConstraints

logger = logging.getLogger(__name__)


class UniswapV3Adapter(ProtocolAdapter):
    """Uniswap V3 protocol adapter for discovering and managing trading edges."""
    
    # Standard Uniswap V3 fee tiers (in hundredths of a bip, so 3000 = 0.3%)
    STANDARD_FEE_TIERS = [100, 500, 3000, 10000]  # 0.01%, 0.05%, 0.3%, 1%
    
    def __init__(self, chain_name: str, provider, token_filter: Optional[TokenFilter] = None, 
                 discovery_config: Optional[PoolDiscoveryConfig] = None,
                 state_update_config: Optional[StateUpdateConfig] = None):
        super().__init__(chain_name, provider)
        self.protocol_name = "uniswapv3"
        self.supported_edge_types = [EdgeType.TRADE]
        
        # Contract instances
        self.factory_contract = None
        self.quoter_contract = None
        
        # Token filtering
        self.token_filter = token_filter or default_token_filter
        
        # Advanced pool discovery
        self.pool_discovery = UniswapV3PoolDiscovery(
            provider, 
            chain_name, 
            discovery_config or PoolDiscoveryConfig()
        )
        
        # Advanced state updater
        self.state_updater = UniswapV3StateUpdater(
            provider,
            chain_name,
            state_update_config or StateUpdateConfig()
        )
        
        # Cache for discovered pools and tokens
        self.discovered_pools: Set[str] = set()
        self.token_decimals_cache: Dict[str, int] = {}
        
        # Pool metadata cache (edge_id -> metadata)
        self.pool_metadata_cache: Dict[str, Dict] = {}
        
        # Discovery mode flags
        self.enable_live_discovery = True
        self.enable_event_scanning = True
        
        # Rate limiting for pool discovery
        self.discovery_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent operations
    
    async def _protocol_specific_init(self) -> bool:
        """Initialize Uniswap V3 specific components."""
        try:
            # Get contract addresses for this chain
            contracts = UNISWAP_V3_CONTRACTS.get(self.chain_name)
            if not contracts:
                logger.error(f"No Uniswap V3 contracts configured for {self.chain_name}")
                return False
            
            # Get Web3 instance
            web3 = await self.provider.get_web3(self.chain_name)
            if not web3:
                logger.error(f"Failed to get Web3 instance for {self.chain_name}")
                return False
            
            # Initialize contract instances
            self.factory_contract = web3.eth.contract(
                address=web3.to_checksum_address(contracts["factory"]),
                abi=UNISWAP_V3_FACTORY_ABI
            )
            
            self.quoter_contract = web3.eth.contract(
                address=web3.to_checksum_address(contracts["quoter"]),
                abi=UNISWAP_V3_QUOTER_ABI
            )
            
            # Initialize token filter if needed
            await self.token_filter.initialize()
            
            # Initialize pool discovery system
            discovery_success = await self.pool_discovery.initialize()
            if not discovery_success:
                logger.warning("Pool discovery system failed to initialize")
            
            # Initialize state updater system
            state_updater_success = await self.state_updater.initialize(contracts["quoter"])
            if not state_updater_success:
                logger.warning("State updater system failed to initialize")
            
            logger.info(f"UniswapV3Adapter initialized for {self.chain_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize UniswapV3Adapter for {self.chain_name}: {e}")
            return False
    
    async def discover_edges(self) -> List[YieldGraphEdge]:
        """Discover trading edges from Uniswap V3 pools using advanced discovery."""
        try:
            # Get list of tokens to consider
            token_addresses = await self._get_filtered_tokens()
            if not token_addresses:
                logger.warning(f"No tokens found for discovery on {self.chain_name}")
                return []
            
            logger.info(f"Discovering Uniswap V3 edges for {len(token_addresses)} tokens on {self.chain_name}")
            
            edges = []
            
            # Use live pool discovery if enabled
            if self.enable_live_discovery:
                discovered_pools = await self.pool_discovery.discover_pools_by_tokens(token_addresses)
                logger.info(f"Live discovery found {len(discovered_pools)} pools")
                
                # Convert discovered pools to edges
                for pool_info in discovered_pools:
                    pool_edges = await self._create_edges_from_pool_info(pool_info)
                    edges.extend(pool_edges)
            
            # Use event-based discovery if enabled and no live discovery
            elif self.enable_event_scanning:
                event_pools = await self.pool_discovery.discover_pools_by_events()
                logger.info(f"Event discovery found {len(event_pools)} pools")
                
                # Convert event pools to edges
                for pool_info in event_pools:
                    pool_edges = await self._create_edges_from_pool_info(pool_info)
                    edges.extend(pool_edges)
            
            # Fallback to legacy discovery method
            else:
                logger.info("Using legacy discovery method")
                edges = await self._legacy_discover_edges(token_addresses)
            
            # Record discovery success
            self._record_discovery_success(len(edges))
            
            logger.info(f"Discovered {len(edges)} Uniswap V3 edges on {self.chain_name}")
            return edges
            
        except Exception as e:
            self._record_discovery_error()
            raise ProtocolError(f"Edge discovery failed: {e}", self.protocol_name, self.chain_name)
    
    async def _get_filtered_tokens(self) -> List[str]:
        """Get list of tokens that pass filtering criteria."""
        try:
            # Start with well-known tokens for this chain
            chain_tokens = WELL_KNOWN_TOKENS.get(self.chain_name, {})
            token_addresses = list(chain_tokens.values())
            
            # Filter tokens based on criteria
            if token_addresses:
                filtered_tokens = await self.token_filter.filter_tokens(
                    token_addresses, 
                    chain=self.chain_name
                )
                return filtered_tokens
            
            return []
            
        except Exception as e:
            logger.error(f"Error filtering tokens: {e}")
            return []
    
    async def _create_edges_from_pool_info(self, pool_info) -> List[YieldGraphEdge]:
        """Create bidirectional trading edges from pool discovery info."""
        try:
            # Skip pools with insufficient liquidity
            if pool_info.tvl_usd and pool_info.tvl_usd < 10000:  # $10k minimum
                return []
            
            edges = []
            
            # Create asset IDs
            token0_asset_id = f"{self.chain_name}_TOKEN_{pool_info.token0_address}"
            token1_asset_id = f"{self.chain_name}_TOKEN_{pool_info.token1_address}"
            
            # Token0 -> Token1 edge
            edge_id_0_to_1 = f"{self.chain_name}_UNISWAPV3_TRADE_{pool_info.token0_address}_{pool_info.token1_address}_{pool_info.fee_tier}"
            
            # Store metadata in our cache
            metadata_0_to_1 = {
                "pool_address": pool_info.pool_address,
                "fee_tier": pool_info.fee_tier,
                "fee_percentage": pool_info.fee_tier / 1_000_000,
                "token0_address": pool_info.token0_address,
                "token1_address": pool_info.token1_address,
                "token0_symbol": pool_info.token0_symbol,
                "token1_symbol": pool_info.token1_symbol,
                "token0_decimals": pool_info.token0_decimals,
                "token1_decimals": pool_info.token1_decimals,
                "created_block": pool_info.created_block,
                "created_timestamp": pool_info.created_timestamp.isoformat() if pool_info.created_timestamp else None
            }
            self.pool_metadata_cache[edge_id_0_to_1] = metadata_0_to_1
            
            # Calculate constraints based on pool liquidity
            max_trade_amount = float(pool_info.liquidity) / 4 if pool_info.liquidity > 0 else 1000000
            
            edges.append(YieldGraphEdge(
                edge_id=edge_id_0_to_1,
                edge_type=EdgeType.TRADE,
                source_asset_id=token0_asset_id,
                target_asset_id=token1_asset_id,
                protocol_name=self.protocol_name,
                chain_name=self.chain_name,
                constraints=EdgeConstraints(
                    min_input_amount=1.0,
                    max_input_amount=max_trade_amount
                ),
                state=EdgeState(
                    liquidity_usd=pool_info.tvl_usd,
                    gas_cost_usd=self._estimate_gas_cost(),
                    last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                    confidence_score=0.95 if pool_info.is_active else 0.5
                )
            ))
            
            # Token1 -> Token0 edge
            edge_id_1_to_0 = f"{self.chain_name}_UNISWAPV3_TRADE_{pool_info.token1_address}_{pool_info.token0_address}_{pool_info.fee_tier}"
            
            metadata_1_to_0 = {
                "pool_address": pool_info.pool_address,
                "fee_tier": pool_info.fee_tier,
                "fee_percentage": pool_info.fee_tier / 1_000_000,
                "token0_address": pool_info.token0_address,
                "token1_address": pool_info.token1_address,
                "token0_symbol": pool_info.token0_symbol,
                "token1_symbol": pool_info.token1_symbol,
                "token0_decimals": pool_info.token0_decimals,
                "token1_decimals": pool_info.token1_decimals,
                "created_block": pool_info.created_block,
                "created_timestamp": pool_info.created_timestamp.isoformat() if pool_info.created_timestamp else None
            }
            self.pool_metadata_cache[edge_id_1_to_0] = metadata_1_to_0
            
            edges.append(YieldGraphEdge(
                edge_id=edge_id_1_to_0,
                edge_type=EdgeType.TRADE,
                source_asset_id=token1_asset_id,
                target_asset_id=token0_asset_id,
                protocol_name=self.protocol_name,
                chain_name=self.chain_name,
                constraints=EdgeConstraints(
                    min_input_amount=1.0,
                    max_input_amount=max_trade_amount
                ),
                state=EdgeState(
                    liquidity_usd=pool_info.tvl_usd,
                    gas_cost_usd=self._estimate_gas_cost(),
                    last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                    confidence_score=0.95 if pool_info.is_active else 0.5
                )
            ))
            
            # Mark pool as discovered
            self.discovered_pools.add(pool_info.pool_address)
            
            return edges
            
        except Exception as e:
            logger.warning(f"Error creating edges from pool info: {e}")
            return []
    
    async def _legacy_discover_edges(self, token_addresses: List[str]) -> List[YieldGraphEdge]:
        """Legacy discovery method using simple token pair enumeration."""
        edges = []
        tasks = []
        
        # Create tasks for all token pairs and fee tiers
        for i, token0 in enumerate(token_addresses):
            for token1 in token_addresses[i + 1:]:  # Avoid duplicates
                for fee_tier in self.STANDARD_FEE_TIERS:
                    tasks.append(self._discover_pool_edges(token0, token1, fee_tier))
        
        # Process tasks in batches to avoid overwhelming the RPC
        batch_size = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Pool discovery error: {result}")
                    continue
                
                if result:  # result is a list of edges
                    edges.extend(result)
            
            # Small delay between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.1)
        
        return edges
    
    async def _discover_pool_edges(self, token0: str, token1: str, fee_tier: int) -> List[YieldGraphEdge]:
        """Discover edges for a specific token pair and fee tier."""
        async with self.discovery_semaphore:
            try:
                # Get pool address
                pool_address = await self._get_pool_address(token0, token1, fee_tier)
                if not pool_address or pool_address == "0x0000000000000000000000000000000000000000":
                    return []
                
                # Skip if we've already discovered this pool
                if pool_address.lower() in self.discovered_pools:
                    return []
                
                # Get pool state to ensure it's active
                pool_state = await self._get_basic_pool_state(pool_address)
                if not pool_state or pool_state.get("liquidity", 0) == 0:
                    return []
                
                # Mark pool as discovered
                self.discovered_pools.add(pool_address.lower())
                
                # Create asset IDs
                token0_asset_id = f"{self.chain_name}_TOKEN_{token0.lower()}"
                token1_asset_id = f"{self.chain_name}_TOKEN_{token1.lower()}"
                
                # Create bidirectional trading edges
                edges = []
                
                # Token0 -> Token1 edge
                edge_id_0_to_1 = f"{self.chain_name}_UNISWAPV3_TRADE_{token0.lower()}_{token1.lower()}_{fee_tier}"
                
                # Store metadata separately in our cache
                metadata_0_to_1 = {
                    "pool_address": pool_address,
                    "fee_tier": fee_tier,
                    "fee_percentage": fee_tier / 1_000_000,  # Convert to percentage
                    "token0_address": token0,
                    "token1_address": token1
                }
                self.pool_metadata_cache[edge_id_0_to_1] = metadata_0_to_1
                
                edges.append(YieldGraphEdge(
                    edge_id=edge_id_0_to_1,
                    edge_type=EdgeType.TRADE,
                    source_asset_id=token0_asset_id,
                    target_asset_id=token1_asset_id,
                    protocol_name=self.protocol_name,
                    chain_name=self.chain_name,
                    constraints=EdgeConstraints(
                        min_input_amount=1.0,  # Minimum 1 unit of source token
                        max_input_amount=float(pool_state.get("liquidity", 0)) / 4  # Quarter of liquidity
                    ),
                    state=EdgeState(
                        liquidity_usd=pool_state.get("tvl_usd"),
                        gas_cost_usd=self._estimate_gas_cost(),
                        last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                        confidence_score=0.9
                    )
                ))
                
                # Token1 -> Token0 edge
                edge_id_1_to_0 = f"{self.chain_name}_UNISWAPV3_TRADE_{token1.lower()}_{token0.lower()}_{fee_tier}"
                
                # Store metadata separately in our cache
                metadata_1_to_0 = {
                    "pool_address": pool_address,
                    "fee_tier": fee_tier,
                    "fee_percentage": fee_tier / 1_000_000,
                    "token0_address": token0,
                    "token1_address": token1
                }
                self.pool_metadata_cache[edge_id_1_to_0] = metadata_1_to_0
                
                edges.append(YieldGraphEdge(
                    edge_id=edge_id_1_to_0,
                    edge_type=EdgeType.TRADE,
                    source_asset_id=token1_asset_id,
                    target_asset_id=token0_asset_id,
                    protocol_name=self.protocol_name,
                    chain_name=self.chain_name,
                    constraints=EdgeConstraints(
                        min_input_amount=1.0,
                        max_input_amount=float(pool_state.get("liquidity", 0)) / 4
                    ),
                    state=EdgeState(
                        liquidity_usd=pool_state.get("tvl_usd"),
                        gas_cost_usd=self._estimate_gas_cost(),
                        last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                        confidence_score=0.9
                    )
                ))
                
                return edges
                
            except Exception as e:
                logger.warning(f"Error discovering pool for {token0}/{token1} fee {fee_tier}: {e}")
                return []
    
    async def _get_pool_address(self, token0: str, token1: str, fee_tier: int) -> Optional[str]:
        """Get pool address for token pair and fee tier."""
        try:
            web3 = await self.provider.get_web3(self.chain_name)
            if not web3:
                return None
            
            # Ensure token addresses are checksummed and properly ordered
            token0_addr = web3.to_checksum_address(token0)
            token1_addr = web3.to_checksum_address(token1)
            
            # Call factory contract to get pool address
            pool_address = await self.factory_contract.functions.getPool(
                token0_addr, token1_addr, fee_tier
            ).call()
            
            return pool_address
            
        except Exception as e:
            logger.debug(f"Error getting pool address for {token0}/{token1}: {e}")
            return None
    
    async def _get_basic_pool_state(self, pool_address: str) -> Optional[Dict]:
        """Get basic pool state information."""
        try:
            web3 = await self.provider.get_web3(self.chain_name)
            if not web3:
                return None
            
            pool_contract = web3.eth.contract(
                address=web3.to_checksum_address(pool_address),
                abi=UNISWAP_V3_POOL_ABI
            )
            
            # Get liquidity and slot0 data
            liquidity = await pool_contract.functions.liquidity().call()
            slot0 = await pool_contract.functions.slot0().call()
            
            return {
                "liquidity": liquidity,
                "sqrt_price_x96": slot0[0],
                "tick": slot0[1],
                "tvl_usd": None  # Would need price data to calculate
            }
            
        except Exception as e:
            logger.debug(f"Error getting pool state for {pool_address}: {e}")
            return None
    
    async def update_edge_state(self, edge: YieldGraphEdge) -> EdgeState:
        """Update the state of a trading edge with comprehensive market data using advanced state updater."""
        # Get metadata from our cache - this should always raise if missing
        metadata = self.pool_metadata_cache.get(edge.edge_id)
        if not metadata or "pool_address" not in metadata:
            raise ProtocolError(f"Missing pool address metadata for edge {edge.edge_id}")
        
        try:
            # Use the advanced state updater for comprehensive edge state updates
            updated_state = await self.state_updater.update_edge_state(edge, metadata)
            
            # Record successful update
            self._record_update_success(0.3)  # ~300ms typical update time with advanced features
            
            return updated_state
            
        except Exception as e:
            self._record_update_error()
            logger.error(f"Error updating edge state for {edge.edge_id}: {e}")
            
            # Return existing state with reduced confidence
            edge.state.confidence_score = max(0.1, edge.state.confidence_score * 0.5)
            edge.state.last_updated_timestamp = datetime.now(timezone.utc).timestamp()
            return edge.state
    
    async def _get_detailed_pool_state(self, pool_address: str) -> Optional[Dict]:
        """Get detailed pool state including TVL estimation."""
        try:
            basic_state = await self._get_basic_pool_state(pool_address)
            if not basic_state:
                return None
            
            # TODO: Calculate TVL in USD using token prices
            # For now, return basic state with placeholder TVL
            basic_state["tvl_usd"] = float(basic_state["liquidity"]) * 0.001  # Rough estimate
            
            return basic_state
            
        except Exception as e:
            logger.debug(f"Error getting detailed pool state for {pool_address}: {e}")
            return None
    
    async def _get_conversion_rate(self, token_in: str, token_out: str, fee_tier: int, amount_in: int) -> Optional[float]:
        """Get conversion rate between tokens using Quoter contract."""
        try:
            web3 = await self.provider.get_web3(self.chain_name)
            if not web3:
                return None
            
            # Get token decimals for proper scaling
            decimals_in = await self._get_token_decimals(token_in)
            decimals_out = await self._get_token_decimals(token_out)
            
            if decimals_in is None or decimals_out is None:
                return None
            
            # Scale amount by token decimals
            scaled_amount_in = amount_in * (10 ** decimals_in)
            
            # Call quoter
            quote_result = await self.quoter_contract.functions.quoteExactInputSingle(
                web3.to_checksum_address(token_in),
                web3.to_checksum_address(token_out),
                fee_tier,
                scaled_amount_in,
                0  # sqrtPriceLimitX96 (0 = no limit)
            ).call()
            
            # Extract amount out from result
            amount_out = quote_result if isinstance(quote_result, int) else quote_result[0]
            
            # Calculate rate with proper decimal scaling
            scaled_amount_out = amount_out / (10 ** decimals_out)
            scaled_amount_in_normalized = amount_in  # amount_in is already in normalized units
            
            if scaled_amount_in_normalized > 0:
                return scaled_amount_out / scaled_amount_in_normalized
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting conversion rate for {token_in}/{token_out}: {e}")
            return None
    
    async def _get_token_decimals(self, token_address: str) -> Optional[int]:
        """Get token decimals with caching."""
        if token_address in self.token_decimals_cache:
            return self.token_decimals_cache[token_address]
        
        try:
            web3 = await self.provider.get_web3(self.chain_name)
            if not web3:
                return None
            
            token_contract = web3.eth.contract(
                address=web3.to_checksum_address(token_address),
                abi=ERC20_ABI
            )
            
            decimals = await token_contract.functions.decimals().call()
            self.token_decimals_cache[token_address] = decimals
            
            return decimals
            
        except Exception as e:
            logger.debug(f"Error getting decimals for token {token_address}: {e}")
            return 18  # Default to 18 decimals
    
    def _estimate_gas_cost(self) -> float:
        """Estimate gas cost for a Uniswap V3 swap in USD."""
        # Rough estimates for different chains
        gas_estimates = {
            "ethereum": 15.0,     # Higher gas costs
            "arbitrum": 2.0,      # Lower gas costs
            "base": 1.5,          # Very low gas costs
            "polygon": 0.5,       # Very low gas costs
            "sonic": 0.1,         # Assumed very low
            "berachain": 0.1      # Assumed very low
        }
        
        return gas_estimates.get(self.chain_name, 5.0)
    
    async def refresh_pool_data(self, pool_addresses: Optional[List[str]] = None) -> int:
        """Refresh data for existing pools."""
        if pool_addresses is None:
            pool_addresses = list(self.discovered_pools)
        
        refreshed_pools = await self.pool_discovery.refresh_pool_data(pool_addresses)
        
        # Update edges with refreshed data
        updated_count = 0
        for pool_address, pool_info in refreshed_pools.items():
            # Find edges for this pool and update their metadata
            for edge_id, metadata in self.pool_metadata_cache.items():
                if metadata.get("pool_address") == pool_address:
                    # Update metadata with fresh data
                    metadata.update({
                        "token0_symbol": pool_info.token0_symbol,
                        "token1_symbol": pool_info.token1_symbol,
                        "token0_decimals": pool_info.token0_decimals,
                        "token1_decimals": pool_info.token1_decimals
                    })
                    updated_count += 1
        
        logger.info(f"Refreshed data for {len(refreshed_pools)} pools, updated {updated_count} edges")
        return updated_count
    
    async def discover_new_pools_by_events(self, from_block: Optional[int] = None) -> List[YieldGraphEdge]:
        """Discover new pools using event scanning and convert to edges."""
        new_pools = await self.pool_discovery.discover_pools_by_events(from_block)
        
        all_edges = []
        for pool_info in new_pools:
            edges = await self._create_edges_from_pool_info(pool_info)
            all_edges.extend(edges)
        
        logger.info(f"Discovered {len(new_pools)} new pools via events, created {len(all_edges)} edges")
        return all_edges
    
    def get_pool_discovery_stats(self) -> Dict:
        """Get comprehensive discovery and state update statistics."""
        discovery_stats = self.pool_discovery.get_discovery_stats()
        adapter_stats = self.get_discovery_stats()
        state_updater_stats = self.state_updater.get_update_stats()
        
        return {
            "adapter_stats": adapter_stats,
            "pool_discovery_stats": discovery_stats,
            "state_updater_stats": state_updater_stats,
            "total_pools_discovered": len(self.discovered_pools),
            "metadata_cache_size": len(self.pool_metadata_cache),
            "discovery_modes": {
                "live_discovery": self.enable_live_discovery,
                "event_scanning": self.enable_event_scanning
            }
        }
    
    def get_pools_by_token(self, token_address: str) -> List[Dict]:
        """Get all pools containing a specific token."""
        token_lower = token_address.lower()
        matching_pools = []
        
        for edge_id, metadata in self.pool_metadata_cache.items():
            if (metadata.get("token0_address") == token_lower or 
                metadata.get("token1_address") == token_lower):
                matching_pools.append(metadata)
        
        return matching_pools
    
    def get_pools_by_fee_tier(self, fee_tier: int) -> List[Dict]:
        """Get all pools with a specific fee tier."""
        matching_pools = []
        
        for edge_id, metadata in self.pool_metadata_cache.items():
            if metadata.get("fee_tier") == fee_tier:
                matching_pools.append(metadata)
        
        return matching_pools
    
    async def batch_update_edge_states(self, edges: List[YieldGraphEdge]) -> Dict[str, EdgeState]:
        """Update multiple edge states in batch for better performance."""
        if not edges:
            return {}
        
        # Prepare edges with metadata for batch processing
        edges_with_metadata = []
        for edge in edges:
            metadata = self.pool_metadata_cache.get(edge.edge_id)
            if metadata and "pool_address" in metadata:
                edges_with_metadata.append((edge, metadata))
            else:
                logger.warning(f"Missing metadata for edge {edge.edge_id}, skipping batch update")
        
        if not edges_with_metadata:
            logger.warning("No edges with valid metadata for batch update")
            return {}
        
        # Use state updater's batch processing
        updated_states = await self.state_updater.batch_update_edges(edges_with_metadata)
        
        logger.info(f"Batch updated {len(updated_states)}/{len(edges)} edge states")
        return updated_states
    
    def get_state_updater_stats(self) -> Dict:
        """Get state updater performance statistics."""
        return self.state_updater.get_update_stats()
    
    def cleanup_caches(self) -> None:
        """Clean up expired cache entries in both discovery and state updater systems."""
        self.state_updater.cleanup_cache()
        # Note: Pool discovery cache cleanup happens automatically during operations
        logger.info("Cache cleanup completed")
    
    def set_discovery_mode(self, live_discovery: bool = True, event_scanning: bool = True) -> None:
        """Configure discovery modes."""
        self.enable_live_discovery = live_discovery
        self.enable_event_scanning = event_scanning
        
        logger.info(f"Discovery modes updated: live={live_discovery}, events={event_scanning}")
    
    def get_supported_tokens(self) -> List[str]:
        """Get list of supported tokens for this chain."""
        chain_tokens = WELL_KNOWN_TOKENS.get(self.chain_name, {})
        return list(chain_tokens.values())
    
    def __str__(self) -> str:
        return f"UniswapV3Adapter({self.chain_name}, pools={len(self.discovered_pools)})"
    
    def __repr__(self) -> str:
        return (f"UniswapV3Adapter(protocol=uniswapv3, chain={self.chain_name}, "
                f"initialized={self.is_initialized}, pools={len(self.discovered_pools)})")