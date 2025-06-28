"""On-chain price oracle for real-time DeFi arbitrage execution."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from decimal import Decimal

from yield_arbitrage.execution.asset_oracle import AssetOracleBase, AssetPrice

logger = logging.getLogger(__name__)


@dataclass
class DEXPoolInfo:
    """Information about a DEX pool for price discovery."""
    pool_address: str
    dex_type: str  # "uniswap_v2", "uniswap_v3", "curve", "balancer"
    token0: str
    token1: str
    fee_tier: Optional[int] = None  # For V3 pools
    pool_type: Optional[str] = None  # For Curve/Balancer
    liquidity_weight: float = 1.0  # Weight for price aggregation


class OnChainPriceOracle(AssetOracleBase):
    """
    Production on-chain price oracle that gets real-time prices directly from DEX contracts.
    
    This is the primary oracle for DeFi arbitrage as it provides:
    - Real-time prices from actual DEX pools
    - No external API dependencies
    - Prices that match execution reality
    - Support for multiple DEX types
    """
    
    def __init__(self, blockchain_provider, redis_client=None):
        """
        Initialize on-chain price oracle.
        
        Args:
            blockchain_provider: BlockchainProvider instance
            redis_client: Optional Redis client for caching
        """
        self.blockchain_provider = blockchain_provider
        self.redis_client = redis_client
        
        # Known high-liquidity pools for price discovery
        self.price_discovery_pools = self._initialize_price_pools()
        
        # Cache for recent prices
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl_seconds = 30  # 30 second cache for on-chain prices
    
    async def initialize(self) -> bool:
        """Initialize the price oracle (compatibility method)."""
        logger.info("On-chain price oracle initialized")
        return True
    
    def _initialize_price_pools(self) -> Dict[str, List[DEXPoolInfo]]:
        """Initialize high-liquidity pools for price discovery."""
        return {
            # ETH price discovery (against stablecoins)
            "ETH_MAINNET_WETH": [
                # Uniswap V3 WETH/USDC pools (highest liquidity)
                DEXPoolInfo(
                    pool_address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
                    dex_type="uniswap_v3",
                    token0="0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                    token1="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    fee_tier=500,  # 0.05%
                    liquidity_weight=3.0  # Highest weight
                ),
                DEXPoolInfo(
                    pool_address="0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",
                    dex_type="uniswap_v3", 
                    token0="0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                    token1="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    fee_tier=3000,  # 0.3%
                    liquidity_weight=2.0
                ),
                # Uniswap V3 WETH/USDT
                DEXPoolInfo(
                    pool_address="0x11b815efB8f581194ae79006d24E0d814B7697F6",
                    dex_type="uniswap_v3",
                    token0="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    token1="0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                    fee_tier=3000,
                    liquidity_weight=1.5
                ),
            ],
            
            # WBTC price discovery (through ETH)
            "ETH_MAINNET_WBTC": [
                DEXPoolInfo(
                    pool_address="0xCBCdF9626bC03E24f779434178A73a0B4bad62eD",
                    dex_type="uniswap_v3",
                    token0="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
                    token1="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    fee_tier=3000,
                    liquidity_weight=2.0
                ),
            ],
            
            # Stablecoins (return ~$1 but check for depegs)
            "ETH_MAINNET_USDC": [],  # Base USD asset
            "ETH_MAINNET_USDT": [
                # USDT/USDC pools for depeg detection
                DEXPoolInfo(
                    pool_address="0x3416cF6C708Da44DB2624D63ea0AAef7113527C6",
                    dex_type="uniswap_v3",
                    token0="0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                    token1="0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                    fee_tier=100,  # 0.01%
                    liquidity_weight=1.0
                ),
            ],
            "ETH_MAINNET_DAI": [
                # DAI/USDC pools for depeg detection  
                DEXPoolInfo(
                    pool_address="0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168",
                    dex_type="uniswap_v3",
                    token0="0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
                    token1="0xA0b86a33E6441b5311eD1be2b26B7Bac4f0d5f0b",  # USDC
                    fee_tier=100,
                    liquidity_weight=1.0
                ),
            ],
        }
    
    async def get_price_usd(self, asset_id: str) -> Optional[float]:
        """Get USD price from on-chain DEX pools."""
        # Check cache first
        if asset_id in self.price_cache:
            price, timestamp = self.price_cache[asset_id]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.cache_ttl_seconds):
                return price
        
        # Stablecoins default to $1 (but may have small depegs)
        if any(stable in asset_id for stable in ["USDC", "USDT", "DAI"]):
            if asset_id == "ETH_MAINNET_USDC":
                price = 1.0  # USDC is our base USD asset
            else:
                # Check for depeg against USDC
                price = await self._get_stablecoin_price(asset_id)
                if price is None:
                    price = 1.0  # Fallback to $1
        else:
            # Get price from DEX pools
            price = await self._get_asset_price_from_pools(asset_id)
        
        # Cache the result
        if price is not None:
            self.price_cache[asset_id] = (price, datetime.utcnow())
        
        return price
    
    async def get_price_details(self, asset_id: str) -> Optional[AssetPrice]:
        """Get detailed price information from on-chain data."""
        price = await self.get_price_usd(asset_id)
        if price is None:
            return None
        
        return AssetPrice(
            asset_id=asset_id,
            symbol=self._extract_symbol(asset_id),
            price_usd=price,
            timestamp=datetime.utcnow(),
            source="on_chain_dex",
            confidence=0.95  # High confidence for on-chain prices
        )
    
    async def get_prices_batch(self, asset_ids: List[str]) -> Dict[str, Optional[float]]:
        """Get prices for multiple assets from on-chain data."""
        # Run all price fetches concurrently
        tasks = [self.get_price_usd(asset_id) for asset_id in asset_ids]
        prices = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for asset_id, price in zip(asset_ids, prices):
            if isinstance(price, Exception):
                logger.warning(f"Failed to get price for {asset_id}: {price}")
                results[asset_id] = None
            else:
                results[asset_id] = price
        
        return results
    
    async def _get_stablecoin_price(self, asset_id: str) -> Optional[float]:
        """Get stablecoin price to detect depegs."""
        pools = self.price_discovery_pools.get(asset_id, [])
        if not pools:
            return 1.0  # Default to $1 if no pools configured
        
        # Get price from the first available pool
        for pool_info in pools:
            try:
                price = await self._get_price_from_uniswap_v3_pool(pool_info)
                if price is not None:
                    return price
            except Exception as e:
                logger.warning(f"Failed to get stablecoin price from pool {pool_info.pool_address}: {e}")
                continue
        
        return 1.0  # Fallback
    
    async def _get_asset_price_from_pools(self, asset_id: str) -> Optional[float]:
        """Get asset price by aggregating from multiple DEX pools."""
        pools = self.price_discovery_pools.get(asset_id, [])
        if not pools:
            logger.warning(f"No price discovery pools configured for {asset_id}")
            return None
        
        # Get prices from all pools
        weighted_prices = []
        
        for pool_info in pools:
            try:
                if pool_info.dex_type == "uniswap_v3":
                    price = await self._get_price_from_uniswap_v3_pool(pool_info)
                elif pool_info.dex_type == "uniswap_v2":
                    price = await self._get_price_from_uniswap_v2_pool(pool_info)
                else:
                    logger.warning(f"Unsupported DEX type: {pool_info.dex_type}")
                    continue
                
                if price is not None:
                    weighted_prices.append((price, pool_info.liquidity_weight))
                    
            except Exception as e:
                logger.warning(f"Failed to get price from pool {pool_info.pool_address}: {e}")
                continue
        
        if not weighted_prices:
            return None
        
        # Calculate weighted average
        total_weight = sum(weight for _, weight in weighted_prices)
        weighted_avg = sum(price * weight for price, weight in weighted_prices) / total_weight
        
        return weighted_avg
    
    async def _get_price_from_uniswap_v3_pool(self, pool_info: DEXPoolInfo) -> Optional[float]:
        """Get price from Uniswap V3 pool."""
        web3 = await self.blockchain_provider.get_web3("ethereum")
        if not web3:
            return None
        
        try:
            # Uniswap V3 pool ABI (just the functions we need)
            pool_abi = [
                {
                    "inputs": [],
                    "name": "slot0",
                    "outputs": [
                        {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                        {"internalType": "int24", "name": "tick", "type": "int24"},
                        {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
                        {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
                        {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
                        {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
                        {"internalType": "bool", "name": "unlocked", "type": "bool"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "token1", 
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            pool_contract = web3.eth.contract(
                address=web3.to_checksum_address(pool_info.pool_address),
                abi=pool_abi
            )
            
            # Get pool state
            slot0 = await pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            
            # Calculate price from sqrtPriceX96
            # Price = (sqrtPriceX96 / 2^96)^2
            price_raw = (sqrt_price_x96 / (2 ** 96)) ** 2
            
            # Adjust for token decimals
            # Most tokens are 18 decimals, USDC/USDT are 6 decimals
            token0_decimals = 6 if any(stable in pool_info.token0 for stable in ["USDC", "USDT"]) else 18
            token1_decimals = 6 if any(stable in pool_info.token1 for stable in ["USDC", "USDT"]) else 18
            
            decimal_adjustment = 10 ** (token1_decimals - token0_decimals)
            price_adjusted = price_raw * decimal_adjustment
            
            # Return price in USD terms
            # If token1 is a stablecoin, price_adjusted is the USD price of token0
            # If token0 is a stablecoin, we need to invert
            if any(stable in pool_info.token1 for stable in ["USDC", "USDT", "DAI"]):
                return price_adjusted
            elif any(stable in pool_info.token0 for stable in ["USDC", "USDT", "DAI"]):
                return 1.0 / price_adjusted if price_adjusted > 0 else None
            else:
                # Neither token is a stablecoin, need to handle differently
                logger.warning(f"Pool {pool_info.pool_address} has no stablecoin reference")
                return None
                
        except Exception as e:
            logger.error(f"Error getting Uniswap V3 price from {pool_info.pool_address}: {e}")
            return None
    
    async def _get_price_from_uniswap_v2_pool(self, pool_info: DEXPoolInfo) -> Optional[float]:
        """Get price from Uniswap V2 style pool."""
        web3 = await self.blockchain_provider.get_web3("ethereum")
        if not web3:
            return None
        
        try:
            # Uniswap V2 pair ABI (just getReserves)
            pair_abi = [
                {
                    "inputs": [],
                    "name": "getReserves",
                    "outputs": [
                        {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
                        {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
                        {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            pair_contract = web3.eth.contract(
                address=web3.to_checksum_address(pool_info.pool_address),
                abi=pair_abi
            )
            
            # Get reserves
            reserves = await pair_contract.functions.getReserves().call()
            reserve0, reserve1 = reserves[0], reserves[1]
            
            if reserve0 == 0 or reserve1 == 0:
                return None
            
            # Calculate price ratio
            price_ratio = reserve1 / reserve0
            
            # Adjust for decimals
            token0_decimals = 6 if any(stable in pool_info.token0 for stable in ["USDC", "USDT"]) else 18
            token1_decimals = 6 if any(stable in pool_info.token1 for stable in ["USDC", "USDT"]) else 18
            
            decimal_adjustment = 10 ** (token1_decimals - token0_decimals)
            price_adjusted = price_ratio * decimal_adjustment
            
            # Return USD price
            if any(stable in pool_info.token1 for stable in ["USDC", "USDT", "DAI"]):
                return price_adjusted
            elif any(stable in pool_info.token0 for stable in ["USDC", "USDT", "DAI"]):
                return 1.0 / price_adjusted if price_adjusted > 0 else None
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting Uniswap V2 price from {pool_info.pool_address}: {e}")
            return None
    
    def _extract_symbol(self, asset_id: str) -> str:
        """Extract symbol from asset ID."""
        parts = asset_id.split("_")
        return parts[-1] if parts else asset_id
    
    async def get_market_depth(self, asset_id: str, trade_amount_usd: float) -> Dict[str, Any]:
        """Get market depth and price impact for a given trade size."""
        # This would implement more sophisticated price impact calculation
        # by simulating trades against multiple pools
        # For now, return basic info
        return {
            "asset_id": asset_id,
            "trade_amount_usd": trade_amount_usd,
            "estimated_price_impact": 0.001,  # 0.1% default
            "available_liquidity_usd": 1_000_000  # Default high liquidity
        }