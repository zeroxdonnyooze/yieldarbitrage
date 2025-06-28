"""Token filtering logic with external API integration."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from decimal import Decimal

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TokenCriteria:
    """Token filtering criteria."""
    min_market_cap_usd: float = 1_000_000  # $1M minimum market cap
    min_daily_volume_usd: float = 50_000   # $50k minimum daily volume
    min_pool_tvl_usd: float = 100_000      # $100k minimum pool TVL
    max_price_impact: float = 0.05         # 5% maximum price impact
    require_verified: bool = True          # Only verified tokens
    blacklisted_tokens: Set[str] = None    # Blacklisted token addresses
    whitelisted_tokens: Set[str] = None    # Whitelisted token addresses (bypass other criteria)
    
    def __post_init__(self):
        if self.blacklisted_tokens is None:
            self.blacklisted_tokens = set()
        if self.whitelisted_tokens is None:
            self.whitelisted_tokens = set()


@dataclass
class TokenInfo:
    """Token information from external APIs."""
    address: str
    symbol: str
    name: str
    decimals: int
    market_cap_usd: Optional[float] = None
    daily_volume_usd: Optional[float] = None
    price_usd: Optional[float] = None
    price_change_24h: Optional[float] = None
    is_verified: bool = False
    coingecko_id: Optional[str] = None
    defillama_id: Optional[str] = None
    last_updated: Optional[datetime] = None
    
    def meets_criteria(self, criteria: TokenCriteria) -> bool:
        """Check if token meets filtering criteria."""
        # Whitelisted tokens bypass all criteria
        if self.address.lower() in criteria.whitelisted_tokens:
            return True
        
        # Blacklisted tokens are always rejected
        if self.address.lower() in criteria.blacklisted_tokens:
            return False
        
        # Check verification requirement
        if criteria.require_verified and not self.is_verified:
            return False
        
        # Check market cap
        if (self.market_cap_usd is not None and 
            self.market_cap_usd < criteria.min_market_cap_usd):
            return False
        
        # Check daily volume
        if (self.daily_volume_usd is not None and 
            self.daily_volume_usd < criteria.min_daily_volume_usd):
            return False
        
        return True


@dataclass
class PoolInfo:
    """Pool information for TVL filtering."""
    pool_address: str
    token0_address: str
    token1_address: str
    tvl_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    fee_tier: Optional[int] = None
    protocol: Optional[str] = None
    last_updated: Optional[datetime] = None
    
    def meets_criteria(self, criteria: TokenCriteria) -> bool:
        """Check if pool meets filtering criteria."""
        # Check TVL requirement
        if (self.tvl_usd is not None and 
            self.tvl_usd < criteria.min_pool_tvl_usd):
            return False
        
        return True


class ExternalAPIClient:
    """Client for fetching data from external APIs."""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.defillama_base_url = "https://api.llama.fi"
        
        # Rate limiting
        self.coingecko_rate_limit = asyncio.Semaphore(10)  # 10 requests per second
        self.defillama_rate_limit = asyncio.Semaphore(20)  # 20 requests per second
    
    async def get_token_info_coingecko(self, contract_address: str, chain: str = "ethereum") -> Optional[TokenInfo]:
        """Get token information from CoinGecko."""
        try:
            async with self.coingecko_rate_limit:
                # Map chain names to CoinGecko platform IDs
                platform_map = {
                    "ethereum": "ethereum",
                    "arbitrum": "arbitrum-one",
                    "base": "base",
                    "polygon": "polygon-pos"
                }
                
                platform_id = platform_map.get(chain.lower(), "ethereum")
                
                url = f"{self.coingecko_base_url}/coins/{platform_id}/contract/{contract_address}"
                
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logger.warning(f"CoinGecko API error {response.status} for {contract_address}")
                        return None
                    
                    data = await response.json()
                    
                    market_data = data.get("market_data", {})
                    
                    return TokenInfo(
                        address=contract_address.lower(),
                        symbol=data.get("symbol", "").upper(),
                        name=data.get("name", ""),
                        decimals=data.get("detail_platforms", {}).get(platform_id, {}).get("decimal_place", 18),
                        market_cap_usd=market_data.get("market_cap", {}).get("usd"),
                        daily_volume_usd=market_data.get("total_volume", {}).get("usd"),
                        price_usd=market_data.get("current_price", {}).get("usd"),
                        price_change_24h=market_data.get("price_change_percentage_24h"),
                        is_verified=data.get("asset_platform_id") is not None,
                        coingecko_id=data.get("id"),
                        last_updated=datetime.now(timezone.utc)
                    )
                    
        except asyncio.TimeoutError:
            logger.warning(f"CoinGecko API timeout for {contract_address}")
            return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data for {contract_address}: {e}")
            return None
    
    async def get_protocol_tvl_defillama(self, protocol: str) -> Optional[Dict]:
        """Get protocol TVL from DeFiLlama."""
        try:
            async with self.defillama_rate_limit:
                url = f"{self.defillama_base_url}/protocol/{protocol}"
                
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logger.warning(f"DeFiLlama API error {response.status} for {protocol}")
                        return None
                    
                    data = await response.json()
                    return data
                    
        except asyncio.TimeoutError:
            logger.warning(f"DeFiLlama API timeout for {protocol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching DeFiLlama data for {protocol}: {e}")
            return None
    
    async def get_pools_defillama(self, protocol: str, chain: str = None) -> List[Dict]:
        """Get pool data from DeFiLlama."""
        try:
            async with self.defillama_rate_limit:
                url = f"{self.defillama_base_url}/pools"
                params = {}
                
                if protocol:
                    params["protocol"] = protocol
                if chain:
                    params["chain"] = chain
                
                async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status != 200:
                        logger.warning(f"DeFiLlama pools API error {response.status}")
                        return []
                    
                    data = await response.json()
                    return data.get("data", [])
                    
        except asyncio.TimeoutError:
            logger.warning(f"DeFiLlama pools API timeout")
            return []
        except Exception as e:
            logger.error(f"Error fetching DeFiLlama pools data: {e}")
            return []


class TokenFilterCache:
    """Cache for token filtering data."""
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour TTL
        self.ttl_seconds = ttl_seconds
        self._token_cache: Dict[str, TokenInfo] = {}
        self._pool_cache: Dict[str, PoolInfo] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        timestamp = self._cache_timestamps.get(key)
        if not timestamp:
            return True
        
        return datetime.now(timezone.utc) - timestamp > timedelta(seconds=self.ttl_seconds)
    
    def get_token(self, address: str) -> Optional[TokenInfo]:
        """Get cached token info."""
        key = f"token_{address.lower()}"
        if self.is_expired(key):
            return None
        return self._token_cache.get(key)
    
    def set_token(self, address: str, token_info: TokenInfo):
        """Cache token info."""
        key = f"token_{address.lower()}"
        self._token_cache[key] = token_info
        self._cache_timestamps[key] = datetime.now(timezone.utc)
    
    def get_pool(self, pool_address: str) -> Optional[PoolInfo]:
        """Get cached pool info."""
        key = f"pool_{pool_address.lower()}"
        if self.is_expired(key):
            return None
        return self._pool_cache.get(key)
    
    def set_pool(self, pool_address: str, pool_info: PoolInfo):
        """Cache pool info."""
        key = f"pool_{pool_address.lower()}"
        self._pool_cache[key] = pool_info
        self._cache_timestamps[key] = datetime.now(timezone.utc)
    
    def clear_expired(self):
        """Clear expired cache entries."""
        expired_keys = [key for key in self._cache_timestamps if self.is_expired(key)]
        
        for key in expired_keys:
            if key.startswith("token_"):
                self._token_cache.pop(key, None)
            elif key.startswith("pool_"):
                self._pool_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")


class TokenFilter:
    """Token filtering system with external API integration."""
    
    def __init__(self, criteria: TokenCriteria = None, cache_ttl: int = 3600):
        self.criteria = criteria or TokenCriteria()
        self.cache = TokenFilterCache(cache_ttl)
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_client: Optional[ExternalAPIClient] = None
        
        # Statistics
        self.stats = {
            "tokens_evaluated": 0,
            "tokens_passed": 0,
            "tokens_failed": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "api_errors": 0
        }
    
    async def initialize(self):
        """Initialize the token filter with HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            self.api_client = ExternalAPIClient(self.session)
            logger.info("Token filter initialized")
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.api_client = None
            logger.info("Token filter closed")
    
    async def get_token_info(self, address: str, chain: str = "ethereum") -> Optional[TokenInfo]:
        """Get token information with caching."""
        if not self.api_client:
            await self.initialize()
        
        # Check cache first
        cached_info = self.cache.get_token(address)
        if cached_info:
            self.stats["cache_hits"] += 1
            return cached_info
        
        # Fetch from API
        self.stats["api_calls"] += 1
        token_info = await self.api_client.get_token_info_coingecko(address, chain)
        
        if token_info:
            self.cache.set_token(address, token_info)
        else:
            self.stats["api_errors"] += 1
        
        return token_info
    
    async def filter_token(self, address: str, chain: str = "ethereum") -> bool:
        """Filter a single token based on criteria."""
        self.stats["tokens_evaluated"] += 1
        
        try:
            token_info = await self.get_token_info(address, chain)
            
            if not token_info:
                self.stats["tokens_failed"] += 1
                return False
            
            passed = token_info.meets_criteria(self.criteria)
            
            if passed:
                self.stats["tokens_passed"] += 1
            else:
                self.stats["tokens_failed"] += 1
            
            return passed
            
        except Exception as e:
            logger.error(f"Error filtering token {address}: {e}")
            self.stats["tokens_failed"] += 1
            return False
    
    async def filter_tokens(self, addresses: List[str], chain: str = "ethereum") -> List[str]:
        """Filter multiple tokens and return addresses that pass criteria."""
        if not addresses:
            return []
        
        # Process tokens in batches to avoid overwhelming APIs
        batch_size = 10
        passed_tokens = []
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.filter_token(addr, chain) for addr in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect passed tokens
            for addr, result in zip(batch, results):
                if result is True:
                    passed_tokens.append(addr)
                elif isinstance(result, Exception):
                    logger.error(f"Error filtering token {addr}: {result}")
            
            # Small delay between batches to be respectful to APIs
            if i + batch_size < len(addresses):
                await asyncio.sleep(0.1)
        
        logger.info(f"Filtered {len(addresses)} tokens: {len(passed_tokens)} passed, "
                   f"{len(addresses) - len(passed_tokens)} failed")
        
        return passed_tokens
    
    async def get_pool_info(self, pool_address: str, protocol: str, chain: str = "ethereum") -> Optional[PoolInfo]:
        """Get pool information from DeFiLlama."""
        if not self.api_client:
            await self.initialize()
        
        # Check cache first
        cached_info = self.cache.get_pool(pool_address)
        if cached_info:
            self.stats["cache_hits"] += 1
            return cached_info
        
        try:
            # Get pools from DeFiLlama
            pools_data = await self.api_client.get_pools_defillama(protocol, chain)
            
            # Find our specific pool
            for pool_data in pools_data:
                if pool_data.get("pool", "").lower() == pool_address.lower():
                    pool_info = PoolInfo(
                        pool_address=pool_address.lower(),
                        token0_address=pool_data.get("token0", "").lower(),
                        token1_address=pool_data.get("token1", "").lower(),
                        tvl_usd=pool_data.get("tvlUsd"),
                        volume_24h_usd=pool_data.get("volumeUsd1d"),
                        protocol=protocol,
                        last_updated=datetime.now(timezone.utc)
                    )
                    
                    self.cache.set_pool(pool_address, pool_info)
                    return pool_info
            
            logger.warning(f"Pool {pool_address} not found in {protocol} data")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching pool info for {pool_address}: {e}")
            self.stats["api_errors"] += 1
            return None
    
    async def filter_pool(self, pool_address: str, protocol: str, chain: str = "ethereum") -> bool:
        """Filter a pool based on TVL and other criteria."""
        try:
            pool_info = await self.get_pool_info(pool_address, protocol, chain)
            
            if not pool_info:
                return False
            
            return pool_info.meets_criteria(self.criteria)
            
        except Exception as e:
            logger.error(f"Error filtering pool {pool_address}: {e}")
            return False
    
    def update_criteria(self, **kwargs):
        """Update filtering criteria."""
        for key, value in kwargs.items():
            if hasattr(self.criteria, key):
                setattr(self.criteria, key, value)
                logger.info(f"Updated criteria {key} = {value}")
    
    def get_stats(self) -> Dict:
        """Get filtering statistics."""
        return {
            **self.stats,
            "cache_size": len(self.cache._token_cache) + len(self.cache._pool_cache),
            "success_rate": (self.stats["tokens_passed"] / max(1, self.stats["tokens_evaluated"]) * 100),
            "cache_hit_rate": (self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["api_calls"]) * 100)
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache._token_cache.clear()
        self.cache._pool_cache.clear()
        self.cache._cache_timestamps.clear()
        logger.info("Token filter cache cleared")


# Default token filter instance
default_token_filter = TokenFilter()