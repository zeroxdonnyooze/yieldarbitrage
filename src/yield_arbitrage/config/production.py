"""Production configuration for real data integration."""
import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Load environment-specific configuration
try:
    from .environment import initialize_environment
    # Initialize environment configuration on import
    initialize_environment()
except ImportError:
    # Fallback to dotenv if environment module not available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # If python-dotenv is not available, try to load manually
        env_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ.setdefault(key, value)

logger = logging.getLogger(__name__)


@dataclass
class OracleConfig:
    """Configuration for price oracles."""
    primary_oracle: str = "on_chain"  # On-chain first for DeFi arbitrage
    fallback_oracles: List[str] = None
    cache_ttl_seconds: int = 30  # 30 seconds for on-chain prices
    stale_threshold_seconds: int = 300  # 5 minutes for on-chain
    
    # On-chain oracle settings
    on_chain_enabled: bool = True
    on_chain_cache_blocks: int = 1  # Cache for 1 block (~12 seconds)
    
    # External API oracles (backup only)
    coingecko_api_key: Optional[str] = None
    coinmarketcap_api_key: Optional[str] = None
    cryptocompare_api_key: Optional[str] = None
    defillama_enabled: bool = False  # Disabled by default for production
    coingecko_enabled: bool = False  # Disabled by default for production
    
    def __post_init__(self):
        if self.fallback_oracles is None:
            # On-chain first, external APIs as emergency backup only
            self.fallback_oracles = ["coingecko"] if self.coingecko_api_key else []


@dataclass
class ProductionConfig:
    """Production configuration settings."""
    # Database
    database_url: str
    redis_url: str
    
    # Blockchain
    alchemy_api_key: str
    ethereum_rpc_url: str
    arbitrum_rpc_url: str
    base_rpc_url: str
    sonic_rpc_url: str
    berachain_rpc_url: str
    
    # Oracle configuration
    oracle: OracleConfig
    
    # Trading parameters
    min_profit_threshold_usd: float = 0.50
    max_gas_percentage: float = 0.02
    beam_width: int = 50
    max_path_length: int = 10
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Load production configuration from environment variables."""
        # Required settings
        database_url = os.getenv("DATABASE_URL")
        redis_url = os.getenv("REDIS_URL")
        alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
        
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        if not redis_url:
            raise ValueError("REDIS_URL environment variable is required")
        if not alchemy_api_key:
            raise ValueError("ALCHEMY_API_KEY environment variable is required")
        
        # Blockchain RPC URLs
        ethereum_rpc_url = os.getenv("ETHEREUM_RPC_URL", f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_api_key}")
        arbitrum_rpc_url = os.getenv("ARBITRUM_RPC_URL", f"https://arb-mainnet.g.alchemy.com/v2/{alchemy_api_key}")
        base_rpc_url = os.getenv("BASE_RPC_URL", f"https://base-mainnet.g.alchemy.com/v2/{alchemy_api_key}")
        sonic_rpc_url = os.getenv("SONIC_RPC_URL", f"https://sonic-mainnet.g.alchemy.com/v2/{alchemy_api_key}")
        berachain_rpc_url = os.getenv("BERACHAIN_RPC_URL", f"https://berachain-mainnet.g.alchemy.com/v2/{alchemy_api_key}")
        
        # Oracle configuration (prioritize on-chain for production)
        oracle_config = OracleConfig(
            primary_oracle=os.getenv("PRICE_ORACLE_PRIMARY", "on_chain"),
            fallback_oracles=os.getenv("PRICE_ORACLE_FALLBACK", "coingecko").split(","),
            cache_ttl_seconds=int(os.getenv("PRICE_ORACLE_CACHE_TTL", "30")),
            stale_threshold_seconds=int(os.getenv("PRICE_ORACLE_FALLBACK_TTL", "300")),
            coingecko_api_key=os.getenv("COINGECKO_API_KEY"),
            coinmarketcap_api_key=os.getenv("COINMARKETCAP_API_KEY"),
            cryptocompare_api_key=os.getenv("CRYPTOCOMPARE_API_KEY"),
            defillama_enabled=os.getenv("DEFILLAMA_ENABLED", "false").lower() == "true",
            on_chain_enabled=os.getenv("ON_CHAIN_ORACLE_ENABLED", "true").lower() == "true",
            coingecko_enabled=os.getenv("COINGECKO_ENABLED", "true").lower() == "true"
        )
        
        return cls(
            database_url=database_url,
            redis_url=redis_url,
            alchemy_api_key=alchemy_api_key,
            ethereum_rpc_url=ethereum_rpc_url,
            arbitrum_rpc_url=arbitrum_rpc_url,
            base_rpc_url=base_rpc_url,
            sonic_rpc_url=sonic_rpc_url,
            berachain_rpc_url=berachain_rpc_url,
            oracle=oracle_config,
            min_profit_threshold_usd=float(os.getenv("MIN_PROFIT_THRESHOLD_USD", "0.50")),
            max_gas_percentage=float(os.getenv("MAX_GAS_PERCENTAGE", "0.02")),
            beam_width=int(os.getenv("BEAM_WIDTH", "50")),
            max_path_length=int(os.getenv("MAX_PATH_LENGTH", "10")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def get_rpc_urls(self) -> Dict[str, str]:
        """Get mapping of chain names to RPC URLs."""
        return {
            "ethereum": self.ethereum_rpc_url,
            "arbitrum": self.arbitrum_rpc_url,
            "base": self.base_rpc_url,
            "sonic": self.sonic_rpc_url,
            "berachain": self.berachain_rpc_url
        }
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate database URL format
        if not self.database_url.startswith(("postgresql://", "postgresql+asyncpg://")):
            errors.append("DATABASE_URL must be a valid PostgreSQL connection string")
        
        # Validate Redis URL format
        if not self.redis_url.startswith("redis://"):
            errors.append("REDIS_URL must be a valid Redis connection string")
        
        # Validate RPC URLs
        rpc_urls = self.get_rpc_urls()
        for chain, url in rpc_urls.items():
            if not url.startswith("https://"):
                errors.append(f"{chain.upper()}_RPC_URL must be a valid HTTPS URL")
        
        # Validate trading parameters
        if self.min_profit_threshold_usd < 0:
            errors.append("MIN_PROFIT_THRESHOLD_USD must be non-negative")
        
        if not 0 < self.max_gas_percentage <= 1:
            errors.append("MAX_GAS_PERCENTAGE must be between 0 and 1")
        
        if self.beam_width < 1:
            errors.append("BEAM_WIDTH must be at least 1")
        
        if self.max_path_length < 1:
            errors.append("MAX_PATH_LENGTH must be at least 1")
        
        # Validate oracle configuration
        if self.oracle.primary_oracle not in ["coingecko", "defillama", "on_chain"]:
            errors.append("PRICE_ORACLE_PRIMARY must be one of: coingecko, defillama, on_chain")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        logger.info("Production configuration validated successfully")


# Global configuration instance
config: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get the global production configuration."""
    global config
    if config is None:
        config = ProductionConfig.from_env()
        config.validate()
    return config


def init_config() -> ProductionConfig:
    """Initialize and return the production configuration."""
    global config
    config = ProductionConfig.from_env()
    config.validate()
    
    logger.info("Production configuration initialized")
    logger.info(f"Database: {config.database_url.split('@')[1] if '@' in config.database_url else config.database_url}")
    logger.info(f"Redis: {config.redis_url}")
    logger.info(f"Primary Oracle: {config.oracle.primary_oracle}")
    logger.info(f"Fallback Oracles: {', '.join(config.oracle.fallback_oracles)}")
    logger.info(f"Blockchain chains configured: {len(config.get_rpc_urls())}")
    
    return config