"""Application settings and configuration."""
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Database settings
    database_url: Optional[str] = Field(
        default=None, 
        description="PostgreSQL database URL",
        alias="DATABASE_URL"
    )
    
    db_pool_size: int = Field(
        default=15,
        description="Database connection pool size",
        alias="DB_POOL_SIZE"
    )
    
    db_max_overflow: int = Field(
        default=25,
        description="Maximum overflow connections",
        alias="DB_MAX_OVERFLOW"
    )
    
    db_pool_timeout: int = Field(
        default=30,
        description="Pool timeout in seconds",
        alias="DB_POOL_TIMEOUT"
    )
    
    db_pool_recycle: int = Field(
        default=3600,
        description="Pool connection recycle time in seconds",
        alias="DB_POOL_RECYCLE"
    )
    
    # Redis settings
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
        alias="REDIS_URL"
    )
    
    # Blockchain settings
    alchemy_api_key: Optional[str] = Field(
        default=None,
        description="Alchemy API key for blockchain access",
        alias="ALCHEMY_API_KEY"
    )
    
    infura_api_key: Optional[str] = Field(
        default=None,
        description="Infura API key for blockchain access",
        alias="INFURA_API_KEY"
    )
    
    # RPC URLs for different chains
    ethereum_rpc_url: Optional[str] = Field(
        default=None,
        description="Ethereum mainnet RPC URL",
        alias="ETHEREUM_RPC_URL"
    )
    
    arbitrum_rpc_url: Optional[str] = Field(
        default=None,
        description="Arbitrum mainnet RPC URL", 
        alias="ARBITRUM_RPC_URL"
    )
    
    base_rpc_url: Optional[str] = Field(
        default=None,
        description="Base mainnet RPC URL",
        alias="BASE_RPC_URL"
    )
    
    sonic_rpc_url: Optional[str] = Field(
        default=None,
        description="Sonic mainnet RPC URL",
        alias="SONIC_RPC_URL"
    )
    
    berachain_rpc_url: Optional[str] = Field(
        default=None,
        description="Berachain mainnet RPC URL",
        alias="BERACHAIN_RPC_URL"
    )
    
    # Telegram settings
    telegram_bot_token: Optional[str] = Field(
        default=None,
        description="Telegram bot token",
        alias="TELEGRAM_BOT_TOKEN"
    )
    
    telegram_allowed_users: str = Field(
        default="",
        description="Comma-separated list of allowed Telegram user IDs",
        alias="TELEGRAM_ALLOWED_USERS"
    )
    
    # ML settings
    model_path: Optional[str] = Field(
        default=None,
        description="Path to ML model files",
        alias="MODEL_PATH"
    )
    
    # Trading settings
    min_profit_threshold_usd: float = Field(
        default=0.50,
        description="Minimum profit threshold in USD",
        alias="MIN_PROFIT_THRESHOLD_USD"
    )
    
    max_gas_percentage: float = Field(
        default=0.02,
        description="Maximum gas cost as percentage of profit",
        alias="MAX_GAS_PERCENTAGE"  
    )
    
    beam_width: int = Field(
        default=50,
        description="Beam width for pathfinding algorithm",
        alias="BEAM_WIDTH"
    )
    
    max_path_length: int = Field(
        default=10,
        description="Maximum path length for arbitrage opportunities",
        alias="MAX_PATH_LENGTH"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from .env
    }


# Global settings instance
settings = Settings()