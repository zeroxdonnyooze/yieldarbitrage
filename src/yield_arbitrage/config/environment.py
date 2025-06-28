"""Environment configuration manager for different deployment environments."""
import os
import logging
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentManager:
    """Manages environment-specific configuration loading."""
    
    def __init__(self):
        self.current_env = self._detect_environment()
        self.config_root = self._get_config_root()
        
    def _detect_environment(self) -> Environment:
        """Detect the current environment from environment variables."""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _get_config_root(self) -> Path:
        """Get the configuration root directory."""
        # Start from the current file and go up to find the config directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        config_dir = project_root / "config"
        
        if not config_dir.exists():
            logger.warning(f"Config directory not found at {config_dir}, using current directory")
            config_dir = Path.cwd() / "config"
            
        return config_dir
    
    def load_environment_config(self) -> Dict[str, str]:
        """Load configuration for the current environment."""
        env_file = self.config_root / f"{self.current_env.value}.env"
        
        if not env_file.exists():
            logger.error(f"Environment file not found: {env_file}")
            raise FileNotFoundError(f"Environment configuration file not found: {env_file}")
        
        logger.info(f"Loading environment configuration from: {env_file}")
        
        config = {}
        
        try:
            with open(env_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        config[key] = value
                    else:
                        logger.warning(f"Malformed line {line_num} in {env_file}: {line}")
        
        except Exception as e:
            logger.error(f"Failed to load environment configuration: {e}")
            raise
        
        # Set environment variables
        for key, value in config.items():
            # Only set if not already set (allow override via actual env vars)
            os.environ.setdefault(key, value)
        
        logger.info(f"Loaded {len(config)} configuration settings for {self.current_env.value}")
        return config
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment."""
        return {
            "environment": self.current_env.value,
            "config_file": str(self.config_root / f"{self.current_env.value}.env"),
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
            "mock_mode": os.getenv("MOCK_MODE", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000"))
        }
    
    def validate_required_vars(self, required_vars: list) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables for {self.current_env.value}: "
                f"{', '.join(missing_vars)}"
            )
        
        logger.info(f"All required environment variables validated for {self.current_env.value}")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.current_env == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.current_env == Environment.DEVELOPMENT
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.current_env == Environment.STAGING


# Global environment manager instance
env_manager = EnvironmentManager()


def get_environment() -> Environment:
    """Get the current environment."""
    return env_manager.current_env


def load_environment() -> Dict[str, str]:
    """Load configuration for the current environment."""
    return env_manager.load_environment_config()


def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment."""
    return env_manager.get_environment_info()


def validate_environment(required_vars: Optional[list] = None) -> None:
    """Validate the current environment configuration."""
    if required_vars is None:
        # Default required variables for all environments
        required_vars = [
            "ENVIRONMENT",
            "DATABASE_URL",
            "REDIS_URL",
            "ALCHEMY_API_KEY"
        ]
        
        # Add production-specific required variables
        if env_manager.is_production():
            required_vars.extend([
                "TRADING_PRIVATE_KEY",
                "API_KEY_SECRET",
                "JWT_SECRET"
            ])
    
    env_manager.validate_required_vars(required_vars)


def initialize_environment() -> Dict[str, Any]:
    """Initialize the environment configuration."""
    logger.info("Initializing environment configuration...")
    
    # Load environment-specific configuration
    config = load_environment()
    
    # Get environment info
    env_info = get_environment_info()
    
    # Log environment information
    logger.info(f"Environment: {env_info['environment']}")
    logger.info(f"Debug mode: {env_info['debug_mode']}")
    logger.info(f"Mock mode: {env_info['mock_mode']}")
    logger.info(f"Log level: {env_info['log_level']}")
    logger.info(f"Server: {env_info['host']}:{env_info['port']}")
    
    # Set up logging level
    log_level = getattr(logging, env_info['log_level'].upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    
    # Validate environment (will raise if required vars are missing)
    try:
        validate_environment()
        logger.info("Environment validation passed")
    except ValueError as e:
        logger.error(f"Environment validation failed: {e}")
        raise
    
    return env_info