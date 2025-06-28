"""
Telegram Bot Configuration Management.

This module handles configuration for the Telegram bot interface including
user authentication, bot settings, and integration parameters.
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuration for the Telegram bot interface."""
    
    # Bot authentication
    telegram_bot_token: str
    allowed_user_ids: List[int] = field(default_factory=list)
    admin_user_ids: List[int] = field(default_factory=list)
    
    # Bot behavior
    command_timeout_seconds: int = 30
    max_message_length: int = 4096
    max_opportunities_displayed: int = 10
    max_alerts_displayed: int = 20
    
    # Integration settings
    enable_position_monitoring: bool = True
    enable_execution_logging: bool = True
    enable_risk_alerts: bool = True
    enable_auto_responses: bool = False
    
    # Display preferences
    default_currency: str = "USD"
    precision_decimals: int = 4
    show_timestamps: bool = True
    compact_mode: bool = False
    
    # Rate limiting
    commands_per_minute: int = 60
    opportunities_cooldown_seconds: int = 10
    status_cooldown_seconds: int = 5
    
    # Notification settings
    alert_severity_threshold: str = "warning"  # info, warning, error, critical
    send_position_alerts: bool = True
    send_execution_alerts: bool = True
    send_system_alerts: bool = True
    
    @classmethod
    def from_env(cls) -> 'BotConfig':
        """Create configuration from environment variables."""
        config = cls(
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            allowed_user_ids=cls._parse_user_ids(os.getenv('TELEGRAM_ALLOWED_USERS', '')),
            admin_user_ids=cls._parse_user_ids(os.getenv('TELEGRAM_ADMIN_USERS', '')),
            command_timeout_seconds=int(os.getenv('TELEGRAM_COMMAND_TIMEOUT', '30')),
            max_opportunities_displayed=int(os.getenv('TELEGRAM_MAX_OPPORTUNITIES', '10')),
            enable_position_monitoring=os.getenv('TELEGRAM_ENABLE_MONITORING', 'true').lower() == 'true',
            enable_execution_logging=os.getenv('TELEGRAM_ENABLE_LOGGING', 'true').lower() == 'true',
            alert_severity_threshold=os.getenv('TELEGRAM_ALERT_THRESHOLD', 'warning').lower()
        )
        
        if not config.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'BotConfig':
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @staticmethod
    def _parse_user_ids(user_ids_str: str) -> List[int]:
        """Parse comma-separated user IDs from string."""
        if not user_ids_str.strip():
            return []
        
        try:
            return [int(uid.strip()) for uid in user_ids_str.split(',') if uid.strip()]
        except ValueError as e:
            logger.warning(f"Failed to parse user IDs '{user_ids_str}': {e}")
            return []
    
    def add_user(self, user_id: int, is_admin: bool = False) -> None:
        """Add a user to the allowed list."""
        if user_id not in self.allowed_user_ids:
            self.allowed_user_ids.append(user_id)
        
        if is_admin and user_id not in self.admin_user_ids:
            self.admin_user_ids.append(user_id)
    
    def remove_user(self, user_id: int) -> None:
        """Remove a user from all lists."""
        if user_id in self.allowed_user_ids:
            self.allowed_user_ids.remove(user_id)
        
        if user_id in self.admin_user_ids:
            self.admin_user_ids.remove(user_id)
    
    def is_user_allowed(self, user_id: int) -> bool:
        """Check if a user is allowed to use the bot."""
        return user_id in self.allowed_user_ids
    
    def is_user_admin(self, user_id: int) -> bool:
        """Check if a user has admin privileges."""
        return user_id in self.admin_user_ids
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.telegram_bot_token:
            raise ValueError("telegram_bot_token is required")
        
        if self.command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        
        if self.max_message_length <= 0:
            raise ValueError("max_message_length must be positive")
        
        if self.max_opportunities_displayed <= 0:
            raise ValueError("max_opportunities_displayed must be positive")
        
        if self.precision_decimals < 0:
            raise ValueError("precision_decimals cannot be negative")
        
        valid_thresholds = ["info", "warning", "error", "critical"]
        if self.alert_severity_threshold not in valid_thresholds:
            raise ValueError(f"alert_severity_threshold must be one of: {valid_thresholds}")
    
    def get_display_settings(self) -> Dict[str, Any]:
        """Get display-related settings as a dictionary."""
        return {
            "currency": self.default_currency,
            "precision": self.precision_decimals,
            "show_timestamps": self.show_timestamps,
            "compact_mode": self.compact_mode,
            "max_message_length": self.max_message_length
        }
    
    def get_integration_settings(self) -> Dict[str, Any]:
        """Get integration-related settings as a dictionary."""
        return {
            "position_monitoring": self.enable_position_monitoring,
            "execution_logging": self.enable_execution_logging,
            "risk_alerts": self.enable_risk_alerts,
            "auto_responses": self.enable_auto_responses
        }


class ConfigManager:
    """Manages configuration loading and updates."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[BotConfig] = None
    
    def load_config(self) -> BotConfig:
        """Load configuration from file or environment."""
        if self.config_path and Path(self.config_path).exists():
            logger.info(f"Loading configuration from file: {self.config_path}")
            self._config = BotConfig.from_file(self.config_path)
        else:
            logger.info("Loading configuration from environment variables")
            self._config = BotConfig.from_env()
        
        self._config.validate()
        return self._config
    
    def save_config(self, config: BotConfig) -> None:
        """Save configuration to file."""
        if not self.config_path:
            raise ValueError("No config_path specified for saving")
        
        config.validate()
        config.to_file(self.config_path)
        self._config = config
        logger.info(f"Configuration saved to: {self.config_path}")
    
    def get_config(self) -> BotConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, **updates) -> BotConfig:
        """Update configuration parameters."""
        if self._config is None:
            self._config = self.load_config()
        
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        self._config.validate()
        
        if self.config_path:
            self.save_config(self._config)
        
        return self._config