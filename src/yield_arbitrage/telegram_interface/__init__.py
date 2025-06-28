"""Telegram Bot Interface for Yield Arbitrage System."""

from .bot import YieldArbitrageBot, create_bot_from_config, run_bot_standalone
from .config import BotConfig, ConfigManager
from .auth import UserAuthenticator, auth_required, admin_required
from .commands import (
    start_command,
    status_command,
    opportunities_command,
    config_command,
    help_command,
    positions_command,
    alerts_command,
    metrics_command,
    portfolio_command,
    users_command
)
from .formatters import (
    format_system_status,
    format_opportunities,
    format_position_alerts,
    format_execution_metrics,
    format_portfolio_health,
    format_config_display
)
from .service_bot import TelegramBotService, run_bot_service
from .adapters import (
    DatabaseGraphAdapter, DatabasePositionMonitor, DatabaseDeltaTracker,
    DatabaseExecutionLogger
)

__all__ = [
    # Core bot classes
    "YieldArbitrageBot",
    "BotConfig", 
    "ConfigManager",
    "UserAuthenticator",
    "TelegramBotService",
    
    # Bot factory functions
    "create_bot_from_config",
    "run_bot_standalone",
    "run_bot_service",
    
    # Authentication decorators
    "auth_required",
    "admin_required",
    
    # Command handlers
    "start_command",
    "status_command",
    "opportunities_command",
    "config_command",
    "help_command",
    "positions_command",
    "alerts_command",
    "metrics_command",
    "portfolio_command",
    "users_command",
    
    # Formatters
    "format_system_status",
    "format_opportunities", 
    "format_position_alerts",
    "format_execution_metrics",
    "format_portfolio_health",
    "format_config_display",
    
    # Production service
    "TelegramBotService",
    "run_bot_service",
    
    # Database adapters
    "DatabaseGraphAdapter",
    "DatabasePositionMonitor", 
    "DatabaseDeltaTracker",
    "DatabaseExecutionLogger"
]