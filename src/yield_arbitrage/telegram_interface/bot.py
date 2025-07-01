"""
Main Telegram Bot Implementation.

This module contains the main YieldArbitrageBot class that integrates all
components of the yield arbitrage system with the Telegram bot interface.
"""
import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    ContextTypes, filters, CallbackContext
)

from .config import BotConfig, ConfigManager
from .auth import UserAuthenticator
from .commands import (
    start_command, help_command, status_command, opportunities_command,
    positions_command, alerts_command, config_command, metrics_command,
    portfolio_command, users_command
)

logger = logging.getLogger(__name__)


class YieldArbitrageBot:
    """
    Main Telegram bot for yield arbitrage system monitoring and control.
    
    Integrates with all system components to provide comprehensive 
    monitoring and control through Telegram interface.
    """
    
    def __init__(
        self,
        config: BotConfig,
        graph=None,
        pathfinder=None,
        simulator=None,
        data_collector=None,
        position_monitor=None,
        delta_tracker=None,
        execution_logger=None,
        config_manager=None
    ):
        """
        Initialize the Telegram bot with system components.
        
        Args:
            config: Bot configuration
            graph: Graph engine instance
            pathfinder: Path finding engine
            simulator: Path simulation engine
            data_collector: Data collection system
            position_monitor: Position monitoring system
            delta_tracker: Portfolio delta tracking
            execution_logger: Execution logging system
            config_manager: Configuration manager
        """
        self.config = config
        self.config_manager = config_manager or ConfigManager()
        
        # System components
        self.graph = graph
        self.pathfinder = pathfinder
        self.simulator = simulator
        self.data_collector = data_collector
        self.position_monitor = position_monitor
        self.delta_tracker = delta_tracker
        self.execution_logger = execution_logger
        
        # Bot infrastructure
        self.application: Optional[Application] = None
        self.authenticator = UserAuthenticator(config)
        self.is_running = False
        self.start_time = datetime.now(timezone.utc)
        self._polling_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "commands_processed": 0,
            "errors_handled": 0,
            "uptime_start": self.start_time,
            "last_restart": None
        }
        
        logger.info("YieldArbitrageBot initialized")
    
    async def initialize(self) -> None:
        """Initialize the bot application and handlers."""
        try:
            # Create application
            builder = Application.builder()
            builder.token(self.config.telegram_bot_token)
            
            # Configure timeouts
            builder.read_timeout(self.config.command_timeout_seconds)
            builder.write_timeout(self.config.command_timeout_seconds)
            
            self.application = builder.build()
            
            # Store system components in bot_data for access by handlers
            self.application.bot_data.update({
                'bot_config': self.config,
                'config_manager': self.config_manager,
                'authenticator': self.authenticator,
                'graph': self.graph,
                'pathfinder': self.pathfinder,
                'simulator': self.simulator,
                'data_collector': self.data_collector,
                'position_monitor': self.position_monitor,
                'delta_tracker': self.delta_tracker,
                'execution_logger': self.execution_logger,
                'bot_instance': self
            })
            
            # Register command handlers
            await self._register_handlers()
            
            # Register error handler
            self.application.add_error_handler(self._error_handler)
            
            logger.info("Bot application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}", exc_info=True)
            raise
    
    async def _register_handlers(self) -> None:
        """Register all command and message handlers."""
        # Core commands
        self.application.add_handler(CommandHandler("start", start_command))
        self.application.add_handler(CommandHandler("help", help_command))
        self.application.add_handler(CommandHandler("status", status_command))
        self.application.add_handler(CommandHandler("opportunities", opportunities_command))
        self.application.add_handler(CommandHandler("positions", positions_command))
        self.application.add_handler(CommandHandler("alerts", alerts_command))
        self.application.add_handler(CommandHandler("config", config_command))
        self.application.add_handler(CommandHandler("metrics", metrics_command))
        self.application.add_handler(CommandHandler("portfolio", portfolio_command))
        
        # Admin commands
        self.application.add_handler(CommandHandler("users", users_command))
        
        # Alias commands
        self.application.add_handler(CommandHandler("ops", opportunities_command))  # Short alias
        self.application.add_handler(CommandHandler("pos", positions_command))     # Short alias
        
        # Unknown command handler
        self.application.add_handler(
            MessageHandler(filters.COMMAND, self._unknown_command_handler)
        )
        
        # Non-command message handler
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._text_message_handler)
        )
        
        logger.info("All command handlers registered")
    
    async def start(self) -> None:
        """Start the bot and begin polling for messages."""
        if not self.application:
            await self.initialize()
        
        try:
            self.is_running = True
            logger.info("Starting Telegram bot...")
            
            # Start polling
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
            logger.info("✅ Telegram bot is running and polling for updates")
            
            # Send startup notification to admin users
            await self._send_startup_notification()
            
            # Run in background instead of blocking
            self._polling_task = asyncio.create_task(self._run_until_stopped())
            logger.info("✅ Telegram bot polling started in background")
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}", exc_info=True)
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the bot gracefully."""
        if not self.is_running:
            return
        
        logger.info("Stopping Telegram bot...")
        self.is_running = False
        
        try:
            # Cancel the polling task
            if self._polling_task and not self._polling_task.done():
                self._polling_task.cancel()
                try:
                    await self._polling_task
                except asyncio.CancelledError:
                    pass
            
            if self.application:
                # Send shutdown notification to admin users
                await self._send_shutdown_notification()
                
                # Stop the application
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            
            logger.info("✅ Telegram bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}", exc_info=True)
    
    async def restart(self) -> None:
        """Restart the bot."""
        logger.info("Restarting Telegram bot...")
        await self.stop()
        await asyncio.sleep(2)
        self.stats['last_restart'] = datetime.now(timezone.utc)
        await self.start()
    
    async def send_alert_to_users(
        self, 
        message: str, 
        severity: str = "info",
        admin_only: bool = False
    ) -> int:
        """
        Send alert message to authorized users.
        
        Args:
            message: Alert message to send
            severity: Alert severity level
            admin_only: If True, send only to admin users
            
        Returns:
            Number of users who received the message
        """
        if not self.application or not self.is_running:
            return 0
        
        # Determine which users to notify
        user_ids = (self.config.admin_user_ids if admin_only 
                   else self.config.allowed_user_ids)
        
        # Add severity emoji
        severity_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '🚨', 
            'critical': '🆘'
        }
        icon = severity_icons.get(severity, 'ℹ️')
        
        formatted_message = f"{icon} **System Alert**\n\n{message}"
        
        sent_count = 0
        for user_id in user_ids:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=formatted_message,
                    parse_mode='Markdown'
                )
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send alert to user {user_id}: {e}")
        
        logger.info(f"Sent {severity} alert to {sent_count}/{len(user_ids)} users")
        return sent_count
    
    async def _run_until_stopped(self) -> None:
        """Keep the bot running until stopped."""
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait until stopped
        while self.is_running:
            await asyncio.sleep(1)
            
            # Periodic cleanup
            if datetime.now().minute % 10 == 0:  # Every 10 minutes
                self.authenticator.cleanup_old_sessions()
    
    async def _send_startup_notification(self) -> None:
        """Send startup notification to admin users."""
        if not self.config.admin_user_ids:
            return
        
        startup_msg = f"""
🚀 **Yield Arbitrage Bot Started**

• Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
• Users authorized: {len(self.config.allowed_user_ids)}
• Position monitoring: {'✅' if self.config.enable_position_monitoring else '❌'}
• Execution logging: {'✅' if self.config.enable_execution_logging else '❌'}

Use `/status` to check system health.
"""
        
        await self.send_alert_to_users(startup_msg, "info", admin_only=True)
    
    async def _send_shutdown_notification(self) -> None:
        """Send shutdown notification to admin users."""
        if not self.config.admin_user_ids:
            return
        
        uptime = datetime.now(timezone.utc) - self.start_time
        shutdown_msg = f"""
🛑 **Yield Arbitrage Bot Shutting Down**

• Uptime: {uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m
• Commands processed: {self.stats['commands_processed']:,}
• Errors handled: {self.stats['errors_handled']}

Bot will be unavailable until restarted.
"""
        
        await self.send_alert_to_users(shutdown_msg, "warning", admin_only=True)
    
    async def _unknown_command_handler(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle unknown commands."""
        # Check authentication first
        session = self.authenticator.authenticate_user(update)
        if not session:
            await update.message.reply_text(
                "❌ Access denied. You are not authorized to use this bot."
            )
            return
        
        command = update.message.text.split()[0]
        await update.message.reply_text(
            f"❓ Unknown command: {command}\n\n"
            f"Use `/help` to see available commands."
        )
    
    async def _text_message_handler(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle non-command text messages."""
        # Check authentication first
        session = self.authenticator.authenticate_user(update)
        if not session:
            await update.message.reply_text(
                "❌ Access denied. You are not authorized to use this bot."
            )
            return
        
        # Provide helpful response for non-command messages
        await update.message.reply_text(
            "💬 I understand commands only. Use `/help` to see what I can do!"
        )
    
    async def _error_handler(
        self, 
        update: Optional[Update], 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle errors that occur during message processing."""
        self.stats['errors_handled'] += 1
        
        # Log the error
        logger.error(f"Error handling update: {context.error}", exc_info=context.error)
        
        # Try to inform the user
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ An error occurred while processing your request. "
                    "Please try again or contact an administrator."
                )
            except Exception as e:
                logger.error(f"Failed to send error message to user: {e}")
        
        # Send error notification to admins for critical errors
        if self.config.admin_user_ids:
            error_msg = f"""
🚨 **Bot Error Detected**

• Error: {str(context.error)[:200]}...
• User: {update.effective_user.id if update and update.effective_user else 'Unknown'}
• Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
• Total errors today: {self.stats['errors_handled']}
"""
            
            await self.send_alert_to_users(error_msg, "error", admin_only=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics."""
        uptime = datetime.now(timezone.utc) - self.start_time
        
        return {
            **self.stats,
            "is_running": self.is_running,
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_formatted": f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m",
            "auth_stats": self.authenticator.get_session_stats() if self.authenticator else {},
            "config_summary": self.config.get_integration_settings() if self.config else {}
        }
    
    def update_component(self, component_name: str, component: Any) -> None:
        """Update a system component reference."""
        if hasattr(self, component_name):
            setattr(self, component_name, component)
            
            # Update in bot_data if application exists
            if self.application and self.application.bot_data:
                self.application.bot_data[component_name] = component
                
            logger.info(f"Updated component: {component_name}")
        else:
            logger.warning(f"Unknown component: {component_name}")


def create_bot_from_config(
    config_path: Optional[str] = None,
    **components
) -> YieldArbitrageBot:
    """
    Factory function to create a YieldArbitrageBot from configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        **components: System components to integrate
        
    Returns:
        Configured YieldArbitrageBot instance
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.load_config()
    
    # Create bot instance
    bot = YieldArbitrageBot(
        config=config,
        config_manager=config_manager,
        **components
    )
    
    return bot


async def run_bot_standalone(
    config_path: Optional[str] = None,
    **components
) -> None:
    """
    Run the bot in standalone mode.
    
    Args:
        config_path: Path to configuration file
        **components: System components to integrate
    """
    bot = create_bot_from_config(config_path, **components)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
    finally:
        await bot.stop()


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run standalone bot for testing
    asyncio.run(run_bot_standalone())