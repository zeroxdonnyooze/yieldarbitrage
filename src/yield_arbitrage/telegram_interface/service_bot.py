#!/usr/bin/env python3
"""
Production Telegram Bot Service.

This module provides the production-ready Telegram bot that integrates
with the actual yield arbitrage service components.
"""
import asyncio
import logging
import os
from typing import Optional
from dotenv import load_dotenv

from yield_arbitrage.config.settings import settings
from yield_arbitrage.database import startup_database, shutdown_database
from yield_arbitrage.cache import get_redis, close_redis
from .config import BotConfig
from .bot import YieldArbitrageBot

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TelegramBotService:
    """Production Telegram bot service that integrates with real system components."""
    
    def __init__(self, graph_engine=None):
        self.bot: Optional[YieldArbitrageBot] = None
        self.config: Optional[BotConfig] = None
        self.components = {}
        self.graph_engine = graph_engine  # Use provided graph engine
    
    async def initialize(self):
        """Initialize all service components and the bot."""
        logger.info("🚀 Initializing Telegram Bot Service...")
        
        # Initialize infrastructure
        await self._initialize_infrastructure()
        
        # Initialize system components
        await self._initialize_components()
        
        # Create bot configuration
        self._create_bot_config()
        
        # Create and initialize bot
        await self._initialize_bot()
        
        logger.info("✅ Telegram Bot Service initialized successfully")
    
    async def _initialize_infrastructure(self):
        """Initialize database and cache connections."""
        logger.info("📊 Connecting to database...")
        await startup_database()
        
        logger.info("📦 Connecting to Redis cache...")
        await get_redis()
        
        logger.info("✅ Infrastructure initialized")
    
    async def _initialize_components(self):
        """Initialize real system components."""
        logger.info("🔧 Initializing system components...")
        
        # Graph Engine - use provided instance
        if self.graph_engine and self.graph_engine.is_initialized:
            logger.info("✅ Using provided Graph Engine")
            self.components['graph'] = self.graph_engine.graph
            self.components['data_collector'] = self.graph_engine.data_collector
            self.components['pathfinder'] = self.graph_engine.pathfinder
        else:
            logger.warning("Graph engine not available - using database queries")
            from .adapters import DatabaseGraphAdapter
            self.components['graph'] = DatabaseGraphAdapter()
        
        # Initialize remaining components if not provided by Graph Engine
        if 'data_collector' not in self.components:
            logger.warning("Data collector not available - using fallback")
            from .adapters import MockDataCollector
            self.components['data_collector'] = MockDataCollector()
        
        if 'pathfinder' not in self.components:
            logger.warning("Pathfinder not available - using fallback")
            from .adapters import MockPathfinder
            self.components['pathfinder'] = MockPathfinder()
        
        # Simulator
        try:
            from yield_arbitrage.execution.hybrid_simulator import HybridSimulator
            self.components['simulator'] = HybridSimulator()
            logger.info("✅ Simulator initialized")
        except ImportError:
            logger.warning("Simulator not available")
            from .adapters import MockSimulator
            self.components['simulator'] = MockSimulator()
        
        # Initialize monitoring components using factory
        try:
            from yield_arbitrage.monitoring.factory import get_monitoring_components
            monitoring_components = await get_monitoring_components()
            
            # Add monitoring components to our components dict
            if monitoring_components['position_monitor']:
                self.components['position_monitor'] = monitoring_components['position_monitor']
                logger.info("✅ Position monitor initialized via factory")
            else:
                logger.warning("Position monitor not available from factory")
                from .adapters import DatabasePositionMonitor
                self.components['position_monitor'] = DatabasePositionMonitor()
            
            if monitoring_components['delta_tracker']:
                self.components['delta_tracker'] = monitoring_components['delta_tracker']
                logger.info("✅ Delta tracker initialized via factory")
            else:
                logger.warning("Delta tracker not available from factory")
                from .adapters import DatabaseDeltaTracker
                self.components['delta_tracker'] = DatabaseDeltaTracker()
            
            if monitoring_components['execution_logger']:
                self.components['execution_logger'] = monitoring_components['execution_logger']
                logger.info("✅ Execution logger initialized via factory")
            else:
                logger.warning("Execution logger not available from factory")
                from .adapters import DatabaseExecutionLogger
                self.components['execution_logger'] = DatabaseExecutionLogger()
                
        except Exception as e:
            logger.error(f"Failed to initialize monitoring components via factory: {e}")
            # Fallback to adapter implementations
            logger.warning("Using fallback adapter implementations")
            from .adapters import DatabasePositionMonitor, DatabaseDeltaTracker, DatabaseExecutionLogger
            self.components['position_monitor'] = DatabasePositionMonitor()
            self.components['delta_tracker'] = DatabaseDeltaTracker()
            self.components['execution_logger'] = DatabaseExecutionLogger()
        
        logger.info(f"✅ {len(self.components)} system components initialized")
    
    def _create_bot_config(self):
        """Create bot configuration from environment variables."""
        logger.info("⚙️  Creating bot configuration...")
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable required")
        
        allowed_users = os.getenv('TELEGRAM_ALLOWED_USERS', '')
        if not allowed_users:
            raise ValueError("TELEGRAM_ALLOWED_USERS environment variable required")
        
        admin_users = os.getenv('TELEGRAM_ADMIN_USERS', allowed_users)
        
        self.config = BotConfig(
            telegram_bot_token=bot_token,
            allowed_user_ids=[int(uid.strip()) for uid in allowed_users.split(',') if uid.strip()],
            admin_user_ids=[int(uid.strip()) for uid in admin_users.split(',') if uid.strip()],
            max_opportunities_displayed=int(os.getenv('TELEGRAM_MAX_OPPORTUNITIES', '10')),
            max_alerts_displayed=int(os.getenv('TELEGRAM_MAX_ALERTS', '20')),
            enable_position_monitoring=os.getenv('TELEGRAM_ENABLE_MONITORING', 'true').lower() == 'true',
            enable_execution_logging=os.getenv('TELEGRAM_ENABLE_LOGGING', 'true').lower() == 'true',
            enable_risk_alerts=os.getenv('TELEGRAM_ENABLE_RISK_ALERTS', 'true').lower() == 'true',
            alert_severity_threshold=os.getenv('TELEGRAM_ALERT_THRESHOLD', 'warning'),
            commands_per_minute=int(os.getenv('TELEGRAM_COMMANDS_PER_MINUTE', '60')),
            opportunities_cooldown_seconds=int(os.getenv('TELEGRAM_OPPORTUNITIES_COOLDOWN', '10')),
            status_cooldown_seconds=int(os.getenv('TELEGRAM_STATUS_COOLDOWN', '5'))
        )
        
        logger.info(f"✅ Bot configured for {len(self.config.allowed_user_ids)} users")
    
    async def _initialize_bot(self):
        """Initialize the Telegram bot with all components."""
        logger.info("🤖 Initializing Telegram bot...")
        
        self.bot = YieldArbitrageBot(
            config=self.config,
            **self.components
        )
        
        await self.bot.initialize()
        logger.info("✅ Telegram bot initialized")
    
    async def start(self):
        """Start the bot service."""
        if not self.bot:
            await self.initialize()
        
        logger.info("🚀 Starting Telegram bot service...")
        await self.bot.start()
    
    async def stop(self):
        """Stop the bot service and cleanup."""
        logger.info("🛑 Stopping Telegram bot service...")
        
        if self.bot:
            await self.bot.stop()
        
        # Cleanup infrastructure
        await shutdown_database()
        await close_redis()
        
        logger.info("✅ Telegram bot service stopped")


async def run_bot_service():
    """Run the Telegram bot service."""
    service = TelegramBotService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Service error: {e}", exc_info=True)
    finally:
        await service.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the service
    asyncio.run(run_bot_service())