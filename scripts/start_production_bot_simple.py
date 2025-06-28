#!/usr/bin/env python3
"""
Simplified Production Telegram Bot.

This script starts the Telegram bot with database integration
but handles missing system components gracefully.
"""
import sys
import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/telegram_bot_production.log')
    ]
)

logger = logging.getLogger(__name__)


async def create_production_components():
    """Create production-ready components with graceful fallbacks."""
    components = {}
    
    # Initialize database connection
    try:
        logger.info("🗄️  Initializing database...")
        from yield_arbitrage.database.connection import startup_database
        await startup_database()
        logger.info("✅ Database connected")
        
        # Use database adapters for real data
        from yield_arbitrage.telegram_interface.adapters import (
            DatabaseGraphAdapter,
            DatabasePositionMonitor, 
            DatabaseDeltaTracker,
            DatabaseExecutionLogger
        )
        
        components['graph'] = DatabaseGraphAdapter()
        components['position_monitor'] = DatabasePositionMonitor()
        components['delta_tracker'] = DatabaseDeltaTracker()
        components['execution_logger'] = DatabaseExecutionLogger()
        
        # Refresh graph data
        await components['graph'].refresh_data()
        
        logger.info("✅ Database adapters initialized")
        
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        logger.info("Using mock components as fallback")
        
        from unittest.mock import Mock, AsyncMock
        
        components['graph'] = Mock(
            edges=[f"edge_{i}" for i in range(100)],
            nodes=[f"node_{i}" for i in range(20)],
            last_update=datetime.now(timezone.utc)
        )
        
        components['position_monitor'] = Mock(
            is_monitoring=False,
            active_positions=[],
            alert_history=[]
        )
        
        components['delta_tracker'] = Mock()
        components['delta_tracker'].get_all_positions = Mock(return_value={})
        components['delta_tracker'].calculate_portfolio_health = AsyncMock(return_value={
            'total_value_usd': 0.0,
            'unrealized_pnl_usd': 0.0,
            'liquidation_risk_score': 0.0,
            'position_count': 0
        })
        
        components['execution_logger'] = Mock()
        components['execution_logger'].get_stats = Mock(return_value={
            'records_created': 0,
            'records_updated': 0,
            'write_errors': 0
        })
        components['execution_logger'].get_execution_analytics = AsyncMock(return_value={
            'total_executions': 0,
            'successful_executions': 0,
            'success_rate': 0.0,
            'avg_predicted_profit_usd': 0.0
        })
    
    # Add mock pathfinder and simulator (these will be implemented later)
    from unittest.mock import Mock, AsyncMock
    
    components['pathfinder'] = Mock()
    components['pathfinder'].search = AsyncMock(return_value=[
        Mock(edges=[f'edge_{i}']) for i in range(3)
    ])
    
    components['simulator'] = Mock()
    components['simulator'].simulate_path = AsyncMock(return_value={
        'is_profitable': False,
        'profit_usd': 0.0,
        'gas_cost_usd': 15.0,
        'profit_percentage': 0.0,
        'estimated_apr': 0.0,
        'risk_score': 0.5
    })
    
    components['data_collector'] = Mock(
        is_running=False,
        last_collection_time=None,
        collections_today=0
    )
    
    return components


async def main():
    """Main entry point for production bot."""
    print("🚀 Starting Production Telegram Bot (Simplified)")
    print("=" * 55)
    
    try:
        # Check environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        allowed_users = os.getenv('TELEGRAM_ALLOWED_USERS')
        
        if not bot_token:
            print("❌ TELEGRAM_BOT_TOKEN not found in environment")
            return
        
        if not allowed_users:
            print("❌ TELEGRAM_ALLOWED_USERS not found in environment")
            return
        
        print(f"✅ Bot token found")
        print(f"✅ Allowed users: {allowed_users}")
        
        # Create components
        print("\n🔧 Initializing Components...")
        components = await create_production_components()
        print(f"✅ {len(components)} components ready")
        
        # Create bot configuration
        print("\n⚙️  Creating Bot Configuration...")
        from yield_arbitrage.telegram_interface.config import BotConfig
        
        config = BotConfig(
            telegram_bot_token=bot_token,
            allowed_user_ids=[int(uid.strip()) for uid in allowed_users.split(',')],
            admin_user_ids=[int(uid.strip()) for uid in os.getenv('TELEGRAM_ADMIN_USERS', allowed_users).split(',')],
            max_opportunities_displayed=10,
            max_alerts_displayed=20,
            enable_position_monitoring=True,
            enable_execution_logging=True,
            enable_risk_alerts=True
        )
        
        print(f"✅ Bot configured for {len(config.allowed_user_ids)} users")
        
        # Create and start bot
        print("\n🤖 Starting Telegram Bot...")
        from yield_arbitrage.telegram_interface.bot import YieldArbitrageBot
        
        bot = YieldArbitrageBot(config=config, **components)
        
        # Start bot
        print("🚀 Bot starting... (Use Ctrl+C to stop)")
        await bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        print(f"❌ Error: {e}")
    finally:
        try:
            if 'bot' in locals():
                await bot.stop()
            
            # Cleanup database if connected
            try:
                from yield_arbitrage.database.connection import shutdown_database
                await shutdown_database()
            except:
                pass
                
            print("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Import datetime here to avoid import issues
    from datetime import datetime, timezone
    
    asyncio.run(main())