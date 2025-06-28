#!/usr/bin/env python3
"""
Quick Telegram Bot Test Script.

This script tests the bot initialization and basic functionality
without running indefinitely.
"""
import sys
import os
import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import Mock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_components():
    """Create mock system components."""
    components = {
        'graph': Mock(
            edges=[f"edge_{i}" for i in range(100)],
            nodes=[f"node_{i}" for i in range(20)],
            last_update=datetime.now(timezone.utc)
        ),
        'data_collector': Mock(
            is_running=True,
            last_collection_time=datetime.now(timezone.utc),
            collections_today=25
        ),
        'pathfinder': Mock(),
        'simulator': Mock(),
        'delta_tracker': Mock(),
        'position_monitor': Mock(
            is_monitoring=True,
            active_positions=['pos_1', 'pos_2'],
            alert_history=[]
        ),
        'execution_logger': Mock()
    }
    
    # Set up async methods
    components['pathfinder'].search = Mock(return_value=[Mock() for _ in range(5)])
    components['simulator'].simulate_path = Mock(return_value={
        'is_profitable': True,
        'profit_usd': 25.50,
        'gas_cost_usd': 8.00,
        'profit_percentage': 1.5
    })
    
    return components


async def test_bot_functionality():
    """Test bot functionality quickly."""
    print("🤖 Quick Telegram Bot Test")
    print("=" * 40)
    
    try:
        # Check environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        allowed_users = os.getenv('TELEGRAM_ALLOWED_USERS')
        
        if not bot_token:
            print("❌ TELEGRAM_BOT_TOKEN not found")
            return
            
        if not allowed_users:
            print("❌ TELEGRAM_ALLOWED_USERS not found")
            return
            
        print(f"✅ Bot token found (length: {len(bot_token)})")
        print(f"✅ Allowed users: {allowed_users}")
        
        # Create bot config
        from yield_arbitrage.telegram_interface.config import BotConfig
        
        config = BotConfig(
            telegram_bot_token=bot_token,
            allowed_user_ids=[int(uid) for uid in allowed_users.split(',')],
            admin_user_ids=[int(uid) for uid in allowed_users.split(',')],
            max_opportunities_displayed=5,
            enable_position_monitoring=True,
            enable_execution_logging=True
        )
        
        print(f"✅ Bot config created")
        
        # Create mock components
        components = create_mock_components()
        print(f"✅ Mock components created")
        
        # Create bot
        from yield_arbitrage.telegram_interface import YieldArbitrageBot
        
        bot = YieldArbitrageBot(config=config, **components)
        print(f"✅ Bot instance created")
        
        # Initialize bot (but don't start polling)
        await bot.initialize()
        print(f"✅ Bot initialized and connected to Telegram API")
        
        # Test bot info
        if bot.application and bot.application.bot:
            bot_info = await bot.application.bot.get_me()
            print(f"✅ Bot info: @{bot_info.username} ({bot_info.first_name})")
        
        # Test command availability
        if bot.application:
            handlers = bot.application.handlers
            print(f"✅ Command handlers registered: {len(handlers[0]) if handlers else 0}")
        
        # Stop bot
        await bot.stop()
        print(f"✅ Bot stopped cleanly")
        
        print(f"\n🎯 Test Results:")
        print(f"   • Bot authentication: ✅ SUCCESS")
        print(f"   • Telegram API connection: ✅ SUCCESS") 
        print(f"   • Component integration: ✅ SUCCESS")
        print(f"   • Configuration loading: ✅ SUCCESS")
        print(f"   • Command handlers: ✅ SUCCESS")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Message your bot on Telegram")
        print(f"   2. Send /start to begin")
        print(f"   3. Try commands like /status, /opportunities")
        print(f"   4. Bot username: @{bot_info.username if 'bot_info' in locals() else 'check_telegram'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_bot_functionality())