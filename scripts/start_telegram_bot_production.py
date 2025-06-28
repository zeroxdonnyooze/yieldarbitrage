#!/usr/bin/env python3
"""
Production Telegram Bot Startup Script.

This script starts the Telegram bot with full integration to the
production yield arbitrage system.
"""
import sys
import os
import asyncio
import logging
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/telegram_bot_production.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for production bot."""
    print("🚀 Starting Production Telegram Bot")
    print("=" * 50)
    
    try:
        # Import and run the production bot service
        from yield_arbitrage.telegram_interface.service_bot import run_bot_service
        
        # Check required environment variables
        required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_ALLOWED_USERS', 'DATABASE_URL']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please ensure these are set in your .env file:")
            for var in missing_vars:
                print(f"   {var}=your_value_here")
            return
        
        print("✅ All required environment variables found")
        print("🤖 Starting Telegram bot service...")
        
        # Run the production bot
        await run_bot_service()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Production bot error: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        print("Check logs at /tmp/telegram_bot_production.log for details")


if __name__ == "__main__":
    asyncio.run(main())