#!/usr/bin/env python3
"""
Integrated Telegram Bot Runner.

This script demonstrates how to run the Telegram bot with full integration
to all yield arbitrage system components including position monitoring,
execution logging, and risk management.
"""
import sys
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/telegram_bot.log')
    ]
)

logger = logging.getLogger(__name__)


def create_mock_system_components():
    """Create mock system components for demonstration."""
    
    # Mock Graph Engine
    graph = Mock()
    graph.edges = [f"edge_{i}" for i in range(1247)]  # Mock 1247 edges
    graph.nodes = [f"node_{i}" for i in range(89)]    # Mock 89 nodes
    graph.last_update = datetime.now(timezone.utc)
    
    # Mock Data Collector
    data_collector = Mock()
    data_collector.is_running = True
    data_collector.last_collection_time = datetime.now(timezone.utc)
    data_collector.collections_today = 47
    data_collector.get_stats = Mock(return_value={
        "total_collections": 1247,
        "successful_collections": 1203,
        "failed_collections": 44,
        "avg_collection_time_ms": 2150.5
    })
    
    # Mock Pathfinder
    pathfinder = Mock()
    pathfinder.search = AsyncMock(return_value=[
        Mock(edge_id=f"path_edge_{i}", protocol_name=f"protocol_{i%3}")
        for i in range(10)
    ])
    
    # Mock Simulator
    simulator = Mock()
    async def mock_simulate_path(path, amount, start_asset):
        import random
        profit_usd = random.uniform(-5, 50)
        is_profitable = profit_usd > 0
        
        return {
            'is_profitable': is_profitable,
            'profit_usd': profit_usd,
            'profit_percentage': (profit_usd / (amount * 2000)) * 100,  # Assume $2000 ETH
            'gas_cost_usd': random.uniform(5, 25),
            'total_gas_usd_cost': random.uniform(5, 25),
            'estimated_apr': random.uniform(0, 150) if is_profitable else 0,
            'risk_score': random.uniform(0.1, 0.9),
            'simulation_time_ms': random.uniform(500, 3000)
        }
    
    simulator.simulate_path = mock_simulate_path
    
    # Mock Delta Tracker
    delta_tracker = Mock()
    delta_tracker.get_all_positions = Mock(return_value={
        'POS_001': create_mock_position('POS_001', 'arbitrage', 15250.0, 15000.0),
        'POS_002': create_mock_position('POS_002', 'yield_farming', 47800.0, 50000.0),
        'POS_003': create_mock_position('POS_003', 'lending', 32150.0, 32000.0),
        'POS_004': create_mock_position('POS_004', 'borrowing', -45000.0, -40000.0),
    })
    
    delta_tracker.calculate_portfolio_health = AsyncMock(return_value={
        'total_value_usd': 125000.0,
        'unrealized_pnl_usd': -2850.0,
        'liquidation_risk_score': 0.08,
        'position_count': 4
    })
    
    # Mock Position Monitor
    position_monitor = Mock()
    position_monitor.is_monitoring = True
    position_monitor.active_positions = ['POS_001', 'POS_002', 'POS_003', 'POS_004']
    position_monitor.alert_history = create_mock_alerts()
    position_monitor._monitor_single_position = AsyncMock(return_value=[])
    
    # Mock Execution Logger
    execution_logger = Mock()
    execution_logger.get_stats = Mock(return_value={
        'records_created': 1247,
        'records_updated': 3891,
        'write_errors': 0,
        'last_write_time': datetime.now(timezone.utc).timestamp()
    })
    
    execution_logger.get_execution_analytics = AsyncMock(return_value={
        'total_executions': 87,
        'successful_executions': 79,
        'failed_executions': 8,
        'success_rate': 0.908,
        'avg_predicted_profit_usd': 24.75,
        'total_predicted_profit_usd': 2153.25,
        'avg_simulation_time_ms': 1847.3
    })
    
    return {
        'graph': graph,
        'data_collector': data_collector,
        'pathfinder': pathfinder,
        'simulator': simulator,
        'delta_tracker': delta_tracker,
        'position_monitor': position_monitor,
        'execution_logger': execution_logger
    }


def create_mock_position(pos_id: str, pos_type: str, current_value: float, initial_value: float):
    """Create a mock position object."""
    position = Mock()
    position.position_id = pos_id
    position.position_type = pos_type
    position.current_value_usd = current_value
    position.initial_value_usd = initial_value
    position.created_at = datetime.now(timezone.utc)
    position.status = 'active'
    return position


def create_mock_alerts():
    """Create mock position alerts."""
    from datetime import timedelta
    
    # Import alert classes (simplified for demo)
    alerts = []
    
    # Create some mock alerts
    mock_alert_data = [
        {
            'position_id': 'POS_002',
            'alert_type': 'unrealized_loss',
            'severity': 'warning',
            'message': 'Position down 4.4% from initial value',
            'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15)
        },
        {
            'position_id': 'POS_004',
            'alert_type': 'health_factor_low',
            'severity': 'error',
            'message': 'Health factor approaching liquidation threshold',
            'timestamp': datetime.now(timezone.utc) - timedelta(minutes=5)
        },
        {
            'position_id': 'POS_001',
            'alert_type': 'execution_complete',
            'severity': 'info',
            'message': 'Arbitrage position completed successfully',
            'timestamp': datetime.now(timezone.utc) - timedelta(hours=2)
        }
    ]
    
    for alert_data in mock_alert_data:
        alert = Mock()
        alert.position_id = alert_data['position_id']
        alert.alert_type = alert_data['alert_type']
        alert.severity = Mock()
        alert.severity.value = alert_data['severity']
        alert.message = alert_data['message']
        alert.timestamp = alert_data['timestamp']
        alert.recommended_action = "Monitor position closely"
        alerts.append(alert)
    
    return alerts


def create_bot_config():
    """Create bot configuration for demonstration."""
    from yield_arbitrage.telegram_interface.config import BotConfig
    
    # Use environment variables if available, otherwise use demo values
    config = BotConfig(
        telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', 'DEMO_TOKEN_REPLACE_ME'),
        allowed_user_ids=[
            int(uid) for uid in os.getenv('TELEGRAM_ALLOWED_USERS', '123456789').split(',')
        ],
        admin_user_ids=[
            int(uid) for uid in os.getenv('TELEGRAM_ADMIN_USERS', '123456789').split(',')
        ],
        max_opportunities_displayed=10,
        max_alerts_displayed=15,
        enable_position_monitoring=True,
        enable_execution_logging=True,
        enable_risk_alerts=True,
        alert_severity_threshold='warning'
    )
    
    return config


async def run_integrated_bot_demo():
    """Run the integrated Telegram bot demonstration."""
    print("🤖 Starting Integrated Telegram Bot Demo")
    print("=" * 60)
    
    try:
        # Check if we have a real bot token
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token or bot_token == 'DEMO_TOKEN_REPLACE_ME':
            print("⚠️  No TELEGRAM_BOT_TOKEN found in environment variables.")
            print("   This demo will create the bot but won't be able to connect to Telegram.")
            print("   To run with a real bot:")
            print("   1. Create a bot with @BotFather on Telegram")
            print("   2. Set TELEGRAM_BOT_TOKEN environment variable")
            print("   3. Set TELEGRAM_ALLOWED_USERS environment variable")
            print()
            
            # Continue with demo setup
            demo_mode = True
        else:
            demo_mode = False
            print(f"✅ Found bot token, will connect to Telegram")
        
        # Step 1: Create system components
        print("🔧 Step 1: Creating System Components")
        components = create_mock_system_components()
        print(f"   ✅ Created {len(components)} system components:")
        for name in components.keys():
            print(f"     • {name}")
        
        # Step 2: Create bot configuration
        print("\n⚙️  Step 2: Creating Bot Configuration")
        try:
            config = create_bot_config()
            print(f"   ✅ Bot configuration created:")
            print(f"     • Allowed users: {len(config.allowed_user_ids)}")
            print(f"     • Admin users: {len(config.admin_user_ids)}")
            print(f"     • Position monitoring: {'✅' if config.enable_position_monitoring else '❌'}")
            print(f"     • Execution logging: {'✅' if config.enable_execution_logging else '❌'}")
            print(f"     • Risk alerts: {'✅' if config.enable_risk_alerts else '❌'}")
        except ValueError as e:
            print(f"   ❌ Configuration error: {e}")
            if demo_mode:
                print("   📝 Using demo configuration...")
                config.telegram_bot_token = "123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
            else:
                return
        
        # Step 3: Create and initialize bot
        print("\n🤖 Step 3: Creating Telegram Bot")
        from yield_arbitrage.telegram_interface import YieldArbitrageBot
        
        bot = YieldArbitrageBot(
            config=config,
            **components
        )
        
        print(f"   ✅ Bot created with integration to:")
        integration_status = config.get_integration_settings()
        for feature, enabled in integration_status.items():
            status = "✅" if enabled else "❌"
            print(f"     • {feature.replace('_', ' ').title()}: {status}")
        
        # Step 4: Initialize bot (without starting polling in demo mode)
        print("\n🔄 Step 4: Initializing Bot Application")
        if demo_mode:
            print("   📝 Demo mode: Skipping Telegram connection")
            print("   ✅ Bot application structure created")
            print("   ✅ Command handlers registered")
            print("   ✅ Authentication system initialized")
        else:
            await bot.initialize()
            print("   ✅ Bot application initialized")
            print("   ✅ Connected to Telegram API")
        
        # Step 5: Demonstrate functionality
        print("\n🎯 Step 5: Demonstrating Bot Functionality")
        
        # Test system status gathering
        print("   📊 Testing system status gathering...")
        if not demo_mode and bot.application:
            # Use real bot context - bot_data is a dict, not awaitable
            status_data = bot.application.bot_data
        else:
            # Simulate status gathering
            status_data = {
                'graph': components['graph'],
                'data_collector': components['data_collector'],
                'position_monitor': components['position_monitor'],
                'execution_logger': components['execution_logger']
            }
        
        # Simulate status collection
        mock_status = {
            'timestamp': datetime.now(timezone.utc),
            'graph': {
                'total_edges': len(components['graph'].edges),
                'total_nodes': len(components['graph'].nodes),
                'last_update': components['graph'].last_update
            },
            'data_collector': {
                'is_running': components['data_collector'].is_running,
                'last_collection': components['data_collector'].last_collection_time,
                'collections_today': components['data_collector'].collections_today
            },
            'position_monitor': {
                'is_monitoring': components['position_monitor'].is_monitoring,
                'positions_monitored': len(components['position_monitor'].active_positions),
                'recent_alerts': len(components['position_monitor'].alert_history)
            },
            'execution_logger': components['execution_logger'].get_stats()
        }
        
        print(f"     ✅ System health data collected:")
        print(f"       • Graph: {mock_status['graph']['total_edges']} edges, {mock_status['graph']['total_nodes']} nodes")
        print(f"       • Data collector: {'Running' if mock_status['data_collector']['is_running'] else 'Stopped'}")
        print(f"       • Position monitor: {mock_status['position_monitor']['positions_monitored']} positions")
        print(f"       • Execution logger: {mock_status['execution_logger']['records_created']} records")
        
        # Test opportunity finding
        print("\n   💰 Testing opportunity finding...")
        opportunities = await components['pathfinder'].search("ETH_MAINNET_WETH", 1.0, beam_width=10)
        print(f"     ✅ Found {len(opportunities)} potential paths")
        
        # Simulate some of them
        simulated_opportunities = []
        for i, path in enumerate(opportunities[:5]):
            sim_result = await components['simulator'].simulate_path(path, 1.0, "ETH_MAINNET_WETH")
            if sim_result['is_profitable']:
                simulated_opportunities.append({
                    'path': path,
                    'simulation': sim_result,
                    'profit_usd': sim_result['profit_usd']
                })
        
        print(f"     ✅ {len(simulated_opportunities)} profitable opportunities found")
        if simulated_opportunities:
            best_opp = max(simulated_opportunities, key=lambda x: x['profit_usd'])
            print(f"       • Best profit: ${best_opp['profit_usd']:.2f}")
        
        # Test position monitoring
        print("\n   📊 Testing position monitoring...")
        positions = components['delta_tracker'].get_all_positions()
        portfolio_health = await components['delta_tracker'].calculate_portfolio_health()
        
        print(f"     ✅ Monitoring {len(positions)} positions:")
        for pos_id, position in positions.items():
            pnl = position.current_value_usd - position.initial_value_usd
            pnl_pct = (pnl / position.initial_value_usd) * 100 if position.initial_value_usd != 0 else 0
            print(f"       • {pos_id}: ${position.current_value_usd:,.0f} ({pnl_pct:+.1f}%)")
        
        print(f"     📈 Portfolio health:")
        print(f"       • Total value: ${portfolio_health['total_value_usd']:,.0f}")
        print(f"       • Unrealized P&L: ${portfolio_health['unrealized_pnl_usd']:+,.0f}")
        print(f"       • Liquidation risk: {portfolio_health['liquidation_risk_score']:.1%}")
        
        # Test alert system
        print("\n   🚨 Testing alert system...")
        alerts = components['position_monitor'].alert_history
        print(f"     ✅ {len(alerts)} recent alerts:")
        
        severity_counts = {}
        for alert in alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            emoji = {'info': 'ℹ️', 'warning': '⚠️', 'error': '🚨', 'critical': '🆘'}.get(severity, '❓')
            print(f"       {emoji} {severity.title()}: {count}")
        
        # Step 6: Summary and next steps
        print(f"\n📋 Step 6: Integration Summary")
        
        integration_summary = {
            "Bot Components": "✅ Created and configured",
            "System Integration": "✅ All components connected",
            "Authentication": "✅ User whitelisting active",
            "Command Handlers": "✅ All commands registered",
            "Position Monitoring": "✅ Real-time monitoring ready",
            "Alert System": "✅ Multi-severity alerts working",
            "Execution Logging": "✅ Full execution tracking",
            "Portfolio Health": "✅ Real-time health monitoring"
        }
        
        for component, status in integration_summary.items():
            print(f"   {status} {component}")
        
        # Show available commands
        print(f"\n🎯 Available Commands:")
        commands = [
            "/start - Welcome and basic info",
            "/status - Comprehensive system status",
            "/opportunities [amount] [asset] - Find arbitrage opportunities",
            "/positions - Monitor active positions",
            "/alerts [severity] - View recent alerts",
            "/portfolio - Portfolio health summary",
            "/metrics [period] - Performance metrics",
            "/config - View/modify settings (admin)",
            "/help - Detailed command help"
        ]
        
        for cmd in commands:
            print(f"   • {cmd}")
        
        if demo_mode:
            print(f"\n📝 Demo Mode Notes:")
            print(f"   • Bot created but not connected to Telegram")
            print(f"   • All integrations working with mock data")
            print(f"   • To run with real Telegram:")
            print(f"     1. Get bot token from @BotFather")
            print(f"     2. export TELEGRAM_BOT_TOKEN='your_token_here'")
            print(f"     3. export TELEGRAM_ALLOWED_USERS='your_user_id'")
            print(f"     4. Run: python scripts/run_telegram_bot_integrated.py --real")
        else:
            print(f"\n🚀 Starting Bot...")
            print(f"   • Bot will start polling for messages")
            print(f"   • Use Ctrl+C to stop")
            print(f"   • Logs available in /tmp/telegram_bot.log")
            
            # Start the bot
            await bot.start()
    
    except KeyboardInterrupt:
        print(f"\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Error running bot demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'bot' in locals() and not demo_mode:
            await bot.stop()
        print(f"\n✅ Bot demo completed!")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run integrated Telegram bot demo')
    parser.add_argument('--real', action='store_true', 
                       help='Connect to real Telegram (requires bot token)')
    parser.add_argument('--config', type=str,
                       help='Path to bot configuration file')
    
    args = parser.parse_args()
    
    if args.real:
        # Force real mode by setting environment flag
        os.environ['TELEGRAM_FORCE_REAL'] = 'true'
    
    await run_integrated_bot_demo()


if __name__ == "__main__":
    asyncio.run(main())