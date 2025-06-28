"""
Telegram Bot Command Handlers.

This module implements all command handlers for the Telegram bot interface,
integrating with the yield arbitrage system components.
"""
import asyncio
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

from telegram import Update
from telegram.ext import ContextTypes

from .auth import auth_required, admin_required, UserSession
from .formatters import (
    format_system_status, format_opportunities, format_position_alerts,
    format_execution_metrics, format_portfolio_health, format_config_display
)

logger = logging.getLogger(__name__)


@auth_required
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command - welcome message and basic info."""
    session: UserSession = context.user_data['session']
    
    welcome_text = f"""
🚀 **Yield Arbitrage Bot**

Welcome, {session.first_name or session.username or 'User'}!

Available commands:
• `/status` - System health and statistics
• `/opportunities` - Current arbitrage opportunities  
• `/positions` - Monitor active positions
• `/alerts` - Recent alerts and notifications
• `/config` - View and modify settings
• `/metrics` - Performance metrics
• `/help` - Detailed command help

Type `/help` for more information about each command.

🔒 You are {'an admin' if session.is_admin else 'a regular'} user.
"""
    
    await update.message.reply_text(welcome_text, parse_mode='Markdown')


@auth_required
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command - detailed command information."""
    session: UserSession = context.user_data['session']
    
    help_text = """
📚 **Command Reference**

**🔍 Monitoring Commands:**
• `/status` - System health, graph stats, data collector status
• `/opportunities [amount] [asset]` - Find profitable arbitrage paths
• `/positions` - Active position monitoring and health
• `/alerts [severity]` - Recent alerts (info/warning/error/critical)

**⚙️ Configuration Commands:**
• `/config` - View current configuration
• `/config set <param> <value>` - Update parameter (admin only)
• `/config reset` - Reset to defaults (admin only)

**📊 Analytics Commands:**
• `/metrics [period]` - Performance metrics (24h/7d/30d)
• `/portfolio` - Portfolio health summary
• `/history [command]` - Command usage history

**💰 Opportunity Parameters:**
• Amount: Trading amount (default: 1.0)
• Asset: Start asset (default: WETH)
• Examples:
  - `/opportunities` (1 WETH)
  - `/opportunities 5.0` (5 WETH)  
  - `/opportunities 1000 USDC` (1000 USDC)

**🚨 Alert Filters:**
• `/alerts` - All recent alerts
• `/alerts warning` - Warning+ alerts only
• `/alerts critical` - Critical alerts only

**⏱️ Rate Limits:**
• General: 60 commands/minute
• Opportunities: 10 second cooldown
• Status: 5 second cooldown
"""
    
    if session.is_admin:
        help_text += """
**🔧 Admin Commands:**
• `/users` - Manage user access
• `/block <user_id>` - Block user
• `/unblock <user_id>` - Unblock user
• `/system restart` - Restart components
• `/system stop` - Emergency stop
"""
    
    await update.message.reply_text(help_text, parse_mode='Markdown')


@auth_required
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - comprehensive system status."""
    try:
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Gather system status from all components
        status_data = await _gather_system_status(context)
        
        # Format and send status message
        status_text = format_system_status(status_data)
        await update.message.reply_text(status_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in status_command: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Error retrieving system status: {str(e)}"
        )


@auth_required
async def opportunities_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /opportunities command - find and display profitable paths."""
    try:
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Parse command arguments
        args = context.args
        amount = 1.0  # Default amount
        start_asset = "ETH_MAINNET_WETH"  # Default asset
        
        if args:
            try:
                amount = float(args[0])
                if len(args) > 1:
                    # Handle asset specification
                    asset_input = args[1].upper()
                    start_asset = _parse_asset_input(asset_input)
            except ValueError:
                await update.message.reply_text(
                    "❌ Invalid amount. Please provide a valid number.\n"
                    "Example: `/opportunities 5.0` or `/opportunities 1000 USDC`"
                )
                return
        
        # Find opportunities using pathfinder
        opportunities = await _find_opportunities(context, start_asset, amount)
        
        if not opportunities:
            await update.message.reply_text(
                f"🔍 No profitable opportunities found for {amount} {_get_asset_symbol(start_asset)}\n"
                f"Current market conditions may not support profitable arbitrage."
            )
            return
        
        # Format and send opportunities
        config = context.bot_data.get('bot_config')
        max_opportunities = config.max_opportunities_displayed if config else 10
        
        opportunities_text = format_opportunities(
            opportunities[:max_opportunities], 
            amount, 
            start_asset
        )
        
        await update.message.reply_text(opportunities_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in opportunities_command: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Error finding opportunities: {str(e)}"
        )


@auth_required
async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /positions command - monitor active positions."""
    try:
        # Check if position monitoring is enabled
        config = context.bot_data.get('bot_config')
        if not config or not config.enable_position_monitoring:
            await update.message.reply_text(
                "📊 Position monitoring is currently disabled."
            )
            return
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Get position monitor
        position_monitor = context.bot_data.get('position_monitor')
        if not position_monitor:
            await update.message.reply_text(
                "❌ Position monitor not available."
            )
            return
        
        # Get active positions and their health
        positions_data = await _get_positions_data(context)
        
        if not positions_data['positions']:
            await update.message.reply_text(
                "📊 No active positions to monitor.\n"
                "Positions will appear here when arbitrage opportunities are executed."
            )
            return
        
        # Format position information
        positions_text = await _format_positions_summary(positions_data)
        await update.message.reply_text(positions_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in positions_command: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Error retrieving positions: {str(e)}"
        )


@auth_required
async def alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /alerts command - show recent alerts and notifications."""
    try:
        # Parse severity filter
        severity_filter = None
        if context.args:
            severity_filter = context.args[0].lower()
            valid_severities = ['info', 'warning', 'error', 'critical']
            if severity_filter not in valid_severities:
                await update.message.reply_text(
                    f"❌ Invalid severity filter. Use one of: {', '.join(valid_severities)}"
                )
                return
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Get recent alerts
        alerts_data = await _get_recent_alerts(context, severity_filter)
        
        if not alerts_data['alerts']:
            filter_text = f" with severity '{severity_filter}'" if severity_filter else ""
            await update.message.reply_text(
                f"✅ No recent alerts{filter_text}.\n"
                "The system is operating normally."
            )
            return
        
        # Format alerts
        config = context.bot_data.get('bot_config')
        max_alerts = config.max_alerts_displayed if config else 20
        
        alerts_text = format_position_alerts(
            alerts_data['alerts'][:max_alerts],
            severity_filter
        )
        
        await update.message.reply_text(alerts_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in alerts_command: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Error retrieving alerts: {str(e)}"
        )


@auth_required
async def metrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /metrics command - show performance metrics."""
    try:
        # Parse time period
        period = "24h"  # default
        if context.args:
            period_input = context.args[0].lower()
            valid_periods = ["1h", "24h", "7d", "30d"]
            if period_input in valid_periods:
                period = period_input
            else:
                await update.message.reply_text(
                    f"❌ Invalid time period. Use one of: {', '.join(valid_periods)}"
                )
                return
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Gather metrics from execution logger and other components
        metrics_data = await _gather_performance_metrics(context, period)
        
        # Format metrics
        metrics_text = format_execution_metrics(metrics_data, period)
        await update.message.reply_text(metrics_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in metrics_command: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Error retrieving metrics: {str(e)}"
        )


@auth_required
async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /config command - view and modify configuration."""
    session: UserSession = context.user_data['session']
    
    try:
        if not context.args:
            # Display current configuration
            config = context.bot_data.get('bot_config')
            if not config:
                await update.message.reply_text("❌ Configuration not available.")
                return
            
            config_text = format_config_display(config, session.is_admin)
            await update.message.reply_text(config_text, parse_mode='Markdown')
            return
        
        # Handle configuration changes (admin only)
        if not session.is_admin:
            await update.message.reply_text(
                "❌ Admin privileges required to modify configuration."
            )
            return
        
        action = context.args[0].lower()
        
        if action == "set" and len(context.args) >= 3:
            param = context.args[1]
            value = context.args[2]
            
            success = await _update_config_parameter(context, param, value)
            if success:
                await update.message.reply_text(
                    f"✅ Configuration updated: {param} = {value}"
                )
            else:
                await update.message.reply_text(
                    f"❌ Failed to update parameter '{param}'"
                )
        
        elif action == "reset":
            success = await _reset_configuration(context)
            if success:
                await update.message.reply_text("✅ Configuration reset to defaults")
            else:
                await update.message.reply_text("❌ Failed to reset configuration")
        
        else:
            await update.message.reply_text(
                "❌ Invalid config command.\n"
                "Usage: `/config set <param> <value>` or `/config reset`"
            )
    
    except Exception as e:
        logger.error(f"Error in config_command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error handling config command: {str(e)}")


@auth_required
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /portfolio command - portfolio health summary."""
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Get portfolio health data
        portfolio_data = await _get_portfolio_health(context)
        
        # Format portfolio summary
        portfolio_text = format_portfolio_health(portfolio_data)
        await update.message.reply_text(portfolio_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in portfolio_command: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Error retrieving portfolio data: {str(e)}"
        )


# Admin commands
@admin_required
async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /users command - manage user access (admin only)."""
    try:
        authenticator = context.bot_data.get('authenticator')
        if not authenticator:
            await update.message.reply_text("❌ Authentication system not available")
            return
        
        stats = authenticator.get_session_stats()
        
        users_text = f"""
👥 **User Management**

**Statistics:**
• Total sessions: {stats['total_sessions']}
• Active sessions: {stats['active_sessions']}
• Blocked users: {stats['blocked_users']}
• Commands processed: {stats['total_commands_processed']}
• Failed auth attempts: {stats['failed_auth_attempts']}

**Commands:**
• `/block <user_id>` - Block a user
• `/unblock <user_id>` - Unblock a user
• `/cleanup` - Clean old sessions
"""
        
        await update.message.reply_text(users_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in users_command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)}")


# Helper functions
async def _gather_system_status(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    """Gather comprehensive system status from all components."""
    status = {
        'timestamp': datetime.now(timezone.utc),
        'uptime': None,
        'graph': {},
        'data_collector': {},
        'pathfinder': {},
        'position_monitor': {},
        'execution_logger': {},
        'database': {}
    }
    
    try:
        # Graph statistics
        graph = context.bot_data.get('graph')
        if graph:
            status['graph'] = {
                'total_edges': len(getattr(graph, 'edges', [])),
                'total_nodes': len(getattr(graph, 'nodes', [])),
                'last_update': getattr(graph, 'last_update', None)
            }
        
        # Data collector status
        data_collector = context.bot_data.get('data_collector')
        if data_collector:
            status['data_collector'] = {
                'is_running': getattr(data_collector, 'is_running', False),
                'last_collection': getattr(data_collector, 'last_collection_time', None),
                'collections_today': getattr(data_collector, 'collections_today', 0)
            }
        
        # Position monitor status
        position_monitor = context.bot_data.get('position_monitor')
        if position_monitor:
            status['position_monitor'] = {
                'is_monitoring': getattr(position_monitor, 'is_monitoring', False),
                'positions_monitored': len(getattr(position_monitor, 'active_positions', [])),
                'recent_alerts': len(getattr(position_monitor, 'alert_history', []))
            }
        
        # Execution logger stats
        execution_logger = context.bot_data.get('execution_logger')
        if execution_logger:
            logger_stats = execution_logger.get_stats() if hasattr(execution_logger, 'get_stats') else {}
            status['execution_logger'] = logger_stats
    
    except Exception as e:
        logger.error(f"Error gathering system status: {e}")
        status['error'] = str(e)
    
    return status


async def _find_opportunities(context: ContextTypes.DEFAULT_TYPE, start_asset: str, amount: float) -> List[Dict[str, Any]]:
    """Find profitable arbitrage opportunities."""
    pathfinder = context.bot_data.get('pathfinder')
    simulator = context.bot_data.get('simulator')
    
    if not pathfinder or not simulator:
        return []
    
    try:
        # Search for paths
        paths = await pathfinder.search(start_asset, amount, beam_width=20)
        
        opportunities = []
        for path in paths[:15]:  # Limit to top 15 for simulation
            try:
                # Simulate path profitability
                sim_result = await simulator.simulate_path(path, amount, start_asset)
                
                if sim_result.get('is_profitable', False):
                    opportunities.append({
                        'path': path,
                        'simulation': sim_result,
                        'profit_usd': sim_result.get('profit_usd', 0),
                        'profit_percentage': sim_result.get('profit_percentage', 0),
                        'gas_cost_usd': sim_result.get('gas_cost_usd', 0),
                        'estimated_apr': sim_result.get('estimated_apr', 0),
                        'risk_score': sim_result.get('risk_score', 0.5)
                    })
            except Exception as e:
                logger.warning(f"Failed to simulate path: {e}")
                continue
        
        # Sort by profitability
        opportunities.sort(key=lambda x: x['profit_usd'], reverse=True)
        return opportunities
        
    except Exception as e:
        logger.error(f"Error finding opportunities: {e}")
        return []


async def _get_positions_data(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    """Get comprehensive positions data."""
    delta_tracker = context.bot_data.get('delta_tracker')
    position_monitor = context.bot_data.get('position_monitor')
    
    positions_data = {
        'positions': [],
        'portfolio_health': {},
        'total_value_usd': 0,
        'unrealized_pnl_usd': 0
    }
    
    if not delta_tracker:
        return positions_data
    
    try:
        # Get all positions from delta tracker
        all_positions = delta_tracker.get_all_positions()
        
        for position_id, position in all_positions.items():
            position_info = {
                'position_id': position_id,
                'position_type': getattr(position, 'position_type', 'unknown'),
                'current_value_usd': getattr(position, 'current_value_usd', 0),
                'initial_value_usd': getattr(position, 'initial_value_usd', 0),
                'unrealized_pnl_usd': 0,
                'health_status': 'unknown',
                'alerts': []
            }
            
            # Calculate P&L
            if position_info['initial_value_usd'] > 0:
                position_info['unrealized_pnl_usd'] = (
                    position_info['current_value_usd'] - position_info['initial_value_usd']
                )
            
            # Get health information from position monitor
            if position_monitor:
                try:
                    alerts = await position_monitor._monitor_single_position(position)
                    position_info['alerts'] = alerts
                    
                    # Determine health status from alerts
                    if any(a.severity.value == 'critical' for a in alerts):
                        position_info['health_status'] = 'critical'
                    elif any(a.severity.value == 'error' for a in alerts):
                        position_info['health_status'] = 'error'
                    elif any(a.severity.value == 'warning' for a in alerts):
                        position_info['health_status'] = 'warning'
                    else:
                        position_info['health_status'] = 'healthy'
                except Exception as e:
                    logger.warning(f"Failed to get health for position {position_id}: {e}")
            
            positions_data['positions'].append(position_info)
        
        # Calculate portfolio totals
        positions_data['total_value_usd'] = sum(p['current_value_usd'] for p in positions_data['positions'])
        positions_data['unrealized_pnl_usd'] = sum(p['unrealized_pnl_usd'] for p in positions_data['positions'])
        
        # Get portfolio health if available
        if hasattr(delta_tracker, 'calculate_portfolio_health'):
            positions_data['portfolio_health'] = await delta_tracker.calculate_portfolio_health()
    
    except Exception as e:
        logger.error(f"Error getting positions data: {e}")
    
    return positions_data


async def _get_recent_alerts(context: ContextTypes.DEFAULT_TYPE, severity_filter: Optional[str] = None) -> Dict[str, Any]:
    """Get recent alerts from position monitor and other sources."""
    alerts_data = {
        'alerts': [],
        'total_count': 0,
        'severity_counts': {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
    }
    
    try:
        # Get alerts from position monitor
        position_monitor = context.bot_data.get('position_monitor')
        if position_monitor and hasattr(position_monitor, 'alert_history'):
            # Get recent alerts (last 24 hours)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            for alert in position_monitor.alert_history:
                if alert.timestamp > cutoff_time:
                    if not severity_filter or alert.severity.value == severity_filter:
                        alerts_data['alerts'].append(alert)
                    
                    alerts_data['severity_counts'][alert.severity.value] += 1
        
        # Sort alerts by timestamp (newest first)
        alerts_data['alerts'].sort(key=lambda x: x.timestamp, reverse=True)
        alerts_data['total_count'] = len(alerts_data['alerts'])
    
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
    
    return alerts_data


async def _gather_performance_metrics(context: ContextTypes.DEFAULT_TYPE, period: str) -> Dict[str, Any]:
    """Gather performance metrics for the specified period."""
    metrics = {
        'period': period,
        'execution_metrics': {},
        'profit_metrics': {},
        'system_metrics': {},
        'error_metrics': {}
    }
    
    try:
        # Get metrics from execution logger
        execution_logger = context.bot_data.get('execution_logger')
        if execution_logger and hasattr(execution_logger, 'get_execution_analytics'):
            hours = {'1h': 1, '24h': 24, '7d': 168, '30d': 720}.get(period, 24)
            analytics = await execution_logger.get_execution_analytics(hours)
            metrics['execution_metrics'] = analytics
        
        # Get system metrics
        authenticator = context.bot_data.get('authenticator')
        if authenticator:
            metrics['system_metrics'] = authenticator.get_session_stats()
    
    except Exception as e:
        logger.error(f"Error gathering performance metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics


async def _get_portfolio_health(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    """Get comprehensive portfolio health information."""
    portfolio_data = {
        'total_value_usd': 0,
        'unrealized_pnl_usd': 0,
        'liquidation_risk_score': 0,
        'position_count': 0,
        'health_distribution': {'healthy': 0, 'warning': 0, 'error': 0, 'critical': 0},
        'top_positions': [],
        'recent_performance': {}
    }
    
    try:
        delta_tracker = context.bot_data.get('delta_tracker')
        if delta_tracker:
            # Get basic portfolio health
            if hasattr(delta_tracker, 'calculate_portfolio_health'):
                health = await delta_tracker.calculate_portfolio_health()
                portfolio_data.update(health)
            
            # Get position details
            positions_data = await _get_positions_data(context)
            portfolio_data['position_count'] = len(positions_data['positions'])
            
            # Calculate health distribution
            for position in positions_data['positions']:
                health_status = position.get('health_status', 'unknown')
                if health_status in portfolio_data['health_distribution']:
                    portfolio_data['health_distribution'][health_status] += 1
            
            # Get top positions by value
            top_positions = sorted(
                positions_data['positions'],
                key=lambda x: x['current_value_usd'],
                reverse=True
            )[:5]
            portfolio_data['top_positions'] = top_positions
    
    except Exception as e:
        logger.error(f"Error getting portfolio health: {e}")
        portfolio_data['error'] = str(e)
    
    return portfolio_data


def _parse_asset_input(asset_input: str) -> str:
    """Parse asset input and return standardized asset ID."""
    # Simple mapping for common assets
    asset_mapping = {
        'ETH': 'ETH_MAINNET_WETH',
        'WETH': 'ETH_MAINNET_WETH',
        'USDC': 'ETH_MAINNET_USDC',
        'USDT': 'ETH_MAINNET_USDT',
        'DAI': 'ETH_MAINNET_DAI'
    }
    
    return asset_mapping.get(asset_input, asset_input)


def _get_asset_symbol(asset_id: str) -> str:
    """Get display symbol for asset ID."""
    if 'WETH' in asset_id:
        return 'ETH'
    elif 'USDC' in asset_id:
        return 'USDC'
    elif 'USDT' in asset_id:
        return 'USDT'
    elif 'DAI' in asset_id:
        return 'DAI'
    else:
        return asset_id.split('_')[-1] if '_' in asset_id else asset_id


async def _format_positions_summary(positions_data: Dict[str, Any]) -> str:
    """Format positions data into readable summary."""
    positions = positions_data['positions']
    total_value = positions_data['total_value_usd']
    total_pnl = positions_data['unrealized_pnl_usd']
    
    pnl_percentage = (total_pnl / total_value * 100) if total_value > 0 else 0
    
    summary = f"""
📊 **Active Positions Summary**

**Portfolio Overview:**
• Total Value: ${total_value:,.2f}
• Unrealized P&L: ${total_pnl:+,.2f} ({pnl_percentage:+.2f}%)
• Position Count: {len(positions)}

**Position Health:**
"""
    
    health_counts = {'healthy': 0, 'warning': 0, 'error': 0, 'critical': 0}
    for position in positions:
        health = position.get('health_status', 'unknown')
        if health in health_counts:
            health_counts[health] += 1
    
    health_icons = {'healthy': '✅', 'warning': '⚠️', 'error': '🚨', 'critical': '🆘'}
    for status, count in health_counts.items():
        if count > 0:
            icon = health_icons.get(status, '❓')
            summary += f"• {icon} {status.title()}: {count}\n"
    
    # Show top positions
    if positions:
        summary += "\n**Top Positions:**\n"
        for i, position in enumerate(positions[:5], 1):
            pnl = position['unrealized_pnl_usd']
            pnl_pct = (pnl / position['initial_value_usd'] * 100) if position['initial_value_usd'] > 0 else 0
            health_icon = health_icons.get(position['health_status'], '❓')
            
            summary += f"{i}. {position['position_id'][:12]}... "
            summary += f"${position['current_value_usd']:,.0f} "
            summary += f"({pnl_pct:+.1f}%) {health_icon}\n"
    
    return summary


async def _update_config_parameter(context: ContextTypes.DEFAULT_TYPE, param: str, value: str) -> bool:
    """Update a configuration parameter."""
    try:
        config_manager = context.bot_data.get('config_manager')
        if not config_manager:
            return False
        
        # Convert value to appropriate type
        converted_value = _convert_config_value(param, value)
        if converted_value is None:
            return False
        
        # Update configuration
        config_manager.update_config(**{param: converted_value})
        return True
    
    except Exception as e:
        logger.error(f"Error updating config parameter {param}: {e}")
        return False


def _convert_config_value(param: str, value: str) -> Any:
    """Convert string value to appropriate type for config parameter."""
    try:
        # Boolean parameters
        if param in ['enable_position_monitoring', 'enable_execution_logging', 'enable_risk_alerts']:
            return value.lower() in ['true', '1', 'yes', 'on']
        
        # Integer parameters
        elif param in ['max_opportunities_displayed', 'max_alerts_displayed', 'command_timeout_seconds']:
            return int(value)
        
        # Float parameters
        elif param in ['precision_decimals']:
            return float(value)
        
        # String parameters
        else:
            return value
    
    except ValueError:
        return None


async def _reset_configuration(context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Reset configuration to defaults."""
    try:
        config_manager = context.bot_data.get('config_manager')
        if not config_manager:
            return False
        
        # Create new default config
        from .config import BotConfig
        default_config = BotConfig.from_env()
        config_manager.save_config(default_config)
        
        # Update bot_data
        context.bot_data['bot_config'] = default_config
        return True
    
    except Exception as e:
        logger.error(f"Error resetting configuration: {e}")
        return False