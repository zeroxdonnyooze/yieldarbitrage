"""
Message Formatters for Telegram Bot.

This module contains functions to format various data structures into 
clean, readable Telegram messages with proper formatting and emojis.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


def format_system_status(status_data: Dict[str, Any]) -> str:
    """Format comprehensive system status information."""
    timestamp = status_data.get('timestamp', datetime.now(timezone.utc))
    
    status_msg = f"""
🔄 **System Status** - {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

"""
    
    # Graph statistics
    graph_data = status_data.get('graph', {})
    if graph_data:
        total_edges = graph_data.get('total_edges', 0)
        total_nodes = graph_data.get('total_nodes', 0)
        last_update = graph_data.get('last_update')
        
        status_msg += f"""**📊 Graph Engine:**
• Edges: {total_edges:,}
• Nodes: {total_nodes:,}
• Last Update: {_format_time_ago(last_update) if last_update else 'Unknown'}

"""
    
    # Data collector status
    collector_data = status_data.get('data_collector', {})
    if collector_data:
        is_running = collector_data.get('is_running', False)
        last_collection = collector_data.get('last_collection')
        collections_today = collector_data.get('collections_today', 0)
        
        status_icon = "🟢" if is_running else "🔴"
        status_msg += f"""**{status_icon} Data Collector:**
• Status: {'Running' if is_running else 'Stopped'}
• Last Collection: {_format_time_ago(last_collection) if last_collection else 'Never'}
• Collections Today: {collections_today}

"""
    
    # Position monitor status
    monitor_data = status_data.get('position_monitor', {})
    if monitor_data:
        is_monitoring = monitor_data.get('is_monitoring', False)
        positions_monitored = monitor_data.get('positions_monitored', 0)
        recent_alerts = monitor_data.get('recent_alerts', 0)
        
        monitor_icon = "🟢" if is_monitoring else "🔴"
        status_msg += f"""**{monitor_icon} Position Monitor:**
• Status: {'Active' if is_monitoring else 'Inactive'}
• Positions: {positions_monitored}
• Recent Alerts: {recent_alerts}

"""
    
    # Execution logger status
    logger_data = status_data.get('execution_logger', {})
    if logger_data:
        records_created = logger_data.get('records_created', 0)
        records_updated = logger_data.get('records_updated', 0)
        write_errors = logger_data.get('write_errors', 0)
        
        logger_health = "🟢" if write_errors == 0 else "🟡" if write_errors < 5 else "🔴"
        status_msg += f"""**{logger_health} Execution Logger:**
• Records Created: {records_created:,}
• Records Updated: {records_updated:,}
• Write Errors: {write_errors}

"""
    
    # Error information
    if 'error' in status_data:
        status_msg += f"❌ **System Error:** {status_data['error']}\n"
    
    return status_msg.strip()


def format_opportunities(opportunities: List[Dict[str, Any]], amount: float, start_asset: str) -> str:
    """Format arbitrage opportunities for display."""
    if not opportunities:
        return "🔍 No profitable opportunities found."
    
    asset_symbol = _get_asset_symbol(start_asset)
    
    msg = f"""
💰 **Arbitrage Opportunities** ({amount} {asset_symbol})

"""
    
    for i, opp in enumerate(opportunities, 1):
        profit_usd = opp.get('profit_usd', 0)
        profit_pct = opp.get('profit_percentage', 0)
        gas_cost = opp.get('gas_cost_usd', 0)
        net_profit = profit_usd - gas_cost
        apr = opp.get('estimated_apr', 0)
        risk_score = opp.get('risk_score', 0.5)
        
        # Risk emoji
        risk_emoji = "🟢" if risk_score < 0.3 else "🟡" if risk_score < 0.7 else "🔴"
        
        # Path summary
        path = opp.get('path', [])
        path_summary = _format_path_summary(path)
        
        msg += f"""**{i}. {path_summary}**
💵 Profit: ${profit_usd:.2f} ({profit_pct:.2f}%)
⛽ Gas: ${gas_cost:.2f}
💰 Net: ${net_profit:.2f}
📈 Est. APR: {apr:.1f}%
{risk_emoji} Risk: {risk_score:.2f}

"""
    
    msg += f"⏰ *Data as of {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}*"
    return msg


def format_position_alerts(alerts: List[Any], severity_filter: Optional[str] = None) -> str:
    """Format position alerts for display."""
    if not alerts:
        filter_text = f" ({severity_filter})" if severity_filter else ""
        return f"✅ No recent alerts{filter_text}."
    
    severity_icons = {
        'info': 'ℹ️',
        'warning': '⚠️', 
        'error': '🚨',
        'critical': '🆘'
    }
    
    filter_text = f" ({severity_filter.upper()})" if severity_filter else ""
    msg = f"🚨 **Recent Alerts{filter_text}**\n\n"
    
    for alert in alerts:
        severity = alert.severity.value
        icon = severity_icons.get(severity, '❓')
        timestamp = alert.timestamp.strftime('%H:%M')
        
        msg += f"{icon} **{alert.position_id[:12]}...** ({timestamp})\n"
        msg += f"└ {alert.message}\n"
        
        if alert.recommended_action:
            msg += f"└ 💡 *{alert.recommended_action}*\n"
        
        msg += "\n"
    
    return msg.strip()


def format_execution_metrics(metrics_data: Dict[str, Any], period: str) -> str:
    """Format execution performance metrics."""
    period_name = {
        '1h': 'Last Hour',
        '24h': 'Last 24 Hours', 
        '7d': 'Last 7 Days',
        '30d': 'Last 30 Days'
    }.get(period, period)
    
    msg = f"📊 **Performance Metrics** ({period_name})\n\n"
    
    # Execution metrics
    exec_metrics = metrics_data.get('execution_metrics', {})
    if exec_metrics:
        total_executions = exec_metrics.get('total_executions', 0)
        successful = exec_metrics.get('successful_executions', 0)
        success_rate = (successful / total_executions * 100) if total_executions > 0 else 0
        avg_profit = exec_metrics.get('avg_predicted_profit_usd', 0)
        
        msg += f"""**🎯 Execution Stats:**
• Total Attempts: {total_executions}
• Successful: {successful}
• Success Rate: {success_rate:.1f}%
• Avg Profit: ${avg_profit:.2f}

"""
    
    # System metrics
    system_metrics = metrics_data.get('system_metrics', {})
    if system_metrics:
        total_commands = system_metrics.get('total_commands_processed', 0)
        active_sessions = system_metrics.get('active_sessions', 0)
        
        msg += f"""**⚙️ System Stats:**
• Commands Processed: {total_commands:,}
• Active Sessions: {active_sessions}

"""
    
    # Error information
    if 'error' in metrics_data:
        msg += f"❌ **Error:** {metrics_data['error']}\n"
    
    return msg.strip()


def format_portfolio_health(portfolio_data: Dict[str, Any]) -> str:
    """Format portfolio health summary."""
    total_value = portfolio_data.get('total_value_usd', 0)
    pnl = portfolio_data.get('unrealized_pnl_usd', 0)
    position_count = portfolio_data.get('position_count', 0)
    liquidation_risk = portfolio_data.get('liquidation_risk_score', 0)
    
    pnl_percentage = (pnl / total_value * 100) if total_value > 0 else 0
    
    # Health status
    if liquidation_risk > 0.7:
        health_status = "🔴 High Risk"
    elif liquidation_risk > 0.4:
        health_status = "🟡 Medium Risk"
    else:
        health_status = "🟢 Low Risk"
    
    msg = f"""
💼 **Portfolio Health**

**Overview:**
• Total Value: ${total_value:,.2f}
• Unrealized P&L: ${pnl:+,.2f} ({pnl_percentage:+.2f}%)
• Positions: {position_count}
• Risk Status: {health_status}

"""
    
    # Health distribution
    health_dist = portfolio_data.get('health_distribution', {})
    if any(health_dist.values()):
        msg += "**Position Health:**\n"
        health_icons = {'healthy': '✅', 'warning': '⚠️', 'error': '🚨', 'critical': '🆘'}
        
        for status, count in health_dist.items():
            if count > 0:
                icon = health_icons.get(status, '❓')
                msg += f"• {icon} {status.title()}: {count}\n"
        msg += "\n"
    
    # Top positions
    top_positions = portfolio_data.get('top_positions', [])
    if top_positions:
        msg += "**Top Positions:**\n"
        for i, pos in enumerate(top_positions[:3], 1):
            value = pos.get('current_value_usd', 0)
            pnl_pos = pos.get('unrealized_pnl_usd', 0)
            pnl_pct_pos = (pnl_pos / pos.get('initial_value_usd', 1) * 100) if pos.get('initial_value_usd', 0) > 0 else 0
            
            msg += f"{i}. {pos.get('position_id', 'Unknown')[:12]}... "
            msg += f"${value:,.0f} ({pnl_pct_pos:+.1f}%)\n"
    
    return msg.strip()


def format_config_display(config: Any, is_admin: bool) -> str:
    """Format configuration settings for display."""
    msg = "⚙️ **Configuration Settings**\n\n"
    
    # Basic settings (visible to all users)
    msg += f"""**Display Settings:**
• Max Opportunities: {config.max_opportunities_displayed}
• Max Alerts: {config.max_alerts_displayed}
• Currency: {config.default_currency}
• Precision: {config.precision_decimals} decimals

**Features:**
• Position Monitoring: {'✅' if config.enable_position_monitoring else '❌'}
• Execution Logging: {'✅' if config.enable_execution_logging else '❌'}
• Risk Alerts: {'✅' if config.enable_risk_alerts else '❌'}

"""
    
    if is_admin:
        msg += f"""**Admin Settings:**
• Command Timeout: {config.command_timeout_seconds}s
• Rate Limit: {config.commands_per_minute}/min
• Alert Threshold: {config.alert_severity_threshold}
• Auto Responses: {'✅' if config.enable_auto_responses else '❌'}

**User Access:**
• Allowed Users: {len(config.allowed_user_ids)}
• Admin Users: {len(config.admin_user_ids)}

"""
    
    msg += "\n💡 *Use `/config set <param> <value>` to modify settings (admin only)*"
    return msg


def _format_time_ago(timestamp: Optional[datetime]) -> str:
    """Format timestamp as 'X time ago' string."""
    if not timestamp:
        return "Never"
    
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return "Unknown"
    
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    delta = now - timestamp
    
    if delta.total_seconds() < 60:
        return "Just now"
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes}m ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        return f"{hours}h ago"
    else:
        days = delta.days
        return f"{days}d ago"


def _format_path_summary(path: List[Any]) -> str:
    """Format arbitrage path into readable summary."""
    if not path:
        return "Unknown Path"
    
    if len(path) <= 3:
        # Simple path - show all protocols
        protocols = []
        for edge in path:
            protocol = getattr(edge, 'protocol_name', 'Unknown')
            if protocol not in protocols:
                protocols.append(protocol)
        return " → ".join(protocols)
    else:
        # Complex path - show count
        protocols = set()
        for edge in path:
            protocol = getattr(edge, 'protocol_name', 'Unknown')
            protocols.add(protocol)
        
        return f"{len(path)}-step path ({len(protocols)} protocols)"


def _get_asset_symbol(asset_id: str) -> str:
    """Get display symbol for asset ID."""
    symbol_mapping = {
        'ETH_MAINNET_WETH': 'ETH',
        'ETH_MAINNET_USDC': 'USDC',
        'ETH_MAINNET_USDT': 'USDT',
        'ETH_MAINNET_DAI': 'DAI'
    }
    
    return symbol_mapping.get(asset_id, asset_id.split('_')[-1] if '_' in asset_id else asset_id)


def format_number(value: float, precision: int = 2) -> str:
    """Format number with appropriate precision and comma separation."""
    if abs(value) >= 1000000:
        return f"{value / 1000000:.{precision}f}M"
    elif abs(value) >= 1000:
        return f"{value / 1000:.{precision}f}K"
    else:
        return f"{value:,.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """Format percentage with appropriate sign."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{precision}f}%"


def format_currency(value: float, currency: str = "USD", precision: int = 2) -> str:
    """Format currency amount."""
    if currency == "USD":
        return f"${value:,.{precision}f}"
    else:
        return f"{value:,.{precision}f} {currency}"


def truncate_text(text: str, max_length: int = 4000) -> str:
    """Truncate text to fit Telegram message limits."""
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a line break
    truncated = text[:max_length-100]
    last_newline = truncated.rfind('\n')
    
    if last_newline > max_length * 0.8:  # If we can keep 80% of content
        truncated = truncated[:last_newline]
    
    return truncated + "\n\n... (message truncated)"