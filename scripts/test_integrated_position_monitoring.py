#!/usr/bin/env python3
"""
Integrated Position Monitoring Demonstration.

This script demonstrates how the PositionMonitor integrates with
the ExecutionEngine and LoggedExecutionEngine for comprehensive
risk management and position tracking.
"""
import sys
import os
import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def create_mock_components():
    """Create all mock components needed for integration testing."""
    
    # Mock DeltaTracker
    delta_tracker = Mock()
    delta_tracker.get_all_positions = Mock(return_value={})
    delta_tracker.get_position = Mock(return_value=None)
    delta_tracker.calculate_portfolio_health = AsyncMock(return_value={
        "total_value_usd": 250000.0,
        "unrealized_pnl_usd": -8500.0,  # 3.4% loss
        "liquidation_risk_score": 0.12,
        "position_count": 6
    })
    
    # Mock AssetOracle
    asset_oracle = Mock()
    asset_oracle.get_price_usd = AsyncMock(side_effect=lambda asset_id: {
        "ETH_MAINNET_WETH": 2150.0,  # ETH down from $2000
        "ETH_MAINNET_USDC": 1.0,
        "ETH_MAINNET_USDT": 1.0
    }.get(asset_id, 1.0))
    asset_oracle.is_stable_asset = Mock(side_effect=lambda asset_id: 
        asset_id in ["ETH_MAINNET_USDC", "ETH_MAINNET_USDT"])
    
    # Mock ExecutionLogger
    execution_logger = Mock()
    execution_logger.log_execution_start = AsyncMock(return_value=True)
    execution_logger.log_pre_flight_results = AsyncMock(return_value=True)
    execution_logger.log_simulation_results = AsyncMock(return_value=True)
    execution_logger.log_execution_completion = AsyncMock(return_value=True)
    execution_logger.log_position_alert = AsyncMock(return_value=True)
    execution_logger.get_stats = Mock(return_value={
        "records_created": 45,
        "records_updated": 128,
        "write_errors": 0,
        "last_write_time": time.time()
    })
    
    # Mock other ExecutionEngine components
    simulator = Mock()
    transaction_builder = Mock()
    mev_router = Mock()
    mev_assessor = Mock()
    
    return {
        "delta_tracker": delta_tracker,
        "asset_oracle": asset_oracle,
        "execution_logger": execution_logger,
        "simulator": simulator,
        "transaction_builder": transaction_builder,
        "mev_router": mev_router,
        "mev_assessor": mev_assessor
    }


def create_comprehensive_positions():
    """Create a comprehensive set of positions for testing."""
    now = datetime.now(timezone.utc)
    positions = {}
    
    # Import PositionType from standalone script
    import sys
    sys.path.append('scripts')
    from test_position_monitor_standalone import PositionType
    
    # 1. Active arbitrage position (healthy)
    arb_healthy = Mock()
    arb_healthy.position_id = "ARB_HEALTHY_001"
    arb_healthy.position_type = PositionType.ARBITRAGE
    arb_healthy.created_at = now - timedelta(seconds=45)  # 45 seconds old
    arb_healthy.expected_completion_time = 60  # 1 minute expected
    arb_healthy.status = "executing"
    arb_healthy.current_value_usd = 10075.0
    arb_healthy.initial_value_usd = 10000.0
    arb_healthy.metadata = {"profit_usd": 75.0, "gas_cost_usd": 12.50}
    positions[arb_healthy.position_id] = arb_healthy
    
    # 2. Slow arbitrage position (warning)
    arb_slow = Mock()
    arb_slow.position_id = "ARB_SLOW_002"
    arb_slow.position_type = PositionType.ARBITRAGE
    arb_slow.created_at = now - timedelta(minutes=8)  # 8 minutes old
    arb_slow.expected_completion_time = 30  # 30 seconds expected
    arb_slow.status = "executing"
    arb_slow.current_value_usd = 5000.0
    arb_slow.initial_value_usd = 5000.0
    arb_slow.metadata = {"tx_hash": "0xabc123...", "stuck_at_step": 2}
    positions[arb_slow.position_id] = arb_slow
    
    # 3. Yield farming with IL (warning)
    yield_il = Mock()
    yield_il.position_id = "YIELD_IL_003"
    yield_il.position_type = PositionType.YIELD_FARMING
    yield_il.created_at = now - timedelta(hours=18)
    yield_il.status = "active"
    yield_il.current_value_usd = 48200.0
    yield_il.initial_value_usd = 50000.0
    yield_il.metadata = {
        "pool_type": "uniswap_v3",
        "initial_eth_price": 2000.0,
        "initial_usdc_price": 1.0,
        "lp_token_amount": 25000.0,
        "fee_tier": 0.003,
        "range_lower": 1800,
        "range_upper": 2200
    }
    yield_il.exposures = {
        "ETH_MAINNET_WETH": Mock(amount=Decimal("12.1")),
        "ETH_MAINNET_USDC": Mock(amount=Decimal("24000.0"))
    }
    positions[yield_il.position_id] = yield_il
    
    # 4. Healthy lending position
    lend_healthy = Mock()
    lend_healthy.position_id = "LEND_HEALTHY_004"
    lend_healthy.position_type = PositionType.LENDING
    lend_healthy.created_at = now - timedelta(days=2)
    lend_healthy.status = "active"
    lend_healthy.current_value_usd = 75680.0
    lend_healthy.initial_value_usd = 75000.0
    lend_healthy.metadata = {
        "protocol": "aave",
        "asset": "ETH_MAINNET_USDC",
        "utilization_rate": 0.72,  # 72% - healthy
        "current_apr": 0.065,
        "initial_apr": 0.050
    }
    positions[lend_healthy.position_id] = lend_healthy
    
    # 5. High utilization lending position (info alert)
    lend_high_util = Mock()
    lend_high_util.position_id = "LEND_HIGH_UTIL_005"
    lend_high_util.position_type = PositionType.LENDING
    lend_high_util.created_at = now - timedelta(hours=8)
    lend_high_util.status = "active"
    lend_high_util.current_value_usd = 32150.0
    lend_high_util.initial_value_usd = 32000.0
    lend_high_util.metadata = {
        "protocol": "compound",
        "asset": "ETH_MAINNET_USDT",
        "utilization_rate": 0.91,  # 91% - high
        "current_apr": 0.095,
        "initial_apr": 0.055
    }
    positions[lend_high_util.position_id] = lend_high_util
    
    # 6. Critical borrowing position (critical alert)
    borrow_critical = Mock()
    borrow_critical.position_id = "BORROW_CRITICAL_006"
    borrow_critical.position_type = PositionType.BORROWING
    borrow_critical.created_at = now - timedelta(hours=4)
    borrow_critical.status = "active"
    borrow_critical.current_value_usd = -45000.0  # Debt position
    borrow_critical.initial_value_usd = -40000.0
    borrow_critical.metadata = {
        "protocol": "aave",
        "collateral_asset": "ETH_MAINNET_WETH",
        "debt_asset": "ETH_MAINNET_USDC",
        "health_factor": 1.08,  # Very close to liquidation
        "liquidation_threshold": 1.0,
        "collateral_value_usd": 55000.0,
        "debt_value_usd": 45000.0,
        "liquidation_price_eth": 1950.0  # ETH price at liquidation
    }
    positions[borrow_critical.position_id] = borrow_critical
    
    return positions


async def demonstrate_integrated_monitoring():
    """Demonstrate integrated position monitoring with ExecutionEngine."""
    print("🏗️  Integrated Position Monitoring Demonstration")
    print("=" * 80)
    
    # Step 1: Create mock components
    print("\n🔧 Step 1: Creating Mock Components")
    components = create_mock_components()
    print(f"   ✅ Created {len(components)} mock components:")
    for name in components.keys():
        print(f"     • {name}")
    
    # Step 2: Create comprehensive positions
    print("\n💼 Step 2: Creating Comprehensive Position Portfolio")
    positions = create_comprehensive_positions()
    components["delta_tracker"].get_all_positions.return_value = positions
    
    print(f"   ✅ Created {len(positions)} positions:")
    for pos_id, position in positions.items():
        age = datetime.now(timezone.utc) - position.created_at
        value_change = position.current_value_usd - position.initial_value_usd
        change_percent = (value_change / abs(position.initial_value_usd)) * 100
        
        print(f"     • {pos_id}:")
        print(f"       Type: {position.position_type.value}")
        print(f"       Age: {age.total_seconds()/3600:.1f}h")
        print(f"       Value: ${position.current_value_usd:,.0f} ({change_percent:+.1f}%)")
    
    # Step 3: Test standalone position monitoring
    print("\n🎯 Step 3: Testing Standalone Position Monitoring")
    
    # Import standalone monitoring classes
    from test_position_monitor_standalone import (
        SimplePositionMonitor, MonitoringConfig, AlertSeverity
    )
    
    config = MonitoringConfig(
        check_interval_seconds=20,
        max_unrealized_loss_percent=8.0,
        impermanent_loss_threshold=3.5,
        liquidation_threshold_buffer=0.12
    )
    
    monitor = SimplePositionMonitor(config)
    alerts = await monitor.monitor_positions(positions)
    
    print(f"   🚨 Generated {len(alerts)} alerts:")
    
    # Group and display alerts by severity
    severity_groups = {}
    for alert in alerts:
        severity = alert.severity
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(alert)
    
    severity_order = [AlertSeverity.CRITICAL, AlertSeverity.ERROR, AlertSeverity.WARNING, AlertSeverity.INFO]
    
    for severity in severity_order:
        if severity in severity_groups:
            severity_icon = {
                AlertSeverity.CRITICAL: "🆘",
                AlertSeverity.ERROR: "🚨",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.INFO: "ℹ️"
            }[severity]
            
            print(f"\n   {severity_icon} {severity.value.upper()} ({len(severity_groups[severity])}):")
            
            for alert in severity_groups[severity]:
                print(f"     • {alert.position_id}: {alert.message}")
                if alert.recommended_action:
                    print(f"       → {alert.recommended_action}")
    
    # Step 4: Portfolio health analysis
    print("\n📊 Step 4: Portfolio Health Analysis")
    portfolio_health = await components["delta_tracker"].calculate_portfolio_health()
    
    total_value = portfolio_health["total_value_usd"]
    pnl = portfolio_health["unrealized_pnl_usd"]
    pnl_percent = (pnl / total_value) * 100
    liquidation_risk = portfolio_health["liquidation_risk_score"]
    
    print(f"   💰 Portfolio Overview:")
    print(f"     - Total Value: ${total_value:,.0f}")
    print(f"     - Unrealized P&L: ${pnl:,.0f} ({pnl_percent:+.1f}%)")
    print(f"     - Liquidation Risk: {liquidation_risk:.1%}")
    print(f"     - Position Count: {portfolio_health['position_count']}")
    
    # Portfolio health assessment
    if abs(pnl_percent) > config.max_unrealized_loss_percent:
        print(f"   ⚠️  Portfolio loss exceeds threshold ({config.max_unrealized_loss_percent}%)")
    elif liquidation_risk > 0.15:
        print(f"   ⚠️  High liquidation risk detected")
    else:
        print(f"   ✅ Portfolio health within acceptable parameters")
    
    # Step 5: Risk analysis by position type
    print("\n📈 Step 5: Risk Analysis by Position Type")
    
    type_analysis = {}
    for position in positions.values():
        pos_type = position.position_type.value
        if pos_type not in type_analysis:
            type_analysis[pos_type] = {
                "count": 0,
                "total_value": 0,
                "alerts": 0,
                "critical_alerts": 0
            }
        
        type_analysis[pos_type]["count"] += 1
        type_analysis[pos_type]["total_value"] += abs(position.current_value_usd)
        
        # Count alerts for this position
        position_alerts = [a for a in alerts if a.position_id == position.position_id]
        type_analysis[pos_type]["alerts"] += len(position_alerts)
        type_analysis[pos_type]["critical_alerts"] += len([
            a for a in position_alerts if a.severity == AlertSeverity.CRITICAL
        ])
    
    for pos_type, analysis in type_analysis.items():
        print(f"   🏷️  {pos_type.replace('_', ' ').title()}:")
        print(f"     - Positions: {analysis['count']}")
        print(f"     - Total Value: ${analysis['total_value']:,.0f}")
        print(f"     - Alerts: {analysis['alerts']} ({analysis['critical_alerts']} critical)")
        
        # Risk assessment
        alert_ratio = analysis['alerts'] / analysis['count'] if analysis['count'] > 0 else 0
        if analysis['critical_alerts'] > 0:
            risk_level = "🔴 HIGH"
        elif alert_ratio > 0.5:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"
        print(f"     - Risk Level: {risk_level}")
    
    # Step 6: Actionable recommendations
    print("\n🎯 Step 6: Actionable Recommendations")
    
    actionable_alerts = [a for a in alerts if a.is_actionable and a.recommended_action]
    
    if actionable_alerts:
        # Sort by severity (critical first)
        severity_priority = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        actionable_alerts.sort(key=lambda x: severity_priority[x.severity])
        
        print(f"   📋 {len(actionable_alerts)} actionable recommendations:")
        
        for i, alert in enumerate(actionable_alerts, 1):
            priority = "🔥 URGENT" if alert.severity == AlertSeverity.CRITICAL else "📌 NORMAL"
            print(f"\n   {i}. {priority} - {alert.position_id}")
            print(f"      Issue: {alert.message}")
            print(f"      Action: {alert.recommended_action}")
            
            # Add estimated time to resolution
            if alert.alert_type == "execution_timeout":
                print(f"      ETA: Immediate intervention required")
            elif alert.alert_type == "health_factor_critical":
                print(f"      ETA: Within 1 hour to avoid liquidation")
            elif alert.alert_type == "impermanent_loss":
                print(f"      ETA: Monitor and act within 24 hours")
            else:
                print(f"      ETA: Review within next monitoring cycle")
    else:
        print("   ✅ No actionable recommendations - portfolio is healthy!")
    
    # Step 7: Integration summary
    print("\n🏁 Step 7: Integration Summary")
    
    execution_stats = components["execution_logger"].get_stats()
    
    integration_summary = {
        "monitoring_enabled": True,
        "positions_monitored": len(positions),
        "alerts_generated": len(alerts),
        "critical_issues": len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
        "portfolio_value": total_value,
        "portfolio_pnl_percent": pnl_percent,
        "execution_records": execution_stats["records_created"],
        "execution_updates": execution_stats["records_updated"],
        "monitoring_healthy": len(actionable_alerts) == 0
    }
    
    print(f"   📊 Integration Metrics:")
    for key, value in integration_summary.items():
        display_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            if 'percent' in key:
                print(f"     - {display_key}: {value:+.1f}%")
            else:
                print(f"     - {display_key}: ${value:,.0f}")
        else:
            print(f"     - {display_key}: {value}")
    
    # Overall health score
    health_score = 100
    if integration_summary["critical_issues"] > 0:
        health_score -= integration_summary["critical_issues"] * 30
    if abs(integration_summary["portfolio_pnl_percent"]) > 5:
        health_score -= 20
    if integration_summary["alerts_generated"] > 5:
        health_score -= 10
    
    health_score = max(0, health_score)
    
    if health_score >= 80:
        health_icon = "🟢"
        health_status = "HEALTHY"
    elif health_score >= 60:
        health_icon = "🟡"
        health_status = "ATTENTION NEEDED"
    else:
        health_icon = "🔴"
        health_status = "CRITICAL"
    
    print(f"\n   {health_icon} Overall Portfolio Health: {health_score}/100 - {health_status}")
    
    print(f"\n✅ Integrated Position Monitoring Demonstration Complete!")
    print(f"\nKey Integration Benefits:")
    print(f"• 🎯 Real-time risk assessment across all position types")
    print(f"• 📊 Portfolio-wide health monitoring with liquidation risk")
    print(f"• 🚨 Intelligent alerting with severity-based prioritization")
    print(f"• 📋 Actionable recommendations with time-sensitive priorities")
    print(f"• 🔄 Seamless integration with ExecutionEngine and logging")
    print(f"• 📈 Historical trend analysis through persistent alerts")
    print(f"• ⚡ High-frequency monitoring for critical positions")


if __name__ == "__main__":
    asyncio.run(demonstrate_integrated_monitoring())