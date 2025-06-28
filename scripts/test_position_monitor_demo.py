#!/usr/bin/env python3
"""
Demonstration script for Position Monitor functionality.

This script demonstrates the PositionMonitor system including specialized
monitors, alert generation, and position health calculations.
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

from yield_arbitrage.monitoring.position_monitor import (
    PositionMonitor, PositionType, RiskLevel, AlertSeverity, PositionAlert, 
    MonitoringConfig, ArbitragePositionMonitor, YieldFarmingPositionMonitor,
    LendingPositionMonitor, BorrowingPositionMonitor
)
from yield_arbitrage.risk.delta_tracker import DeltaPosition, AssetExposure


def create_mock_components():
    """Create mock components for testing."""
    # Mock delta tracker
    delta_tracker = Mock()
    delta_tracker.get_all_positions = Mock(return_value={})
    delta_tracker.get_position = Mock(return_value=None)
    delta_tracker.calculate_portfolio_health = AsyncMock(return_value={
        "total_value_usd": 125000.0,
        "unrealized_pnl_usd": 5250.0,
        "liquidation_risk_score": 0.08,
        "position_count": 4
    })
    
    # Mock asset oracle
    asset_oracle = Mock()
    asset_oracle.get_price_usd = AsyncMock(return_value=2000.0)
    asset_oracle.is_stable_asset = Mock(return_value=False)
    
    # Mock execution logger
    execution_logger = Mock()
    execution_logger.log_position_alert = AsyncMock(return_value=True)
    
    return delta_tracker, asset_oracle, execution_logger


def create_sample_positions():
    """Create sample positions for testing."""
    now = datetime.now(timezone.utc)
    
    positions = {}
    
    # 1. Arbitrage position (fast execution expected)
    arb_position = Mock()
    arb_position.position_id = "ARB_001"
    arb_position.position_type = PositionType.ARBITRAGE
    arb_position.created_at = now - timedelta(minutes=2)  # 2 minutes old
    arb_position.status = "executing"
    arb_position.expected_completion_time = 30  # Expected 30 seconds
    arb_position.current_value_usd = 10125.0
    arb_position.initial_value_usd = 10000.0
    arb_position.metadata = {
        "path_id": "uniswap_sushiswap_eth_usdc",
        "expected_profit_usd": 150.0
    }
    positions["ARB_001"] = arb_position
    
    # 2. Yield farming position with impermanent loss
    yield_position = Mock()
    yield_position.position_id = "YIELD_001"
    yield_position.position_type = PositionType.YIELD_FARMING
    yield_position.created_at = now - timedelta(hours=24)
    yield_position.status = "active"
    yield_position.current_value_usd = 47250.0  # Down from 50k
    yield_position.initial_value_usd = 50000.0
    yield_position.metadata = {
        "pool_type": "uniswap_v3",
        "initial_eth_price": 2000.0,
        "initial_usdc_price": 1.0,
        "lp_token_amount": 25000.0
    }
    yield_position.exposures = {
        "ETH_MAINNET_WETH": Mock(amount=Decimal("12.5")),
        "ETH_MAINNET_USDC": Mock(amount=Decimal("25000.0"))
    }
    positions["YIELD_001"] = yield_position
    
    # 3. Lending position with high utilization
    lend_position = Mock()
    lend_position.position_id = "LEND_001"
    lend_position.position_type = PositionType.LENDING
    lend_position.created_at = now - timedelta(hours=12)
    lend_position.status = "active"
    lend_position.current_value_usd = 75420.0
    lend_position.initial_value_usd = 75000.0
    lend_position.metadata = {
        "protocol": "aave",
        "asset": "ETH_MAINNET_USDC",
        "utilization_rate": 0.92,  # 92% utilization
        "current_apr": 0.087,      # 8.7% APR
        "initial_apr": 0.065       # Was 6.5%
    }
    positions["LEND_001"] = lend_position
    
    # 4. Borrowing position with declining health factor
    borrow_position = Mock()
    borrow_position.position_id = "BORROW_001"
    borrow_position.position_type = PositionType.BORROWING
    borrow_position.created_at = now - timedelta(hours=6)
    borrow_position.status = "active"
    borrow_position.current_value_usd = -40000.0  # Debt position
    borrow_position.initial_value_usd = -35000.0  # Debt increased
    borrow_position.metadata = {
        "protocol": "aave",
        "collateral_asset": "ETH_MAINNET_WETH",
        "debt_asset": "ETH_MAINNET_USDC",
        "health_factor": 1.35,     # Getting lower
        "liquidation_threshold": 1.0,
        "collateral_value_usd": 60000.0,
        "debt_value_usd": 40000.0
    }
    positions["BORROW_001"] = borrow_position
    
    return positions


async def demonstrate_position_monitor():
    """Demonstrate the PositionMonitor system."""
    print("🏛️  Position Monitor Demonstration")
    print("=" * 70)
    
    # Step 1: Create components
    print("\n📋 Step 1: Creating Components")
    delta_tracker, asset_oracle, execution_logger = create_mock_components()
    
    config = MonitoringConfig(
        check_interval_seconds=10,
        high_frequency_interval_seconds=2,
        max_unrealized_loss_percent=8.0,
        impermanent_loss_threshold=4.0,
        liquidation_threshold_buffer=0.15
    )
    
    print(f"   - Config: {config.check_interval_seconds}s intervals")
    print(f"   - Max loss threshold: {config.max_unrealized_loss_percent}%")
    print(f"   - IL threshold: {config.impermanent_loss_threshold}%")
    print(f"   - Liquidation buffer: {config.liquidation_threshold_buffer * 100}%")
    
    # Step 2: Create PositionMonitor
    print("\n🎯 Step 2: Creating PositionMonitor")
    monitor = PositionMonitor(
        delta_tracker=delta_tracker,
        asset_oracle=asset_oracle,
        execution_logger=execution_logger,
        config=config
    )
    
    print(f"   - Specialized monitors: {len(monitor.position_monitors)}")
    for spec_monitor in monitor.position_monitors:
        print(f"     • {type(spec_monitor).__name__}: {spec_monitor.position_type.value}")
    
    print(f"   - Is monitoring: {monitor.is_monitoring}")
    print(f"   - Alert history: {len(monitor.alert_history)} alerts")
    
    # Step 3: Create sample positions
    print("\n💼 Step 3: Creating Sample Positions")
    positions = create_sample_positions()
    delta_tracker.get_all_positions.return_value = positions
    
    for pos_id, position in positions.items():
        age_minutes = (datetime.now(timezone.utc) - position.created_at).total_seconds() / 60
        print(f"   - {pos_id}: {position.position_type.value}")
        print(f"     Status: {position.status}, Age: {age_minutes:.1f}min")
        print(f"     Value: ${position.current_value_usd:,.0f} "
              f"(Initial: ${position.initial_value_usd:,.0f})")
    
    # Step 4: Mock asset prices for IL calculation
    print("\n💰 Step 4: Setting Asset Prices")
    asset_oracle.get_price_usd.side_effect = lambda asset_id: {
        "ETH_MAINNET_WETH": 2300.0,  # ETH up 15% from $2000
        "ETH_MAINNET_USDC": 1.0      # USDC stable
    }.get(asset_id, 1.0)
    
    print(f"   - ETH price: $2,300 (up 15% from $2,000)")
    print(f"   - USDC price: $1.00 (stable)")
    
    # Step 5: Check portfolio health
    print("\n📊 Step 5: Checking Portfolio Health")
    portfolio_health = await monitor._check_portfolio_health()
    
    print(f"   - Total value: ${portfolio_health['total_value_usd']:,.0f}")
    print(f"   - Unrealized P&L: ${portfolio_health['unrealized_pnl_usd']:,.0f}")
    pnl_percent = (portfolio_health['unrealized_pnl_usd'] / portfolio_health['total_value_usd']) * 100
    print(f"   - P&L percentage: {pnl_percent:.1f}%")
    print(f"   - Liquidation risk: {portfolio_health['liquidation_risk_score']:.1%}")
    print(f"   - Position count: {portfolio_health['position_count']}")
    
    # Step 6: Monitor individual positions
    print("\n🔍 Step 6: Monitoring Individual Positions")
    all_alerts = []
    
    for spec_monitor in monitor.position_monitors:
        monitor_type = type(spec_monitor).__name__
        print(f"\n   📋 {monitor_type}:")
        
        # Find positions matching this monitor type
        matching_positions = [
            pos for pos in positions.values()
            if pos.position_type == spec_monitor.position_type
        ]
        
        if not matching_positions:
            print(f"     - No {spec_monitor.position_type.value} positions")
            continue
        
        for position in matching_positions:
            print(f"     - Checking {position.position_id}...")
            
            try:
                alerts = await spec_monitor.check_position_health(position)
                all_alerts.extend(alerts)
                
                if alerts:
                    for alert in alerts:
                        severity_icon = {
                            AlertSeverity.INFO: "ℹ️",
                            AlertSeverity.WARNING: "⚠️",
                            AlertSeverity.ERROR: "🚨",
                            AlertSeverity.CRITICAL: "🆘"
                        }[alert.severity]
                        
                        print(f"       {severity_icon} {alert.alert_type}: {alert.message}")
                        if alert.recommended_action:
                            print(f"         → {alert.recommended_action}")
                else:
                    print(f"       ✅ Position healthy")
                    
            except Exception as e:
                print(f"       ❌ Error checking position: {e}")
    
    # Step 7: Display alert summary
    print(f"\n📢 Step 7: Alert Summary")
    if all_alerts:
        print(f"   - Total alerts generated: {len(all_alerts)}")
        
        # Group by severity
        by_severity = {}
        for alert in all_alerts:
            severity = alert.severity.value
            if severity not in by_severity:
                by_severity[severity] = 0
            by_severity[severity] += 1
        
        for severity, count in by_severity.items():
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "critical": "🆘"}[severity]
            print(f"     {icon} {severity.title()}: {count}")
        
        # Show actionable alerts
        actionable = [a for a in all_alerts if a.is_actionable]
        print(f"   - Actionable alerts: {len(actionable)}")
        
        if actionable:
            print("\n   🎯 Recommended Actions:")
            for i, alert in enumerate(actionable[:3], 1):  # Show top 3
                print(f"     {i}. {alert.position_id}: {alert.recommended_action}")
    else:
        print("   ✅ No alerts - all positions healthy!")
    
    # Step 8: Test specific calculations
    print(f"\n🧮 Step 8: Testing Specific Calculations")
    
    # Test impermanent loss calculation
    yield_monitor = next(m for m in monitor.position_monitors 
                        if isinstance(m, YieldFarmingPositionMonitor))
    
    yield_pos = positions["YIELD_001"]
    initial_prices = {"ETH_MAINNET_WETH": 2000.0, "ETH_MAINNET_USDC": 1.0}
    
    il_percent = await yield_monitor._calculate_impermanent_loss(
        yield_pos.exposures, initial_prices
    )
    print(f"   - Impermanent Loss: {il_percent:.2f}%")
    
    # Test health factor calculation
    borrowing_monitor = next(m for m in monitor.position_monitors 
                            if isinstance(m, BorrowingPositionMonitor))
    
    borrow_pos = positions["BORROW_001"]
    health_factor = borrowing_monitor._calculate_health_factor(
        collateral_value=borrow_pos.metadata["collateral_value_usd"],
        debt_value=borrow_pos.metadata["debt_value_usd"],
        liquidation_threshold=0.75  # 75% LTV
    )
    print(f"   - Health Factor: {health_factor:.2f}")
    
    # Test risk level classification
    risk_level = borrowing_monitor._get_risk_level(health_factor)
    print(f"   - Risk Level: {risk_level.value}")
    
    # Step 9: Simulate monitoring loop (short demo)
    print(f"\n🔄 Step 9: Simulating Monitoring Loop")
    print("   Starting brief monitoring simulation...")
    
    # Start monitoring for a few seconds
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    
    # Let it run for 3 seconds
    await asyncio.sleep(3)
    
    # Stop monitoring
    await monitor.stop_monitoring()
    
    print(f"   - Monitoring stopped")
    print(f"   - Is monitoring: {monitor.is_monitoring}")
    print(f"   - Alert history: {len(monitor.alert_history)} total alerts")
    
    # Step 10: Final statistics
    print(f"\n📈 Step 10: Final Statistics")
    stats = {
        "positions_monitored": len(positions),
        "specialized_monitors": len(monitor.position_monitors),
        "alerts_generated": len(all_alerts),
        "critical_alerts": len([a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]),
        "actionable_alerts": len([a for a in all_alerts if a.is_actionable]),
        "portfolio_value": portfolio_health["total_value_usd"],
        "portfolio_pnl": portfolio_health["unrealized_pnl_usd"],
        "liquidation_risk": portfolio_health["liquidation_risk_score"]
    }
    
    print(f"   - Positions monitored: {stats['positions_monitored']}")
    print(f"   - Alerts generated: {stats['alerts_generated']}")
    print(f"   - Critical alerts: {stats['critical_alerts']}")
    print(f"   - Portfolio value: ${stats['portfolio_value']:,.0f}")
    print(f"   - Portfolio P&L: ${stats['portfolio_pnl']:,.0f}")
    print(f"   - Liquidation risk: {stats['liquidation_risk']:.1%}")
    
    print("\n✅ Position Monitor Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("• Specialized monitoring for different position types")
    print("• Real-time health factor and IL calculations")
    print("• Risk-based alert generation with severity levels")
    print("• Portfolio-wide health assessment")
    print("• Actionable recommendations for position management")
    print("• Multi-frequency monitoring loops (main, high-freq, daily)")
    print("• Integration with DeltaTracker and AssetOracle")


if __name__ == "__main__":
    asyncio.run(demonstrate_position_monitor())