#!/usr/bin/env python3
"""
Principal Token (pT) and Yield Token (yT) Position Monitor Test.

This script demonstrates the specialized monitoring for Pendle-like protocol
positions including maturity risk, time decay, yield deviations, and balance checks.
"""
import sys
import os
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import standalone components for testing
from test_position_monitor_standalone import (
    PositionType, AlertSeverity, PositionAlert, MonitoringConfig,
    BasePositionMonitor
)


class PrincipalYieldPositionMonitor(BasePositionMonitor):
    """
    Monitor for Principal Token (pT) and Yield Token (yT) positions on Pendle-like protocols.
    
    This is a standalone version for testing without complex dependencies.
    """
    
    def __init__(self, config: MonitoringConfig, asset_oracle):
        super().__init__(config, asset_oracle)
        # Add PRINCIPAL_YIELD to enum if not exists
        if not hasattr(PositionType, 'PRINCIPAL_YIELD'):
            PositionType.PRINCIPAL_YIELD = "principal_yield"
        self.position_type = PositionType.PRINCIPAL_YIELD
    
    async def check_position_health(self, position) -> list[PositionAlert]:
        """Check health of pT/yT positions with focus on maturity and yield risks."""
        alerts = []
        
        try:
            # Extract position metadata
            metadata = getattr(position, 'metadata', {})
            
            # Get position details
            maturity_date = metadata.get('maturity_date')
            position_side = metadata.get('position_side')  # 'principal', 'yield', or 'both'
            underlying_asset = metadata.get('underlying_asset')
            initial_implied_yield = metadata.get('initial_implied_yield')
            current_implied_yield = metadata.get('current_implied_yield')
            
            # Calculate time to maturity
            if maturity_date:
                if isinstance(maturity_date, str):
                    maturity_date = datetime.fromisoformat(maturity_date.replace('Z', '+00:00'))
                elif isinstance(maturity_date, (int, float)):
                    maturity_date = datetime.fromtimestamp(maturity_date, tz=timezone.utc)
                
                days_to_maturity = (maturity_date - datetime.now(timezone.utc)).days
                hours_to_maturity = (maturity_date - datetime.now(timezone.utc)).total_seconds() / 3600
                
                # Maturity warnings
                maturity_alerts = await self._check_maturity_risk(
                    position, days_to_maturity, hours_to_maturity
                )
                alerts.extend(maturity_alerts)
                
                # Time decay analysis
                time_decay_alerts = await self._check_time_decay(
                    position, days_to_maturity, metadata
                )
                alerts.extend(time_decay_alerts)
            
            # Yield rate analysis
            if initial_implied_yield and current_implied_yield:
                yield_alerts = await self._check_yield_deviation(
                    position, initial_implied_yield, current_implied_yield
                )
                alerts.extend(yield_alerts)
            
            # Principal/Yield token balance analysis
            if position_side == 'both':
                balance_alerts = await self._check_pt_yt_balance(position, metadata)
                alerts.extend(balance_alerts)
            
            # Underlying asset exposure analysis
            if underlying_asset:
                exposure_alerts = await self._check_underlying_exposure(
                    position, underlying_asset, metadata
                )
                alerts.extend(exposure_alerts)
            
        except Exception as e:
            print(f"Error checking pT/yT position health for {position.position_id}: {e}")
            alerts.append(PositionAlert(
                position_id=position.position_id,
                position_type=self.position_type,
                alert_type="monitoring_error",
                severity=AlertSeverity.WARNING,
                message=f"Failed to monitor position: {str(e)}",
                details={"error": str(e)},
                is_actionable=False
            ))
        
        return alerts
    
    async def _check_maturity_risk(
        self, 
        position, 
        days_to_maturity: int, 
        hours_to_maturity: float
    ) -> list[PositionAlert]:
        """Check for maturity-related risks."""
        alerts = []
        
        # Critical: Very close to maturity
        if days_to_maturity <= 1:  # Critical threshold: 1 day
            alerts.append(PositionAlert(
                position_id=position.position_id,
                position_type=self.position_type,
                alert_type="maturity_critical",
                severity=AlertSeverity.CRITICAL,
                message=f"Position expires in {hours_to_maturity:.1f} hours",
                details={
                    "days_to_maturity": days_to_maturity,
                    "hours_to_maturity": hours_to_maturity,
                    "maturity_critical_threshold": 1
                },
                recommended_action="URGENT: Close position or prepare for settlement immediately"
            ))
        
        # Warning: Approaching maturity
        elif days_to_maturity <= 7:  # Warning threshold: 7 days
            alerts.append(PositionAlert(
                position_id=position.position_id,
                position_type=self.position_type,
                alert_type="maturity_warning",
                severity=AlertSeverity.WARNING,
                message=f"Position expires in {days_to_maturity} days",
                details={
                    "days_to_maturity": days_to_maturity,
                    "hours_to_maturity": hours_to_maturity,
                    "maturity_warning_threshold": 7
                },
                recommended_action="Consider closing position or preparing for maturity settlement"
            ))
        
        # Info: Maturity timeline update
        elif days_to_maturity <= 30:  # Monthly updates for positions <30 days
            alerts.append(PositionAlert(
                position_id=position.position_id,
                position_type=self.position_type,
                alert_type="maturity_info",
                severity=AlertSeverity.INFO,
                message=f"Position matures in {days_to_maturity} days",
                details={
                    "days_to_maturity": days_to_maturity,
                    "maturity_category": "medium_term" if days_to_maturity > 14 else "short_term"
                },
                is_actionable=False
            ))
        
        return alerts
    
    async def _check_time_decay(
        self, 
        position, 
        days_to_maturity: int, 
        metadata: dict
    ) -> list[PositionAlert]:
        """Check for time decay (theta) effects on position value."""
        alerts = []
        
        try:
            # Calculate time decay rate based on position type and maturity
            position_side = metadata.get('position_side', 'both')
            initial_time_to_maturity = metadata.get('initial_days_to_maturity', 365)
            
            if initial_time_to_maturity > 0:
                time_elapsed_ratio = 1 - (days_to_maturity / initial_time_to_maturity)
                
                # Estimate time decay impact (simplified model)
                if position_side in ['yield', 'both']:
                    # Yield tokens experience more time decay
                    estimated_decay = time_elapsed_ratio * 0.3  # Up to 30% decay
                else:
                    # Principal tokens less affected by time decay
                    estimated_decay = time_elapsed_ratio * 0.1  # Up to 10% decay
                
                if estimated_decay > 0.15:  # 15% threshold
                    severity = (AlertSeverity.WARNING if estimated_decay < 0.25 
                              else AlertSeverity.ERROR)
                    
                    alerts.append(PositionAlert(
                        position_id=position.position_id,
                        position_type=self.position_type,
                        alert_type="time_decay",
                        severity=severity,
                        message=f"Significant time decay: {estimated_decay:.1%}",
                        details={
                            "estimated_time_decay": estimated_decay,
                            "time_elapsed_ratio": time_elapsed_ratio,
                            "days_to_maturity": days_to_maturity,
                            "position_side": position_side,
                            "decay_threshold": 0.15
                        },
                        recommended_action=(
                            "Consider closing position to avoid further time decay" 
                            if estimated_decay > 0.2 else
                            "Monitor time decay closely as maturity approaches"
                        )
                    ))
        
        except Exception as e:
            print(f"Failed to calculate time decay for {position.position_id}: {e}")
        
        return alerts
    
    async def _check_yield_deviation(
        self, 
        position, 
        initial_yield: float, 
        current_yield: float
    ) -> list[PositionAlert]:
        """Check for significant changes in implied yield rates."""
        alerts = []
        
        yield_change_percent = ((current_yield - initial_yield) / initial_yield) * 100
        abs_yield_change = abs(yield_change_percent)
        
        if abs_yield_change > 25.0:  # 25% threshold
            # Determine if this is good or bad based on position side
            metadata = getattr(position, 'metadata', {})
            position_side = metadata.get('position_side', 'both')
            
            is_favorable = False
            if position_side == 'principal' and yield_change_percent > 0:
                is_favorable = True  # Higher yields favor principal tokens
            elif position_side == 'yield' and yield_change_percent < 0:
                is_favorable = True  # Lower yields favor yield tokens
            
            severity = (AlertSeverity.INFO if is_favorable 
                       else AlertSeverity.WARNING if abs_yield_change < 50
                       else AlertSeverity.ERROR)
            
            direction = "increased" if yield_change_percent > 0 else "decreased"
            impact = "favorable" if is_favorable else "unfavorable"
            
            alerts.append(PositionAlert(
                position_id=position.position_id,
                position_type=self.position_type,
                alert_type="yield_deviation",
                severity=severity,
                message=f"Implied yield {direction} by {abs_yield_change:.1f}% ({impact})",
                details={
                    "initial_yield": initial_yield,
                    "current_yield": current_yield,
                    "yield_change_percent": yield_change_percent,
                    "position_side": position_side,
                    "is_favorable": is_favorable,
                    "threshold": 25.0
                },
                recommended_action=(
                    "Consider rebalancing position based on yield environment changes" 
                    if not is_favorable else
                    "Monitor for profit-taking opportunities"
                )
            ))
        
        return alerts
    
    async def _check_pt_yt_balance(
        self, 
        position, 
        metadata: dict
    ) -> list[PositionAlert]:
        """Check balance between principal and yield token exposures."""
        alerts = []
        
        try:
            pt_amount = metadata.get('pt_amount', 0)
            yt_amount = metadata.get('yt_amount', 0)
            target_ratio = metadata.get('target_pt_yt_ratio', 1.0)  # 1:1 by default
            
            if pt_amount > 0 and yt_amount > 0:
                current_ratio = pt_amount / yt_amount
                ratio_deviation = abs(current_ratio - target_ratio) / target_ratio
                
                if ratio_deviation > 0.15:  # 15% deviation threshold
                    severity = (AlertSeverity.WARNING if ratio_deviation < 0.3 
                              else AlertSeverity.ERROR)
                    
                    which_side = "principal" if current_ratio > target_ratio else "yield"
                    
                    alerts.append(PositionAlert(
                        position_id=position.position_id,
                        position_type=self.position_type,
                        alert_type="pt_yt_imbalance",
                        severity=severity,
                        message=f"pT/yT ratio imbalance: {current_ratio:.2f} (target: {target_ratio:.2f})",
                        details={
                            "pt_amount": pt_amount,
                            "yt_amount": yt_amount,
                            "current_ratio": current_ratio,
                            "target_ratio": target_ratio,
                            "ratio_deviation": ratio_deviation,
                            "excess_side": which_side
                        },
                        recommended_action=f"Rebalance by adjusting {which_side} token exposure"
                    ))
        
        except Exception as e:
            print(f"Failed to check pT/yT balance for {position.position_id}: {e}")
        
        return alerts
    
    async def _check_underlying_exposure(
        self, 
        position, 
        underlying_asset: str, 
        metadata: dict
    ) -> list[PositionAlert]:
        """Check exposure to underlying asset price movements."""
        alerts = []
        
        try:
            # Mock current price (in real implementation, would use asset oracle)
            price_changes = {
                "ETH_MAINNET_WETH": 15.0,    # ETH up 15%
                "ETH_MAINNET_STETH": -8.0,   # stETH down 8%
                "ETH_MAINNET_USDC": 0.1      # USDC stable
            }
            
            price_change_percent = price_changes.get(underlying_asset, 0.0)
            initial_price = metadata.get('initial_underlying_price', 100.0)
            current_price = initial_price * (1 + price_change_percent / 100)
            
            # High volatility warning
            if abs(price_change_percent) > 20:  # 20% price movement
                position_side = metadata.get('position_side', 'both')
                
                # Assess impact based on position type
                if abs(price_change_percent) > 40:
                    impact_severity = AlertSeverity.ERROR
                elif abs(price_change_percent) > 30:
                    impact_severity = AlertSeverity.WARNING
                else:
                    impact_severity = AlertSeverity.INFO
                
                alerts.append(PositionAlert(
                    position_id=position.position_id,
                    position_type=self.position_type,
                    alert_type="underlying_volatility",
                    severity=impact_severity,
                    message=f"Underlying asset {underlying_asset} moved {price_change_percent:+.1f}%",
                    details={
                        "underlying_asset": underlying_asset,
                        "current_price": current_price,
                        "initial_price": initial_price,
                        "price_change_percent": price_change_percent,
                        "position_side": position_side
                    },
                    recommended_action=(
                        "Review position sizing due to high underlying volatility"
                    )
                ))
        
        except Exception as e:
            print(f"Failed to check underlying exposure for {position.position_id}: {e}")
        
        return alerts


def create_pt_yt_test_positions():
    """Create various pT/yT test positions with different scenarios."""
    now = datetime.now(timezone.utc)
    positions = {}
    
    # Add PRINCIPAL_YIELD to the PositionType enum if not exists
    if not hasattr(PositionType, 'PRINCIPAL_YIELD'):
        PositionType.PRINCIPAL_YIELD = "principal_yield"
    
    # 1. Position expiring very soon (critical)
    critical_maturity = Mock()
    critical_maturity.position_id = "PT_YT_CRITICAL_001"
    critical_maturity.position_type = PositionType.PRINCIPAL_YIELD
    critical_maturity.created_at = now - timedelta(days=180)  # 6 months old
    critical_maturity.current_value_usd = 48500.0
    critical_maturity.initial_value_usd = 50000.0
    critical_maturity.metadata = {
        "maturity_date": (now + timedelta(hours=18)).isoformat(),  # Expires in 18 hours
        "position_side": "both",
        "underlying_asset": "ETH_MAINNET_STETH",
        "initial_implied_yield": 0.055,  # 5.5%
        "current_implied_yield": 0.048,  # 4.8% - yield dropped
        "initial_days_to_maturity": 183,
        "pt_amount": 25.0,
        "yt_amount": 25.0,
        "target_pt_yt_ratio": 1.0,
        "initial_underlying_price": 2000.0,
        "protocol": "pendle"
    }
    positions[critical_maturity.position_id] = critical_maturity
    
    # 2. Position with high time decay (warning)
    high_decay = Mock()
    high_decay.position_id = "PT_YT_DECAY_002"
    high_decay.position_type = PositionType.PRINCIPAL_YIELD
    high_decay.created_at = now - timedelta(days=330)  # 11 months old
    high_decay.current_value_usd = 28200.0  # Significant decay
    high_decay.initial_value_usd = 40000.0
    high_decay.metadata = {
        "maturity_date": (now + timedelta(days=35)).isoformat(),  # 35 days left
        "position_side": "yield",  # Only yield tokens (more decay)
        "underlying_asset": "ETH_MAINNET_WETH",
        "initial_implied_yield": 0.065,
        "current_implied_yield": 0.078,  # Yield increased (bad for yT)
        "initial_days_to_maturity": 365,
        "yt_amount": 20.0,
        "initial_underlying_price": 1800.0,
        "protocol": "pendle"
    }
    positions[high_decay.position_id] = high_decay
    
    # 3. Position with significant yield deviation (unfavorable)
    yield_deviation = Mock()
    yield_deviation.position_id = "PT_YT_YIELD_003"
    yield_deviation.position_type = PositionType.PRINCIPAL_YIELD
    yield_deviation.created_at = now - timedelta(days=45)
    yield_deviation.current_value_usd = 73500.0
    yield_deviation.initial_value_usd = 75000.0
    yield_deviation.metadata = {
        "maturity_date": (now + timedelta(days=120)).isoformat(),  # 4 months left
        "position_side": "principal",
        "underlying_asset": "ETH_MAINNET_STETH",
        "initial_implied_yield": 0.045,  # 4.5%
        "current_implied_yield": 0.028,  # 2.8% - big drop (bad for pT)
        "initial_days_to_maturity": 165,
        "pt_amount": 37.5,
        "initial_underlying_price": 1950.0,
        "protocol": "pendle"
    }
    positions[yield_deviation.position_id] = yield_deviation
    
    # 4. Position with pT/yT imbalance (error)
    imbalanced = Mock()
    imbalanced.position_id = "PT_YT_IMBALANCE_004"
    imbalanced.position_type = PositionType.PRINCIPAL_YIELD
    imbalanced.created_at = now - timedelta(days=90)
    imbalanced.current_value_usd = 92800.0
    imbalanced.initial_value_usd = 90000.0
    imbalanced.metadata = {
        "maturity_date": (now + timedelta(days=275)).isoformat(),  # 9 months left
        "position_side": "both",
        "underlying_asset": "ETH_MAINNET_WETH",
        "initial_implied_yield": 0.052,
        "current_implied_yield": 0.058,  # Slight increase
        "initial_days_to_maturity": 365,
        "pt_amount": 50.0,   # Too much principal
        "yt_amount": 30.0,   # Too little yield
        "target_pt_yt_ratio": 1.0,  # Target 1:1
        "initial_underlying_price": 2100.0,
        "protocol": "pendle"
    }
    positions[imbalanced.position_id] = imbalanced
    
    # 5. Healthy position with minor maturity warning
    healthy_warning = Mock()
    healthy_warning.position_id = "PT_YT_HEALTHY_005"
    healthy_warning.position_type = PositionType.PRINCIPAL_YIELD
    healthy_warning.created_at = now - timedelta(days=30)
    healthy_warning.current_value_usd = 126500.0
    healthy_warning.initial_value_usd = 125000.0
    healthy_warning.metadata = {
        "maturity_date": (now + timedelta(days=5)).isoformat(),  # 5 days left (warning)
        "position_side": "both",
        "underlying_asset": "ETH_MAINNET_STETH",
        "initial_implied_yield": 0.048,
        "current_implied_yield": 0.051,  # Small favorable increase
        "initial_days_to_maturity": 35,
        "pt_amount": 62.5,
        "yt_amount": 62.5,
        "target_pt_yt_ratio": 1.0,
        "initial_underlying_price": 2000.0,
        "protocol": "pendle"
    }
    positions[healthy_warning.position_id] = healthy_warning
    
    # 6. Long-term position with minimal issues
    long_term = Mock()
    long_term.position_id = "PT_YT_LONGTERM_006"
    long_term.position_type = PositionType.PRINCIPAL_YIELD
    long_term.created_at = now - timedelta(days=15)
    long_term.current_value_usd = 201500.0
    long_term.initial_value_usd = 200000.0
    long_term.metadata = {
        "maturity_date": (now + timedelta(days=350)).isoformat(),  # ~1 year left
        "position_side": "yield",
        "underlying_asset": "ETH_MAINNET_WETH",
        "initial_implied_yield": 0.062,
        "current_implied_yield": 0.059,  # Small favorable decrease for yT
        "initial_days_to_maturity": 365,
        "yt_amount": 100.0,
        "initial_underlying_price": 2000.0,
        "protocol": "pendle"
    }
    positions[long_term.position_id] = long_term
    
    return positions


async def demonstrate_pt_yt_monitoring():
    """Demonstrate pT/yT position monitoring functionality."""
    print("🪙 Principal/Yield Token (pT/yT) Position Monitor Demonstration")
    print("=" * 80)
    
    # Step 1: Create configuration
    print("\n⚙️  Step 1: Creating pT/yT Monitoring Configuration")
    config = MonitoringConfig(
        check_interval_seconds=20,
        max_unrealized_loss_percent=12.0,
        impermanent_loss_threshold=4.0
    )
    
    # Add pT/yT specific config attributes
    config.maturity_warning_days = 7
    config.maturity_critical_days = 1
    config.time_decay_threshold = 0.15
    config.yield_rate_deviation_threshold_pt = 30.0
    config.implied_yield_deviation_threshold = 25.0
    
    print(f"   ✅ pT/yT Config created:")
    print(f"     - Maturity warning: {config.maturity_warning_days} days")
    print(f"     - Maturity critical: {config.maturity_critical_days} days")
    print(f"     - Time decay threshold: {config.time_decay_threshold:.1%}")
    print(f"     - Yield deviation threshold: {config.implied_yield_deviation_threshold}%")
    
    # Step 2: Create monitor
    print("\n🎯 Step 2: Creating pT/yT Position Monitor")
    
    # Mock asset oracle
    asset_oracle = Mock()
    asset_oracle.get_price_usd = AsyncMock(side_effect=lambda asset_id: {
        "ETH_MAINNET_WETH": 2300.0,   # ETH up 15%
        "ETH_MAINNET_STETH": 1840.0,  # stETH down 8%
        "ETH_MAINNET_USDC": 1.0       # USDC stable
    }.get(asset_id, 1.0))
    
    monitor = PrincipalYieldPositionMonitor(config, asset_oracle)
    
    print(f"   ✅ Monitor created:")
    print(f"     - Position type: {monitor.position_type}")
    print(f"     - Asset oracle: Configured with mock prices")
    
    # Step 3: Create test positions
    print("\n💼 Step 3: Creating pT/yT Test Positions")
    positions = create_pt_yt_test_positions()
    
    for pos_id, position in positions.items():
        metadata = position.metadata
        maturity_date = datetime.fromisoformat(metadata['maturity_date'].replace('Z', '+00:00'))
        days_to_maturity = (maturity_date - datetime.now(timezone.utc)).days
        
        print(f"   - {pos_id}:")
        print(f"     Side: {metadata.get('position_side', 'N/A')}")
        print(f"     Maturity: {days_to_maturity} days")
        print(f"     Yield: {metadata.get('initial_implied_yield', 0):.1%} → {metadata.get('current_implied_yield', 0):.1%}")
        print(f"     Value: ${position.current_value_usd:,.0f}")
    
    # Step 4: Monitor all positions
    print("\n🔍 Step 4: Monitoring pT/yT Positions")
    all_alerts = []
    
    for position in positions.values():
        print(f"\n   📋 Checking {position.position_id}...")
        alerts = await monitor.check_position_health(position)
        all_alerts.extend(alerts)
        
        if alerts:
            for alert in alerts:
                severity_icon = {
                    AlertSeverity.INFO: "ℹ️",
                    AlertSeverity.WARNING: "⚠️",
                    AlertSeverity.ERROR: "🚨",
                    AlertSeverity.CRITICAL: "🆘"
                }[alert.severity]
                
                print(f"     {severity_icon} {alert.alert_type}: {alert.message}")
                if alert.recommended_action:
                    print(f"       → {alert.recommended_action}")
        else:
            print(f"     ✅ Position healthy")
    
    # Step 5: Alert analysis by type
    print(f"\n📊 Step 5: pT/yT Alert Analysis")
    print(f"   🚨 Total alerts generated: {len(all_alerts)}")
    
    # Group by alert type
    alert_types = {}
    severity_counts = {}
    
    for alert in all_alerts:
        alert_type = alert.alert_type
        severity = alert.severity.value
        
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print(f"\n   📋 By Alert Type:")
    for alert_type, count in alert_types.items():
        print(f"     • {alert_type.replace('_', ' ').title()}: {count}")
    
    print(f"\n   🎯 By Severity:")
    for severity, count in severity_counts.items():
        icon = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "critical": "🆘"}[severity]
        print(f"     {icon} {severity.title()}: {count}")
    
    # Step 6: Maturity analysis
    print(f"\n📅 Step 6: Maturity Analysis")
    
    maturity_analysis = {
        "expired_soon": 0,      # <1 day
        "short_term": 0,        # 1-7 days
        "medium_term": 0,       # 7-30 days
        "long_term": 0          # >30 days
    }
    
    total_value_by_maturity = {
        "expired_soon": 0,
        "short_term": 0,
        "medium_term": 0,
        "long_term": 0
    }
    
    for position in positions.values():
        metadata = position.metadata
        maturity_date = datetime.fromisoformat(metadata['maturity_date'].replace('Z', '+00:00'))
        days_to_maturity = (maturity_date - datetime.now(timezone.utc)).days
        
        if days_to_maturity <= 1:
            category = "expired_soon"
        elif days_to_maturity <= 7:
            category = "short_term"
        elif days_to_maturity <= 30:
            category = "medium_term"
        else:
            category = "long_term"
        
        maturity_analysis[category] += 1
        total_value_by_maturity[category] += position.current_value_usd
    
    print(f"   📊 Position distribution:")
    for category, count in maturity_analysis.items():
        value = total_value_by_maturity[category]
        category_name = category.replace('_', ' ').title()
        
        if category == "expired_soon":
            icon = "🆘"
        elif category == "short_term":
            icon = "⚠️"
        elif category == "medium_term":
            icon = "ℹ️"
        else:
            icon = "✅"
        
        print(f"     {icon} {category_name}: {count} positions, ${value:,.0f}")
    
    # Step 7: Risk assessment
    print(f"\n⚖️  Step 7: Risk Assessment")
    
    # Calculate overall risk score
    risk_factors = {
        "critical_alerts": len([a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]) * 40,
        "error_alerts": len([a for a in all_alerts if a.severity == AlertSeverity.ERROR]) * 20,
        "warning_alerts": len([a for a in all_alerts if a.severity == AlertSeverity.WARNING]) * 10,
        "expiring_soon": maturity_analysis["expired_soon"] * 30,
        "short_term_positions": maturity_analysis["short_term"] * 15
    }
    
    total_risk_score = sum(risk_factors.values())
    max_possible_risk = len(positions) * 50  # Theoretical max
    normalized_risk = min(100, (total_risk_score / max_possible_risk) * 100) if max_possible_risk > 0 else 0
    
    print(f"   📊 Risk Factor Breakdown:")
    for factor, score in risk_factors.items():
        if score > 0:
            print(f"     • {factor.replace('_', ' ').title()}: {score} points")
    
    print(f"\n   🎯 Overall Risk Assessment:")
    print(f"     - Risk Score: {total_risk_score}/{max_possible_risk} ({normalized_risk:.0f}%)")
    
    if normalized_risk >= 70:
        risk_status = "🔴 HIGH RISK"
        recommendation = "Immediate attention required - multiple critical issues"
    elif normalized_risk >= 40:
        risk_status = "🟡 MEDIUM RISK"
        recommendation = "Monitor closely and address warnings promptly"
    else:
        risk_status = "🟢 LOW RISK"
        recommendation = "Continue regular monitoring"
    
    print(f"     - Status: {risk_status}")
    print(f"     - Recommendation: {recommendation}")
    
    # Step 8: Action prioritization
    print(f"\n🎯 Step 8: Action Prioritization")
    
    # Sort alerts by priority
    priority_order = {
        AlertSeverity.CRITICAL: 0,
        AlertSeverity.ERROR: 1,
        AlertSeverity.WARNING: 2,
        AlertSeverity.INFO: 3
    }
    
    actionable_alerts = [a for a in all_alerts if a.is_actionable and a.recommended_action]
    actionable_alerts.sort(key=lambda x: (priority_order[x.severity], x.position_id))
    
    if actionable_alerts:
        print(f"   📋 Priority Actions ({len(actionable_alerts)} total):")
        
        for i, alert in enumerate(actionable_alerts[:5], 1):  # Show top 5
            urgency = "🔥 URGENT" if alert.severity == AlertSeverity.CRITICAL else "📌 NORMAL"
            
            # Add time estimate
            if alert.alert_type == "maturity_critical":
                time_est = "⏰ <24 hours"
            elif alert.alert_type == "maturity_warning":
                time_est = "📅 Within 7 days"
            elif alert.alert_type == "time_decay":
                time_est = "📈 Monitor continuously"
            else:
                time_est = "⏱️ Next review cycle"
            
            print(f"\n     {i}. {urgency} - {alert.position_id}")
            print(f"        Issue: {alert.message}")
            print(f"        Action: {alert.recommended_action}")
            print(f"        Timeline: {time_est}")
    else:
        print(f"   ✅ No immediate actions required")
    
    print(f"\n✅ pT/yT Position Monitor Demonstration Complete!")
    print(f"\nKey pT/yT Monitoring Features:")
    print(f"• 🕐 Maturity date tracking with time-based alerts")
    print(f"• ⏳ Time decay (theta) calculation and warnings")
    print(f"• 📈 Implied yield rate deviation monitoring")
    print(f"• ⚖️ Principal/Yield token balance tracking")
    print(f"• 💰 Underlying asset exposure analysis")
    print(f"• 🎯 Position-side aware risk assessment")
    print(f"• 📊 Portfolio-wide maturity distribution analysis")


if __name__ == "__main__":
    asyncio.run(demonstrate_pt_yt_monitoring())