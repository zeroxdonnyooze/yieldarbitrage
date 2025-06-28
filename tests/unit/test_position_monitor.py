"""
Unit tests for Position Monitor functionality.

Tests the PositionMonitor system, specialized position monitors,
and position health calculations with comprehensive coverage.
"""
import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.monitoring.position_monitor import (
    PositionMonitor, PositionType, RiskLevel, AlertSeverity, PositionAlert, 
    MonitoringConfig, ArbitragePositionMonitor, YieldFarmingPositionMonitor,
    LendingPositionMonitor, BorrowingPositionMonitor
)
from yield_arbitrage.risk.delta_tracker import DeltaPosition, AssetExposure
from yield_arbitrage.execution.asset_oracle import AssetOracleBase


class TestPositionMonitor:
    """Test main PositionMonitor functionality."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Create test monitoring configuration."""
        return MonitoringConfig(
            check_interval_seconds=5,
            high_frequency_interval_seconds=1,
            max_unrealized_loss_percent=10.0,
            liquidation_threshold_buffer=0.10,
            impermanent_loss_threshold=3.0
        )
    
    @pytest.fixture
    def mock_delta_tracker(self):
        """Create mock delta tracker."""
        tracker = Mock()
        tracker.get_all_positions = Mock(return_value={})
        tracker.get_position = Mock(return_value=None)
        tracker.calculate_portfolio_health = AsyncMock(return_value={
            "total_value_usd": 50000.0,
            "unrealized_pnl_usd": 1250.0,
            "liquidation_risk_score": 0.15,
            "position_count": 3
        })
        return tracker
    
    @pytest.fixture
    def mock_asset_oracle(self):
        """Create mock asset oracle."""
        oracle = Mock(spec=AssetOracleBase)
        oracle.get_price_usd = AsyncMock(return_value=2000.0)
        oracle.is_stable_asset = Mock(return_value=False)
        return oracle
    
    @pytest.fixture
    def mock_execution_logger(self):
        """Create mock execution logger."""
        logger = Mock()
        logger.log_position_alert = AsyncMock(return_value=True)
        return logger
    
    @pytest.fixture
    def position_monitor(self, monitoring_config, mock_delta_tracker, 
                        mock_asset_oracle, mock_execution_logger):
        """Create PositionMonitor instance."""
        return PositionMonitor(
            delta_tracker=mock_delta_tracker,
            asset_oracle=mock_asset_oracle,
            execution_logger=mock_execution_logger,
            config=monitoring_config
        )
    
    def test_position_monitor_initialization(self, position_monitor, monitoring_config):
        """Test PositionMonitor initialization."""
        assert position_monitor.config == monitoring_config
        assert not position_monitor.is_monitoring
        assert len(position_monitor.position_monitors) == 4  # 4 specialized monitors
        assert position_monitor.monitoring_tasks == []
        assert position_monitor.alert_history == []
        assert position_monitor.last_portfolio_health is None
    
    def test_specialized_monitor_registration(self, position_monitor):
        """Test that specialized monitors are properly registered."""
        monitor_types = {type(monitor).__name__ for monitor in position_monitor.position_monitors}
        expected_types = {
            "ArbitragePositionMonitor",
            "YieldFarmingPositionMonitor", 
            "LendingPositionMonitor",
            "BorrowingPositionMonitor"
        }
        assert monitor_types == expected_types
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, position_monitor):
        """Test starting and stopping monitoring."""
        # Test starting monitoring
        await position_monitor.start_monitoring()
        assert position_monitor.is_monitoring
        assert len(position_monitor.monitoring_tasks) == 3  # 3 monitoring loops
        
        # Test stopping monitoring
        await position_monitor.stop_monitoring()
        assert not position_monitor.is_monitoring
        assert len(position_monitor.monitoring_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_check_portfolio_health(self, position_monitor, mock_delta_tracker):
        """Test portfolio health checking."""
        # Setup mock return
        health_data = {
            "total_value_usd": 75000.0,
            "unrealized_pnl_usd": -5000.0,  # 6.67% loss
            "liquidation_risk_score": 0.25,
            "position_count": 4
        }
        mock_delta_tracker.calculate_portfolio_health.return_value = health_data
        
        health = await position_monitor._check_portfolio_health()
        
        assert health == health_data
        assert position_monitor.last_portfolio_health == health_data
        mock_delta_tracker.calculate_portfolio_health.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_portfolio_alert_high_loss(self, position_monitor):
        """Test portfolio alert generation for high unrealized loss."""
        health_data = {
            "total_value_usd": 100000.0,
            "unrealized_pnl_usd": -12000.0,  # 12% loss, above 10% threshold
            "liquidation_risk_score": 0.15,
            "position_count": 3
        }
        
        alert = await position_monitor._generate_portfolio_alert(health_data)
        
        assert alert is not None
        assert alert.alert_type == "unrealized_loss_high"
        assert alert.severity == AlertSeverity.WARNING
        assert "12.0%" in alert.message
        assert alert.is_actionable
    
    @pytest.mark.asyncio
    async def test_generate_portfolio_alert_critical_loss(self, position_monitor):
        """Test portfolio alert generation for critical unrealized loss."""
        health_data = {
            "total_value_usd": 100000.0,
            "unrealized_pnl_usd": -25000.0,  # 25% loss, critical level
            "liquidation_risk_score": 0.45,
            "position_count": 2
        }
        
        alert = await position_monitor._generate_portfolio_alert(health_data)
        
        assert alert is not None
        assert alert.alert_type == "unrealized_loss_critical"
        assert alert.severity == AlertSeverity.CRITICAL
        assert "25.0%" in alert.message
        assert "emergency" in alert.recommended_action.lower()
    
    @pytest.mark.asyncio
    async def test_generate_portfolio_alert_high_liquidation_risk(self, position_monitor):
        """Test portfolio alert generation for high liquidation risk."""
        health_data = {
            "total_value_usd": 50000.0,
            "unrealized_pnl_usd": 1000.0,
            "liquidation_risk_score": 0.80,  # High liquidation risk
            "position_count": 5
        }
        
        alert = await position_monitor._generate_portfolio_alert(health_data)
        
        assert alert is not None
        assert alert.alert_type == "liquidation_risk_high"
        assert alert.severity == AlertSeverity.ERROR
        assert "80.0%" in alert.message
        assert "reduce" in alert.recommended_action.lower()
    
    @pytest.mark.asyncio
    async def test_no_alert_healthy_portfolio(self, position_monitor):
        """Test no alert generation for healthy portfolio."""
        health_data = {
            "total_value_usd": 100000.0,
            "unrealized_pnl_usd": 2500.0,  # 2.5% gain
            "liquidation_risk_score": 0.05,  # Low risk
            "position_count": 3
        }
        
        alert = await position_monitor._generate_portfolio_alert(health_data)
        assert alert is None
    
    def test_position_alert_creation(self):
        """Test PositionAlert creation and conversion."""
        alert = PositionAlert(
            position_id="test_pos_123",
            position_type=PositionType.YIELD_FARMING,
            alert_type="impermanent_loss",
            severity=AlertSeverity.WARNING,
            message="IL threshold exceeded",
            details={"il_percent": 7.5},
            recommended_action="Consider reducing LP position"
        )
        
        assert alert.position_id == "test_pos_123"
        assert alert.position_type == PositionType.YIELD_FARMING
        assert alert.alert_type == "impermanent_loss"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.is_actionable
        
        # Test conversion to dict
        alert_dict = alert.to_dict()
        assert alert_dict["position_id"] == "test_pos_123"
        assert alert_dict["position_type"] == "yield_farming"
        assert alert_dict["severity"] == "warning"
        assert alert_dict["details"]["il_percent"] == 7.5
    
    def test_monitoring_config_defaults(self):
        """Test MonitoringConfig default values."""
        config = MonitoringConfig()
        
        assert config.check_interval_seconds == 30
        assert config.high_frequency_interval_seconds == 5
        assert config.daily_health_check_interval == 86400
        assert config.max_unrealized_loss_percent == 15.0
        assert config.liquidation_threshold_buffer == 0.05
        assert config.impermanent_loss_threshold == 5.0
        assert config.max_position_size_usd == 100000.0
    
    @pytest.mark.asyncio
    async def test_monitor_positions_with_alerts(self, position_monitor, mock_delta_tracker):
        """Test monitoring positions that generate alerts."""
        # Create mock position with high loss
        mock_position = Mock()
        mock_position.position_id = "pos_123"
        mock_position.position_type = PositionType.YIELD_FARMING
        mock_position.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_position.current_value_usd = 8500.0  # Down from 10000
        mock_position.initial_value_usd = 10000.0
        mock_position.exposures = {
            "ETH_MAINNET_WETH": Mock(amount=Decimal("1.0")),
            "ETH_MAINNET_USDC": Mock(amount=Decimal("1500.0"))
        }
        
        mock_delta_tracker.get_all_positions.return_value = {"pos_123": mock_position}
        
        # Mock specialized monitor to return an alert
        mock_yield_monitor = Mock()
        mock_yield_monitor.check_position_health = AsyncMock(return_value=[
            PositionAlert(
                position_id="pos_123",
                position_type=PositionType.YIELD_FARMING,
                alert_type="unrealized_loss",
                severity=AlertSeverity.WARNING,
                message="Position down 15%",
                details={"loss_percent": 15.0}
            )
        ])
        
        # Replace the yield farming monitor
        position_monitor.position_monitors = [mock_yield_monitor]
        
        alerts = await position_monitor._monitor_positions()
        
        assert len(alerts) == 1
        assert alerts[0].position_id == "pos_123"
        assert alerts[0].alert_type == "unrealized_loss"
        assert alerts[0].severity == AlertSeverity.WARNING


class TestSpecializedPositionMonitors:
    """Test specialized position monitor classes."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Create test monitoring configuration."""
        return MonitoringConfig(impermanent_loss_threshold=5.0)
    
    @pytest.fixture
    def mock_asset_oracle(self):
        """Create mock asset oracle."""
        oracle = Mock(spec=AssetOracleBase)
        oracle.get_price_usd = AsyncMock(return_value=2000.0)
        oracle.is_stable_asset = Mock(return_value=False)
        return oracle
    
    @pytest.fixture
    def arbitrage_monitor(self, monitoring_config, mock_asset_oracle):
        """Create ArbitragePositionMonitor."""
        return ArbitragePositionMonitor(monitoring_config, mock_asset_oracle)
    
    @pytest.fixture
    def yield_farming_monitor(self, monitoring_config, mock_asset_oracle):
        """Create YieldFarmingPositionMonitor."""
        return YieldFarmingPositionMonitor(monitoring_config, mock_asset_oracle)
    
    @pytest.fixture
    def lending_monitor(self, monitoring_config, mock_asset_oracle):
        """Create LendingPositionMonitor."""
        return LendingPositionMonitor(monitoring_config, mock_asset_oracle)
    
    @pytest.fixture
    def borrowing_monitor(self, monitoring_config, mock_asset_oracle):
        """Create BorrowingPositionMonitor."""
        return BorrowingPositionMonitor(monitoring_config, mock_asset_oracle)
    
    def test_monitor_initialization(self, arbitrage_monitor, yield_farming_monitor,
                                   lending_monitor, borrowing_monitor):
        """Test all specialized monitors initialize correctly."""
        monitors = [arbitrage_monitor, yield_farming_monitor, lending_monitor, borrowing_monitor]
        
        for monitor in monitors:
            assert monitor.config is not None
            assert monitor.asset_oracle is not None
            assert hasattr(monitor, 'position_type')
    
    @pytest.mark.asyncio
    async def test_arbitrage_monitor_fast_completion_check(self, arbitrage_monitor):
        """Test arbitrage monitor checks for fast completion."""
        # Create mock arbitrage position that's taking too long
        position = Mock()
        position.position_id = "arb_pos_123"
        position.position_type = PositionType.ARBITRAGE
        position.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)  # 10 minutes old
        position.status = "executing"
        position.expected_completion_time = 30  # Expected 30 seconds
        
        alerts = await arbitrage_monitor.check_position_health(position)
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == "execution_timeout"
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "10.0 minutes" in alerts[0].message
    
    @pytest.mark.asyncio
    async def test_yield_farming_monitor_impermanent_loss(self, yield_farming_monitor, mock_asset_oracle):
        """Test yield farming monitor detects impermanent loss."""
        # Mock asset prices for IL calculation
        mock_asset_oracle.get_price_usd.side_effect = lambda asset_id: {
            "ETH_MAINNET_WETH": 2500.0,  # ETH up 25% from $2000
            "ETH_MAINNET_USDC": 1.0      # USDC stable
        }.get(asset_id, 1.0)
        
        position = Mock()
        position.position_id = "yield_pos_123"
        position.position_type = PositionType.YIELD_FARMING
        position.created_at = datetime.now(timezone.utc) - timedelta(hours=24)
        position.metadata = {
            "pool_type": "uniswap_v3",
            "initial_eth_price": 2000.0,
            "initial_usdc_price": 1.0,
            "lp_token_amount": 1000.0
        }
        position.exposures = {
            "ETH_MAINNET_WETH": Mock(amount=Decimal("1.0")),
            "ETH_MAINNET_USDC": Mock(amount=Decimal("2000.0"))
        }
        
        alerts = await yield_farming_monitor.check_position_health(position)
        
        # Should detect impermanent loss above threshold
        il_alerts = [a for a in alerts if a.alert_type == "impermanent_loss"]
        assert len(il_alerts) >= 1
        assert il_alerts[0].severity in [AlertSeverity.WARNING, AlertSeverity.ERROR]
    
    @pytest.mark.asyncio
    async def test_lending_monitor_utilization_rate(self, lending_monitor):
        """Test lending monitor checks utilization rates."""
        position = Mock()
        position.position_id = "lend_pos_123"
        position.position_type = PositionType.LENDING
        position.created_at = datetime.now(timezone.utc) - timedelta(hours=12)
        position.metadata = {
            "protocol": "aave",
            "asset": "ETH_MAINNET_USDC",
            "utilization_rate": 0.95,  # 95% utilization - high
            "current_apr": 0.08
        }
        position.current_value_usd = 50000.0
        
        alerts = await lending_monitor.check_position_health(position)
        
        util_alerts = [a for a in alerts if a.alert_type == "utilization_rate_high"]
        assert len(util_alerts) == 1
        assert util_alerts[0].severity == AlertSeverity.WARNING
        assert "95.0%" in util_alerts[0].message
    
    @pytest.mark.asyncio
    async def test_borrowing_monitor_health_factor(self, borrowing_monitor):
        """Test borrowing monitor checks health factor."""
        position = Mock()
        position.position_id = "borrow_pos_123"
        position.position_type = PositionType.BORROWING
        position.created_at = datetime.now(timezone.utc) - timedelta(hours=6)
        position.metadata = {
            "protocol": "aave",
            "collateral_asset": "ETH_MAINNET_WETH",
            "debt_asset": "ETH_MAINNET_USDC",
            "health_factor": 1.25,  # Low health factor
            "liquidation_threshold": 1.0
        }
        position.current_value_usd = -25000.0  # Negative for debt
        
        alerts = await borrowing_monitor.check_position_health(position)
        
        health_alerts = [a for a in alerts if a.alert_type == "health_factor_low"]
        assert len(health_alerts) == 1
        assert health_alerts[0].severity == AlertSeverity.ERROR
        assert "1.25" in health_alerts[0].message
        assert "add collateral" in health_alerts[0].recommended_action.lower()
    
    @pytest.mark.asyncio
    async def test_borrowing_monitor_critical_health_factor(self, borrowing_monitor):
        """Test borrowing monitor detects critical health factor."""
        position = Mock()
        position.position_id = "borrow_pos_456"
        position.position_type = PositionType.BORROWING
        position.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        position.metadata = {
            "protocol": "compound",
            "health_factor": 1.02,  # Very close to liquidation
            "liquidation_threshold": 1.0
        }
        position.current_value_usd = -100000.0
        
        alerts = await borrowing_monitor.check_position_health(position)
        
        critical_alerts = [a for a in alerts if a.alert_type == "health_factor_critical"]
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL
        assert "1.02" in critical_alerts[0].message


class TestPositionHealthCalculations:
    """Test position health calculation methods."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Create test monitoring configuration."""
        return MonitoringConfig()
    
    @pytest.fixture
    def mock_asset_oracle(self):
        """Create mock asset oracle."""
        oracle = Mock(spec=AssetOracleBase)
        oracle.get_price_usd = AsyncMock(return_value=2000.0)
        return oracle
    
    @pytest.fixture
    def yield_monitor(self, monitoring_config, mock_asset_oracle):
        """Create YieldFarmingPositionMonitor for testing calculations."""
        return YieldFarmingPositionMonitor(monitoring_config, mock_asset_oracle)
    
    @pytest.mark.asyncio
    async def test_impermanent_loss_calculation(self, yield_monitor, mock_asset_oracle):
        """Test impermanent loss calculation accuracy."""
        # Set up price changes: ETH 2000->2500 (+25%), USDC stable
        mock_asset_oracle.get_price_usd.side_effect = lambda asset_id: {
            "ETH_MAINNET_WETH": 2500.0,
            "ETH_MAINNET_USDC": 1.0
        }.get(asset_id, 1.0)
        
        initial_prices = {"ETH_MAINNET_WETH": 2000.0, "ETH_MAINNET_USDC": 1.0}
        exposures = {
            "ETH_MAINNET_WETH": Mock(amount=Decimal("1.0")),
            "ETH_MAINNET_USDC": Mock(amount=Decimal("2000.0"))
        }
        
        il_percent = await yield_monitor._calculate_impermanent_loss(
            exposures, initial_prices
        )
        
        # For 25% price increase in 50/50 pool, IL should be around 0.6%
        assert 0.4 <= il_percent <= 0.8  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_health_factor_calculation(self, monitoring_config, mock_asset_oracle):
        """Test health factor calculation for borrowing positions."""
        borrowing_monitor = BorrowingPositionMonitor(monitoring_config, mock_asset_oracle)
        
        # Mock collateral and debt values
        collateral_value = 100000.0  # $100k collateral
        debt_value = 60000.0         # $60k debt
        liquidation_threshold = 0.75  # 75% LTV
        
        health_factor = borrowing_monitor._calculate_health_factor(
            collateral_value, debt_value, liquidation_threshold
        )
        
        # Health factor = (collateral * threshold) / debt
        # = (100000 * 0.75) / 60000 = 1.25
        assert abs(health_factor - 1.25) < 0.01
    
    def test_risk_level_classification(self, monitoring_config, mock_asset_oracle):
        """Test risk level classification logic."""
        borrowing_monitor = BorrowingPositionMonitor(monitoring_config, mock_asset_oracle)
        
        # Test different health factor risk levels
        assert borrowing_monitor._get_risk_level(2.5) == RiskLevel.LOW
        assert borrowing_monitor._get_risk_level(1.8) == RiskLevel.MEDIUM
        assert borrowing_monitor._get_risk_level(1.3) == RiskLevel.HIGH
        assert borrowing_monitor._get_risk_level(1.05) == RiskLevel.CRITICAL


if __name__ == "__main__":
    # Run basic functionality test
    print("🧪 Testing Position Monitor")
    print("=" * 50)
    
    # Test PositionAlert creation
    alert = PositionAlert(
        position_id="test_pos",
        position_type=PositionType.YIELD_FARMING,
        alert_type="test_alert",
        severity=AlertSeverity.WARNING,
        message="Test message",
        details={"test": "data"}
    )
    
    print(f"✅ PositionAlert created:")
    print(f"   - ID: {alert.position_id}")
    print(f"   - Type: {alert.position_type.value}")
    print(f"   - Severity: {alert.severity.value}")
    print(f"   - Timestamp: {alert.timestamp}")
    
    # Test MonitoringConfig
    config = MonitoringConfig()
    print(f"\n✅ MonitoringConfig defaults:")
    print(f"   - Check interval: {config.check_interval_seconds}s")
    print(f"   - High frequency: {config.high_frequency_interval_seconds}s")
    print(f"   - Max loss: {config.max_unrealized_loss_percent}%")
    print(f"   - IL threshold: {config.impermanent_loss_threshold}%")
    
    # Test alert dictionary conversion
    alert_dict = alert.to_dict()
    print(f"\n✅ Alert to dict conversion:")
    print(f"   - Keys: {list(alert_dict.keys())}")
    print(f"   - Timestamp format: {alert_dict['timestamp']}")
    
    print("\n✅ Position Monitor test passed!")